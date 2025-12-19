from data_utils import Dataset, DistributedBucketSampler,TextAudioSpeakerCollate
from torch.utils.data import DataLoader
from tts.model import ParaStyleTTS,MultiPeriodDiscriminator
import torch
import utils
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os 
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from tts.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import tts.commons as commons
from tts.loss import feature_loss, discriminator_loss, generator_loss, kl_loss
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sentence_transformers import SentenceTransformer
from g2p import all_ipa_phoneme


import os 
def load_dataset():
    dataset = pd.read_csv("fileloader/example_processed.csv",delimiter="\t")
    seed = 42
    # shuffle the dataset
    dataset = dataset.sample(frac=1,random_state=seed).reset_index(drop=True)
    print(dataset)
    print("Total number of speech samples are: ", len(dataset))
    return dataset

global_step = 0
def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = Dataset(load_dataset(), hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        #100~= 1.16s, 1300 ~= maximum 15s
        [32,300,400,500,600,700,800,900,1000,1100,1200,1300],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextAudioSpeakerCollate()
    loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)

    n_style = 8
    net_g = ParaStyleTTS(len(all_ipa_phoneme),n_style,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(net_g.parameters(),hps.train.learning_rate, betas=hps.train.betas,eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(),hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    try:
        # this is to load the latest checkpoint automatically
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(loader)
    except:
        # if no checkpoint found, start from scratch
        epoch_str = 1
        global_step = 0


    net_g = DDP(net_g, device_ids=[rank],find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank],find_unused_parameters=True)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scaler = GradScaler(enabled=hps.train.fp16_run)
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank==0:
            # only log on rank 0
            train(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, loader, logger,writer)
        else: 
            train(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, loader, None, None)

        scheduler_g.step()
        scheduler_d.step()

def train(rank, epoch, hps, nets, optims, schedulers, scaler, loader, logger,writer):
    global global_step
    net_g, net_d = nets
    optim_g, optim_d = optims
    # this to load the text encoder to extract textual style embedding
    style_model = SentenceTransformer('all-mpnet-base-v2')
    scheduler_g, scheduler_d = schedulers

    loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths,style, captions, spec, spec_lengths, y, y_lengths) in enumerate(tqdm(loader)):
        if x.dtype == torch.complex64: x = x.to()
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        style = style.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        abs_style = style_model.encode(captions, convert_to_tensor=True,show_progress_bar=False).cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x,style,x_lengths,spec,spec_lengths,abs_style)

            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate,hps.data.hop_length, 
                hps.data.win_length, 
                hps.data.mel_fmin, 
                hps.data.mel_fmax
            )

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            
        with autocast(enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank==0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                epoch,
                100. * batch_idx / len(loader)))
                #loss is: discriminator-loss, generation_loss, feature_matching loss, mel reconstruction loss, duration loss, kl loss, step and lr
                logger.info([x.item() for x in losses] + [global_step, lr])
                
                scalar_dict = {"loss/generator/total": loss_gen_all, "loss/discriminator/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/feature_match": loss_fm, "loss/g/mel_reconstruction": loss_mel, "loss/g/duration": loss_dur, "loss/g/kl_divergence": loss_kl})

                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

                utils.summarize(writer=writer,global_step=global_step,scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        global_step += 1
    
    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    
    n_gpus = torch.cuda.device_count()
    print("Number of GPUs: ", n_gpus)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8000'
    n_gpus = torch.cuda.device_count()
    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))

if __name__ == '__main__':
    main()