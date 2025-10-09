from tts.model import ParaStyleTTS
import utils
import torch
from utils import save_audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from g2p import all_ipa_phoneme,mix_to_ipa,ipa_to_idx,mixed_sentence_to_ipa
hps = utils.get_hparams()

n_tones = 8
tts = ParaStyleTTS(len(all_ipa_phoneme),n_tones,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
utils.load_checkpoint(f"./ckp/ParaStyleTTS.pth", tts, None)
tts.eval()


from sentence_transformers import SentenceTransformer
style_encoder = SentenceTransformer('all-mpnet-base-v2',trust_remote_code=True)

zh = "太开心了,终于又见到你了!".lower()
#get phoneme and tone
phon,tone = mix_to_ipa(zh)
phon_index = ipa_to_idx(phon)
phon_index = torch.tensor([phon_index]).to(device)  
tone = torch.tensor([tone]).to(device)
src_lens = torch.tensor([phon_index.shape[-1]]).to(device)
style_prompt = [f"A child female speaking Chinese with neutral emotion"]
style_embed = style_encoder.encode(style_prompt, convert_to_tensor=True,show_progress_bar=False).to(device)
wave,_,_,_ = tts.infer(phon_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0,noise_scale_w=0,ab_style=style_embed)
save_audio(wave[0].cpu().detach(), 22050, f"ch","./sample/")

en = "I'm so excited to see you again!".lower()
phon,tone = mix_to_ipa(en)
phon_index = ipa_to_idx(phon)
phon_index = torch.tensor([phon_index]).to(device)  
tone = torch.tensor([tone]).to(device)
src_lens = torch.tensor([phon_index.shape[-1]]).to(device)
style_prompt = [f"A adult male speaking English with happy emotion"]
style_embed = style_encoder.encode(style_prompt, convert_to_tensor=True,show_progress_bar=False).to(device)
wave,_,_,_ = tts.infer(phon_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0,noise_scale_w=0,ab_style=style_embed)
save_audio(wave[0].cpu().detach(), 22050, f"en","./sample/")

#ParaStyleTTS supports code-switched speech generation within a single speech.
mix = "这个weekend我想去city的shopping centre逛逛,然后顺便喝个coffee"
phon,tone = mixed_sentence_to_ipa(mix)
phon_index = ipa_to_idx(phon)
phon_index = torch.tensor([phon_index]).to(device)  
tone = torch.tensor([tone]).to(device)
src_lens = torch.tensor([phon_index.shape[-1]]).to(device)
style_prompt = [f"A adult female speaking English with happy emotion"]
style_embed = style_encoder.encode(style_prompt, convert_to_tensor=True,show_progress_bar=False).to(device)
wave,_,_,_ = tts.infer(phon_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0,noise_scale_w=0,ab_style=style_embed)
save_audio(wave[0].cpu().detach(), 22050, f"mix_en","./sample/")

#Language prompt is also effecting the accent of the generate speech
style_prompt = [f"A adult female speaking Chinese with happy emotion"]
style_embed = style_encoder.encode(style_prompt, convert_to_tensor=True,show_progress_bar=False).to(device)
wave,_,_,_ = tts.infer(phon_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0,noise_scale_w=0,ab_style=style_embed)
save_audio(wave[0].cpu().detach(), 22050, f"mix_ch","./sample/")