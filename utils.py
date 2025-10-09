import os
import argparse
import json
import os
import argparse
import json
import numpy as np
from scipy.io.wavfile import read
import torch
import torchaudio

def normalize_audio(audio, target_peak=0.95):
    # Peak normalization to target_peak level (usually < 1.0)
    peak = audio.abs().max()
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio

def save_audio(audio,sample_rate,name="",root="./sample/"):
    audio = normalize_audio(audio)
    torchaudio.save(f'{root}/{name}.wav', audio, sample_rate)

def model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

def get_hparams():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  
  args = parser.parse_args()

  config_path = args.config
 
  with open(config_path, "r") as f:
    data = f.read()

  config = json.loads(data)
  hparams = HParams(**config)
  return hparams

def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text

def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      print("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
    print("Loaded checkpoint '{}' (iteration {})" .format(checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration

def resample(waveforms, orig_sr=22050, target_sr=48000):
    assert waveforms.dim() == 3 and waveforms.shape[1] == 1, "Input must be (B, 1, T)"
    
    B = waveforms.shape[0]
    device = waveforms.device
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr).to(device)

    resampled_list = []
    max_len = 0

    for i in range(B):
        resampled = resampler(waveforms[i])  # shape: (1, T')
        resampled_list.append(resampled)
        max_len = max(max_len, resampled.shape[-1])
    
    # Pad to max length
    padded_batch = torch.zeros((B, 1, max_len), dtype=torch.float32, device=device)
    for i in range(B):
        padded_batch[i, 0, :resampled_list[i].shape[-1]] = resampled_list[i]

    return padded_batch