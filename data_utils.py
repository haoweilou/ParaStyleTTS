import time
import os
import random
import numpy as np
import torch
import torch.utils.data

from tts.mel_processing import spectrogram_torch
from utils import load_wav_to_torch
import pandas as pd
import ast
import torchaudio

ipa_pho_dict = {'EMPTY': 0, 'a': 1, 'an': 2, 'aɪ': 3, 'aʊ': 4, 'b': 5, 'd': 6, 'dʒ': 7, 'eɪ': 8, 'f': 9, 'g': 10, 'h': 11, 'i': 12, 'in': 13, 'iŋ': 14, 'j': 15, 'ja': 16, 'jan': 17, 'jaʊ': 18, 'je': 19, 'joʊ': 20, 'jɑŋ': 21, 'k': 22, 'kʰ': 23, 'l': 24, 'm': 25, 'n': 26, 'o': 27, 'oʊ': 28, 'p': 29, 'pʰ': 30, 's': 31, 't': 32, 'ts': 33, 'tsʰ': 34, 'tɕ': 35, 'tɕʰ': 36, 'tʃ': 37, 'tʰ': 38, 'u': 39, 'v': 40, 'w': 41, 'wa': 42, 'wan': 43, 'waɪ': 44, 'weɪ': 45, 'wo': 46, 'wɑŋ': 47, 'wən': 48, 'x': 49, 'y': 50, 'yan': 51, 'yn': 52, 'yɛ': 53, 'z': 54, 'æ': 55, 'ð': 56, 'ŋ': 57, 'ɑ': 58, 'ɑŋ': 59, 'ɔ': 60, 'ɔɪ': 61, 'ɕ': 62, 'ən': 63, 'əŋ': 64, 'ɚ': 65, 'ɛ': 66, 'ɝ': 67, 'ɤ': 68, 'ɪ': 69, 'ɹ': 70, 'ɻ': 71, 'ʂ': 72, 'ʃ': 73, 'ʈʂ': 74, 'ʈʂʰ': 75, 'ʊ': 76, 'ʊŋ': 77, 'ʌ': 78, 'ʒ': 79, 'θ': 80, '|': 81, '&': 82, 'START': 83, 'END': 84}
# & is used for phrase seperatation, | is used for word seperation


def resample_audio(audio_tensor, orig_sample_rate, new_sample_rate):
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate)
    return resampler(audio_tensor)

class Dataset(torch.utils.data.Dataset):
    """
        1) loads audio, phoneme, tone, speaking style
        2) computes spectrograms from audio files.
    """
    def __init__(self, data_path, hparams):
        if type(data_path) == str:
            self.meta_data = pd.read_csv(data_path,sep="\t")
        else:
            self.meta_data = data_path
            
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length

        self.meta_data = self.meta_data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 512)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim * sample_rate) * 22050= file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        dataset = []
        lengths = []
        srs = []
        for row in self.meta_data.itertuples(index=False):
            ipa, style = ast.literal_eval(row.ipa), ast.literal_eval(row.style)
            ipa = [ipa_pho_dict[i] for i in ipa]#convert ipa to index
            assert len(ipa) == len(style)
            caption = row.caption
            if self.min_text_len <= len(row.ipa) and len(row.ipa) <= self.max_text_len:
                dataset.append([row.file_path, ipa, style, caption])
                spec_len = os.path.getsize(row.file_path) * 22050 // (2 * self.hop_length * 22050)
                lengths.append(spec_len)
            srs.append(22050)

        self.dataset = dataset
        self.lengths = lengths
        self.sample_rates = srs
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        file_path, ipa, style, caption = self.dataset[index]#caption is just a string
        ipa = torch.LongTensor(ipa)
        style = torch.LongTensor(style)
        sample_rate = self.sample_rates[index]
        spec, audio = self.get_audio(file_path,sample_rate=sample_rate)
        return spec, audio, ipa, style, caption

    def get_audio(self, filename,sample_rate=22050):
        audio, _ = load_wav_to_torch(filename)
        # print(max(audio),min(audio))
        if max(audio) >= 10:
            audio_norm = audio / self.max_wav_value#-1 to 1
        else: 
            audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec")
        if os.path.exists(spec_filename):
            # print("Loading from cache:", spec_filename)
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            # torch.save(spec, spec_filename)#only for nci gadi, to reduce the iusage
        return spec, audio_norm

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [spec, wav, ipa, style, emotion, language, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[2]) for x in batch])
        max_spec_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        text_padded = torch.LongTensor(len(batch), max_text_len)
        style_padded = torch.LongTensor(len(batch), max_text_len)

        captions = []

        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        style_padded.zero_()

        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            # batch: [spec, audio, ipa, style, style_embed]

            text = row[2]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
            
            style = row[3]
            style_padded[i, :text.size(0)] = style

            caption = row[4]
            captions.append(caption)

            spec = row[0]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[1]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, style_padded, captions, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, style_padded, captions, spec_padded, spec_lengths, wav_padded, wav_lengths