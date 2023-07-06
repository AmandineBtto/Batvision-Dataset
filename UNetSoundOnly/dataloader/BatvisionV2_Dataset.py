import os
import torch
import pandas as pd
import torchaudio
import cv2
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np

from .utils_dataset import get_transform

class BatvisionV2Dataset(Dataset):
    
    def __init__(self, cfg, annotation_file, location_blacklist=None):
   
        self.cfg = cfg
        self.root_dir = cfg.dataset.dataset_dir
        self.audio_format = cfg.dataset.audio_format

        location_list = os.listdir(self.root_dir)
        if location_blacklist:
            location_list = [location for location in location_list if location not in location_blacklist]
        location_csv_paths = [os.path.join(self.root_dir, location, annotation_file) for location in location_list]
                
        self.instances = []
        
        for location_csv in location_csv_paths:
            self.instances.append(pd.read_csv(location_csv))
            
        self.instances = pd.concat(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        # Access instance 
        instance = self.instances.iloc[idx]
        
        # Load path
        depth_path = os.path.join(self.root_dir,instance['depth path'],instance['depth file name'])
        audio_path = os.path.join(self.root_dir,instance['audio path'],instance['audio file name'])

        ## Depth
        # Load depth map
        depth = np.load(depth_path).astype(np.float32)
        depth = depth / 1000 # to go from mm to m
        if self.cfg.dataset.max_depth:
            depth[depth > self.cfg.dataset.max_depth] = self.cfg.dataset.max_depth 
        # Transform
        depth_transform = get_transform(self.cfg, convert =  True, depth_norm = self.cfg.dataset.depth_norm)
        gt_depth = depth_transform(depth)
        
        ## Audio 
        # Load audio binaural waveform
        waveform, sr = torchaudio.load(audio_path)
        # STFT parameters for full length audio
        win_length = 200 
        n_fft = 400
        hop_length = 100

        # Cut audio to fit max depth
        if self.cfg.dataset.max_depth:
            cut = int((2*self.cfg.dataset.max_depth / 340) * sr)
            waveform = waveform[:,:cut]
            # Update STFT parameters 
            win_length = 64
            n_fft = 512
            hop_length=64//4

        # Process sound
        if 'spectrogram' in self.audio_format:
            if 'mel' in self.audio_format:
                spec = self._get_melspectrogram(waveform, n_fft = n_fft, power = 1.0, win_length = win_length)
            else:
                spec = self._get_spectrogram(waveform, n_fft = n_fft, power = 1.0, win_length = win_length, hop_length =  hop_length)
            spec_transform =  get_transform(self.cfg, convert = False) # convert False because already a tensor
            audio2return = spec_transform(spec)
        elif 'waveform' in self.audio_format:
            audio2return = waveform
        
        return audio2return, gt_depth
    
    # audio transformation: spectrogram
    def _get_spectrogram(self, waveform, n_fft = 400, power = 1.0, win_length = 400, hop_length=100): 

        spectrogram = T.Spectrogram(
          n_fft=n_fft,
          win_length=win_length,
          power=power,
          hop_length=hop_length,
        )
        #db = T.AmplitudeToDB(stype = 'magnitude')
        return spectrogram(waveform)
    
    # audio transformation: mel spectrogram
    def _get_melspectrogram(self, waveform, n_fft = 400, power = 1.0, win_length = 400, f_min = 20.0, f_max = 20000.0): 

        melspectrogram = T.MelSpectrogram(sample_rate = 44100, 
          n_fft=n_fft,
          win_length=win_length,
          power=power,
          f_min = f_min, 
          f_max = f_max,
          n_mels = 32, 
        )
        return melspectrogram(waveform)
    