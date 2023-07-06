import os
import torch
import pandas as pd
import numpy as np
import torchaudio
import cv2
import torchaudio.transforms as T
from torch.utils.data import Dataset

from .utils_dataset import get_transform


class BatvisionV1Dataset(Dataset):
    
    def __init__(self, cfg, annotation_file):
        
        self.cfg = cfg
        self.root_dir = cfg.dataset.dataset_dir
        self.audio_format = cfg.dataset.audio_format
        self.instances = pd.read_csv(os.path.join(self.root_dir, annotation_file))
            
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        # Access instance 
        instance = self.instances.iloc[idx]
        
        # Load path
        depth_path = os.path.join(self.root_dir,instance['depth path'])
        audio_path_left = os.path.join(self.root_dir,instance['audio path left'])
        audio_path_right = os.path.join(self.root_dir,instance['audio path right'])
        
        ## Depth
        # Load depth map
        depth = np.load(depth_path)

        # Set nan value to 0
        depth = np.nan_to_num(depth)
        depth[depth == -np.inf] = 0
        depth[depth == np.inf] = 0
        
        depth = depth / 1000 # to go from mm to m
        depth[depth > self.cfg.dataset.max_depth] = self.cfg.dataset.max_depth 
        depth[depth < 0.0] = 0.0
        
        # Transform 
        depth_transform = get_transform(self.cfg, convert =  True, depth_norm = self.cfg.dataset.depth_norm)
        gt_depth = depth_transform(depth)
        
        ## Audio
        # Load audio binaural waveform
        waveform_left = np.load(audio_path_left).astype(np.float32)
        waveform_right = np.load(audio_path_right).astype(np.float32)
        waveform = torch.from_numpy(np.stack((waveform_left,waveform_right)))
        
        # Transform audio according to the desired format
        if 'spectrogram' in self.audio_format:
            spec = self._get_spectrogram(waveform, n_fft = 512, power = 1.0, win_length = 64, hop_length=64//4)  # parameters from the original batvision paper
            spec_transform =  get_transform(self.cfg, convert = False) # convert=False because already a tensor
            audio2return = spec_transform(spec)
        
        elif 'waveform' in self.audio_format:
            audio2return = waveform

        return audio2return, gt_depth

    # audio transformation
    def _get_spectrogram(self, waveform, n_fft = 400, power = 1.0, win_length = 400, hop_length=100): 

        spectrogram = T.Spectrogram(
          n_fft=n_fft,
          win_length=win_length,
          power=power,
          hop_length=hop_length
        )
        #db = T.AmplitudeToDB(stype = 'magnitude') # better results without dB conversion
        return spectrogram(waveform)
    