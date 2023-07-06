
import torch
import torchvision.transforms as transforms

import os, glob
import pandas as pd
import numpy as np


def get_transform(cfg, convert =  False, depth_norm = False):
    # Create list of transform to apply to data
    transform_list = []

    if convert:
        # Convert data to Tensor type
        transform_list += [transforms.ToTensor()]

    if 'resize' in cfg.dataset.preprocess:
        # Resize
        transform_list.append(transforms.Resize((cfg.dataset.images_size,cfg.dataset.images_size)))

    if depth_norm:
        # MinMax depth normalization
        max_depth_dataset = cfg.dataset.max_depth 
        min_depth_dataset = 0.0
        transform_list += [MinMaxNorm(min = min_depth_dataset, max = max_depth_dataset)]

    return transforms.Compose(transform_list)

# Custom transforms 
class MinMaxNorm(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        assert isinstance(min, (float, tuple)) and isinstance(max, (float, tuple))
        self.min = torch.tensor(min)
        self.max = torch.tensor(max)
        
    def forward(self, tensor):
        if tensor.shape[0] == 2:
            norm_tensor_c0 = (tensor[0,...] - self.min[0]) / (self.max[0] - self.min[0])
            norm_tensor_c1 = (tensor[1,...] - self.min[1]) / (self.max[1] - self.min[1])
            norm_tensor = torch.concatenate([norm_tensor_c0.unsqueeze(0), norm_tensor_c1.unsqueeze(0)], dim = 0)

        else:
            norm_tensor = (tensor - self.min) / (self.max - self.min)

        return norm_tensor


