import torch
import torch.nn as nn
import numpy as np

from random import random


def init_augmenter(augmenter_name, augmenter_config):
    # augmenter_name = args.data_config['augmenter']
    # augmenter_config = args.data_config['augmenter_config'].get(augmenter_name, {})

    if augmenter_name == "GaussianNoise":
        print("Loading GaussianNoise augmenter...")
        return GaussianNoise(**augmenter_config)
    elif augmenter_name == "AmplitudeScale":
        print("Loading AmplitudeScale augmenter...")
        return AmplitudeScale(**augmenter_config)
    elif augmenter_name == "NoAugmenter":
        print("Loading NoAugmenter augmenter...")
        return NoAugmenter()
    else:
        raise NotImplementedError

class NoAugmenter(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        
    def forward(self, loc_inputs, labels=None):
        
        return loc_inputs, None, labels
    
    
class GaussianNoise(nn.Module):
    def __init__(self, max_noise_std=0.1):
        super().__init__()
        self.max_noise_std = max_noise_std
        
    def forward(self, loc_inputs):
        device = loc_inputs.device
        noise_stds = torch.rand(loc_inputs.size(0), 1, device=device) * self.max_noise_std
        noise = torch.randn_like(loc_inputs) * noise_stds

        return loc_inputs + noise

    
class AmplitudeScale(nn.Module):
    def __init__(self, amplitude_scale=0.5):
        super().__init__()
        self.amplitude_scale = amplitude_scale

    def forward(self, loc_inputs):
        device = loc_inputs.device
        batch_size = loc_inputs.shape[0]
        amplitude_scale = torch.rand(batch_size, device=device) * self.amplitude_scale + 1
        loc_inputs = loc_inputs * amplitude_scale.view(-1, 1)
        
        return  loc_inputs