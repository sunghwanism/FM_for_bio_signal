import torch
import torch.nn as nn
import numpy as np

from random import random


def init_augmenter(args):
    if args.augmenter == "GaussianNoise":
        print("Loading GaussianNoise augmenter...")
        return GaussianNoise(max_noise_std=args.max_noise_std)
    elif args.augmenter == "AmplitudeScale":
        print("Loading AmplitudeScale augmenter...")
        return AmplitudeScale(amplitude_scale=args.amplitude_scale)
    elif args.augmenter == "NoAugmenter":
        print("Loading NoAugmenter augmenter...")
        return NoAugmenter(args)
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
        
        noise_stds = torch.rand(loc_inputs.size(0), 1, 1) * self.max_noise_std
        noise = torch.randn_like(loc_inputs) * noise_stds

        return loc_inputs + noise

    
class AmplitudeScale(nn.Module):
    def __init__(self, amplitude_scale=0.5):
        super().__init__()
        self.amplitude_scale = amplitude_scale

    def forward(self, loc_inputs):
        batch_size = loc_inputs.shape[0]
        amplitude_scale = torch.rand(batch_size) * self.amplitude_scale + 1
        loc_inputs = loc_inputs * amplitude_scale.view(-1, 1)
        
        return  loc_inputs