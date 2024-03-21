import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd



class MESAPairDataset(Dataset):
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor_img = self.data[idx]
        
        if self.transform:
            pos_img = self.transform(anchor_img)

        else:
            pos_img = anchor_img
        
        while True:
            neg_idx = np.random.randint(0, len(self.data))
            if neg_idx != idx:
                break
        neg_img = self.data[neg_idx]
        
        return anchor_img, pos_img, neg_img
