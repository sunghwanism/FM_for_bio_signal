import logging
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset
import torch.backends.cudnn as cudnn
import random

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)


class applewatch:

    def __init__(self, PATH, length):
        
        self.x_data = []
        self.y_data = []
        
        data = pd.read_csv(PATH)
            
        for k in range(int(len(data)/length)):
            front_idx = int(k*length)
            post_idx = int((k+1)*length)
            
            temp = data[front_idx:post_idx]                
            HR = temp["heart_rate"].to_numpy()
            activity = temp["steps"].to_numpy()
            stage = temp["psg_status"].to_numpy()[0].astype(int)
            
            if stage in [1,2]:
                stage = 1
                
            elif stage in [3,4]:
                stage = 2
            elif stage == 5:
                stage = 3
            
            # self.x_data.append(np.stack([x_move, y_move, z_move, HR, activity], axis=1))
            self.x_data.append(np.stack([HR, activity], axis=1))
            self.y_data.append(stage)
        
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        
        self.x_data = torch.FloatTensor(self.x_data)
        self.y_data = torch.FloatTensor(self.y_data).long()
        
        self.x_data = self.x_data.permute(0, 2, 1)
        
        print("x_data shape: ", self.x_data.shape)
        print("y_data shape: ", self.y_data.shape, "class_num", self.y_data.unique().shape[0])
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        
        return self.x_data[idx], self.y_data[idx]