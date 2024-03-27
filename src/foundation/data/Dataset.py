import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MESAPairDataset(Dataset):
    def __init__(self, file_path, modalities=['ecg', 'hr'], subject_idx='subject_idx', stage='stage'):
        super(MESAPairDataset, self).__init__()
        self.root_dir = file_path
        self.files = os.listdir(file_path)
        self.modalities = modalities
        self.subject_idx = subject_idx
        self.stage = stage
        
        
        for i, datafile in enumerate(tqdm(self.files)):
            data = np.load(os.path.join(self.root_dir, datafile)) # numpy file on each sample (segments)

            if i == 0:
                self.modality_1 = data[self.modalities[0]]
                self.modality_2 = data[self.modalities[1]]
                self.subject_id = [data[self.subject_idx]]
                self.sleep_stage = [data[self.stage]]

            else:
                self.modality_1 = np.concatenate([self.modality_1, data[self.modalities[0]]])
                self.modality_2 = np.concatenate([self.modality_2, data[self.modalities[1]]])
                self.subject_id.append(data[self.subject_idx])
                self.sleep_stage.append(data[self.stage])

        self.sleep_stage = np.array(self.sleep_stage)
        self.subject_id = np.array(self.subject_id)
                
    def __len__(self):

        return len(self.subject_idx)


    def __getitem__(self, idx):
        
       self.modality_1[idx]
       self.modality_2[idx]
       self.subject_id[idx]
       self.sleep_stage[idx]
       
       self.modality_1 = torch.FloatTensor(self.modality_1)
       self.modality_2 = torch.FloatTensor(self.modality_2)
       self.subject_id = torch.Tensor(self.subject_id).long()
       self.sleep_stage = torch.Tensor(self.sleep_stage).long()
       
       sample = [self.modality_1, self.modality_2, self.subject_id, self.sleep_stage]
       
       return sample
    