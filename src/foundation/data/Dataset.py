import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# class MESAPairDataset(Dataset):
#     def __init__(self, file_path, modalities=['ecg', 'hr'], subject_idx='subject_idx', stage='stage'):
#         super(MESAPairDataset, self).__init__()
#         self.root_dir = file_path
#         self.files = os.listdir(file_path)
#         self.modalities = modalities
#         self.subject_idx = subject_idx
#         self.stage = stage
        
#     def __len__(self):

#         return len(self.files)


#     def __getitem__(self, idx):
#         data = np.load(os.path.join(self.root_dir, self.files[idx])) # numpy file on each sample (segments)
        
#         self.modality_1 = torch.tensor(data[self.modalities[0]], dtype=torch.float)
#         self.modality_2 = torch.tensor(data[self.modalities[1]], dtype=torch.float)
#         self.subject_id = torch.tensor(data[self.subject_idx], dtype=torch.long)
#         self.sleep_stage = torch.tensor(data[self.stage], dtype=torch.long)
        
#         sample = [self.modality_1, self.modality_2, self.subject_id, self.sleep_stage]
        
#         return sample
    
    
    
class MESAPairDataset(Dataset):
    def __init__(self, file_path, modalities=['ecg', 'hr'], subject_idx='subject_idx', stage='stage'):
        super(MESAPairDataset, self).__init__()
        self.root_dir = file_path
        self.files = os.listdir(file_path)
        self.modalities = modalities
        self.subject_idx = subject_idx
        self.stage = stage
        
    def __len__(self):

        return len(self.files)


    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root_dir, self.files[idx])) # numpy file on each sample (segments)
        
        self.modality_1 = torch.tensor(data[self.modalities[0]], dtype=torch.float)
        self.modality_2 = torch.tensor(data[self.modalities[1]], dtype=torch.float)
        self.subject_id = torch.tensor(data[self.subject_idx], dtype=torch.long)
        stage = data[self.stage]
        
        #if self.num_outputs == 4:
        if stage in [1, 2]:
            stage = 1
        elif stage in [3, 4]:
            stage = 2
        elif stage == 5:
            stage = 3
        
        # elif self.num_outputs == 2:
        #     if labels in [1, 2, 3, 4, 5]:
        #         labels = 1
        self.sleep_stage = torch.tensor(stage, dtype=torch.long)

        sample = [self.modality_1, self.modality_2, self.subject_id, self.sleep_stage]
        
        return sample