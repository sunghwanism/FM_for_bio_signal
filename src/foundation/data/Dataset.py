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

        return len(self.sleep_stage)


    def __getitem__(self, idx):
        
       batch_modal_1 = self.modality_1[idx]
       batch_modal_2 = self.modality_2[idx]
       batch_subj_id = self.subject_id[idx]
       batch_stage = self.sleep_stage[idx]
       
       self.batch_modal_1 = torch.tensor(batch_modal_1, dtype=torch.float)
       self.batch_modal_2 = torch.tensor(batch_modal_2, dtype=torch.float)
       self.batch_subj_id = torch.tensor(batch_subj_id, dtype=torch.long)
       self.batch_stage = torch.tensor(batch_stage, dtype=torch.long)
       
       sample = [self.batch_modal_1, self.batch_modal_2, self.batch_subj_id, self.batch_stage]
       
       return sample