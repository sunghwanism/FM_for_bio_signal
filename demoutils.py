import os

import torch
from torch.utils.data import Dataset
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


class MESADataset(Dataset):
    def __init__(self, file_path, modalities=['ecg', 'hr'], subject_idx='subject_idx', stage='stage'):
        super(MESADataset, self).__init__()
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


def get_predict_lable_from_dataloader(model, dataloder, device, criterion):
    
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    for i, data in enumerate(dataloder):
        ecg, hr, _, sleep_stage = data
        ecg = ecg.to(device)
        hr = hr.to(device)
        sleep_stage = sleep_stage.to(device)
        
        output = model(ecg, hr, class_head=True, proj_head=True)
        loss = criterion(output, sleep_stage)
        
        total_loss += loss.item()
        total_correct += torch.sum(torch.argmax(output, dim=1) == sleep_stage).item()
        total_samples += sleep_stage.size(0)
        
    return total_correct / total_samples, total_loss / (i+1)



def Load_Foundation_Model(model_ckpt):
    pass


def predict_using_individual_model_mesa(subj_index):
    
    
    subj_test_data_path = f'pair_test_subj/subj_{subj_index}_test'
    
    # model은 deepsense
    
    
    predict, labels = get_predict_lable_from_dataloader(model, dataloder, device, criterion)
    
    
    # Accuracy
    # F1-score macro
    # confusion matrix
    
    
    print(acc)
    print(f1score)
    