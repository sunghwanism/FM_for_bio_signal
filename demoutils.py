import os
import sys
sys.path.append("./src")
sys.path.append("./src/foundation")
sys.path.append("./basemodel")

import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from foundation.data.Augmentaion import init_augmenter
from foundation.models.FOCALModules import FOCAL
from foundation.models.Backbone import DeepSense
from basemodel.DeepSense import DeepSense as BASEDEEPSENSE
from downstream.classifier import SleepStageClassifier
import foundation.args as args
import downstream.downargs as downargs
import baselineargs
import seaborn as sns

import matplotlib.pyplot as plt

device = torch.device('mps')

# aug_1 = init_augmenter("NoAugmenter", None).to(device)
# aug_2 = init_augmenter("NoAugmenter", None).to(device)


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
    
    

def get_predict_label_from_fm_classifier(model, downstream_model, dataloder, device):
    
    model.eval()
    
    pred_list = []
    labels = []
    
    for i, data in enumerate(dataloder):
        ecg, hr, _, sleep_stage = data
        ecg = ecg.to(device)
        hr = hr.to(device)
        sleep_stage = sleep_stage.to(device)
        
        aug_1_modal_1 = aug_1(ecg)
        aug_2_modal_1 = aug_2(ecg)
        
        aug_1_modal_2 = aug_1(hr)
        aug_2_modal_2 = aug_2(hr)
        
        mod_feature1, mod_feature2 = model(aug_1_modal_1, aug_1_modal_2, 
                                           aug_2_modal_1, aug_2_modal_2, proj_head=True, class_head=False)
        
        preds = downstream_model(mod_feature1, mod_feature2)
        pred_list.extend(preds.detch().cpu().numpy().squeeze())
        labels.extend(sleep_stage.detch().cpu().numpy().squeeze())
        
    return pred_list, labels




def get_predict_label_from_individual_model(individual_model, dataloder, device):
    
    individual_model.eval()
    
    preds = []
    labels = []
    
    for i, data in enumerate(dataloder):
        ecg, hr, _, sleep_stage = data
        
        ecg = ecg.to(device)
        hr = hr.to(device)
        sleep_stage = sleep_stage.to(device)

        pred = individual_model(ecg, hr, class_head=True, proj_head=True)
        
        pred = torch.argmax(pred, dim=1)
        
        preds.extend(pred.detach().cpu().numpy().squeeze())
        labels.extend(sleep_stage.detach().cpu().numpy().squeeze())
        
        
    return preds, labels


def Load_Foundation_Model(model_ckpt):
    pass


def predict_using_individual_model(subj_index):
    
    subj_test_data_path = f'pair_test_subj/subj_{subj_index}_test'
    subj_test_model_ckpt = f'models/individual_best/DeepSense_{subj_index}.pth'
    
    ckpt = torch.load(subj_test_model_ckpt, map_location=device)
    
    baselineargs.data_config = ckpt['data_config']
    baselineargs.trainer_config = ckpt['trainer_config']
    model_param = ckpt['state_dict']
    
    individual_model = BASEDEEPSENSE(baselineargs).to(device)
    individual_model.load_state_dict(model_param)

    test_dataset = MESAPairDataset(file_path=subj_test_data_path,
                                modalities=baselineargs.data_config['modalities'],
                                subject_idx=baselineargs.data_config['subject_key'],
                                stage=baselineargs.data_config['label_key'])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=128,
                                                shuffle=False,
                                                num_workers=4)
    
    predict, labels = get_predict_label_from_individual_model(individual_model, test_loader, device)
    
    # Calculate metrics
    acc = accuracy_score(labels, predict)
    f1score = f1_score(labels, predict, average='macro')
    conf_matrix = confusion_matrix(labels, predict)

    print(f'Accuracy: {round(acc, 3)}')
    print(f'F1-score (Macro): {round(f1score,3)}')
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    return acc, f1score

def predict_using_fm_classifier(subj_index, model_index):
    
    subj_test_data_path = f'pair_test_subj/subj_{subj_index}_test'
    if subj_index == '0560':
        subj_test_model_ckpt = f'models/ckpt_down/{subj_index}/FM_based_classfier_0140.pth'
    elif subj_index == "0565":
        subj_test_model_ckpt = f'models/ckpt_down/{subj_index}/FM_based_classfier_0143.pth'
    elif subj_index == "0583":
        subj_test_model_ckpt = f'models/ckpt_down/{subj_index}/FM_based_classfier_0143.pth'
    elif subj_index == "0558":
        subj_test_model_ckpt = f'models/ckpt_down/{subj_index}/FM_based_classfier_0143.pth'
    elif subj_index == "0571":
        subj_test_model_ckpt = f'models/ckpt_down/{subj_index}/FM_based_classfier_0143.pth'
    else:
        ValueError("Invalid subject index")
    
    model_config = torch.load(subj_test_model_ckpt, map_location=device)
    
    args['data_config'] = model_config['focal_data_config']
    args['trainer_config'] = model_config['focal_trainer_config']
    args['focal_config'] = model_config['focal_config']
    
    downargs['downstream_config'] = model_config['downstream_config']
    
    
    backbone = DeepSense(args).to(device)
    focal_model = FOCAL(args, backbone).to(device)
    focal_model.load_state_dict(model_config["focal_state_dict"], strict=False)
    
    downstream_model = SleepStageClassifier(args).to(device)
    downstream_model.load_state_dict(model_config["down_state_dict"], strict=False)
    

    test_dataset = MESAPairDataset(file_path=subj_test_data_path,
                                modalities=args.data_config['modalities'],
                                subject_idx=args.data_config['subject_key'],
                                stage=args.data_config['label_key'])

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.trainer_config['batch_size']//4,
                                                shuffle=False,
                                                num_workers=2)
    
    predict, labels = get_predict_label_from_fm_classifier(model, downstream_model, test_loader, device)
    
    # Calculate metrics
    acc = accuracy_score(labels, predict)
    f1score = f1_score(labels, predict, average='macro')
    conf_matrix = confusion_matrix(labels, predict)

    print(f'Accuracy: {round(acc, 3)}')
    print(f'F1-score (Macro): {round(f1score,3)}')
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    
def loss_acc_plotting(subj_index, model_index='0140'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    
    ax[0].plot(np.load(f"./logs/logs_down/{subj_index}/FM_based_classfier_{model_index}.npz")["arr_0"][0], label='train')
    ax[0].plot(np.load(f"./logs/logs_down/{subj_index}/FM_based_classfier_{model_index}.npz")["arr_0"][2], label='val')
    
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    ax[1].plot(np.load(f"./logs/logs_down/{subj_index}/FM_based_classfier_{model_index}.npz")["arr_0"][1], label='train')
    ax[1].plot(np.load(f"./logs/logs_down/{subj_index}/FM_based_classfier_{model_index}.npz")["arr_0"][3], label='val')
    
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    plt.show()