import os
import sys
sys.path.append('./')

import args
import argparse
import logging
from tqdm import tqdm

import torch
from models.AdversarialModel import AdversarialModel
from models.FOCALModules import FOCAL
from models.loss import FOCALLoss
from models.Backbone import DeepSense

from data.Dataset import MESAPairDataset
import datetime


from data.Augmentaion import init_augmenter

def train_SA_Focal(train_loader, valid_loader, model, advs_model, 
                   optimizer, advs_optimizer, focal_loss_fn, device, args):
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    trainer_config = args.trainer_config
    
    aug_1_name = args.data_config['augmentation'][0]
    aug_1_config = args.data_config['augmenter_config'].get(aug_1_name, {})
    aug_2_name = args.data_config['augmentation'][1]
    aug_2_config = args.data_config['augmenter_config'].get(aug_2_name, {})
    
    aug_1 = init_augmenter(aug_1_name, aug_1_config)
    aug_2 = init_augmenter(aug_2_name, aug_2_config)
    
    model.train()
    best_val_loss = float('inf')
    
    for ep in tqdm(range(trainer_config['epochs'])):
        running_advs_train_loss = 0
        focal_train_loss = 0
        model.train()
        advs_model.train()
        focal_loss_fn.train()
        
        for raw_modal_1, raw_modal_2, subj_label, sleep_label in train_loader:
            raw_modal_1, raw_modal_2, subj_label, sleep_label = raw_modal_1.to(device), raw_modal_2.to(device), subj_label.to(device), sleep_label.to(device) # [B, 30], [B, 30*256], [B, 1]
            
            aug_1_modal_1 = aug_1(raw_modal_1)
            aug_2_modal_1 = aug_2(raw_modal_1)
            
            aug_1_modal_2 = aug_1(raw_modal_2)
            aug_2_modal_2 = aug_2(raw_modal_2)
            
            # For updating the only advs_model (classifier)
            for param in model.parameters():
                param.requires_grad = False
            for param in advs_model.parameters():
                param.requires_grad = True
                
            advs_optimizer.zero_grad()
            
            # Using Encoder for classify the subject
            enc_feature_1, enc_feature_2 = model(aug_1_modal_1, aug_1_modal_2, aug_2_modal_1, aug_2_modal_2, proj_head=True)
            # enc_feature1 -> dict // (example) enc_feature1['ecg'] & enc_feature1['hr'] from Augmentation 1
            # enc_feature2 -> dict // (example) enc_feature2['ecg'] & enc_feature2['hr'] from Augmentation 2
            
            
            # Predict the subject
            subj_pred = advs_model(enc_feature_1, enc_feature_2) 
            # or subj_pred = advs_model(enc_feature_1['ecg'], enc_feature_1['hr], enc_feature_2['ecg'], enc_feature_2['hr'])
            
            advs_loss = advs_model.forward_adversarial_loss(subj_pred, subj_label)
            
            # To-do for calculating the accuracy
            # num_adversary_correct_train_preds += adversarial_loss_fn.get_number_of_correct_preds(x_t1_initial_subject_preds, y)
            # total_num_adversary_train_preds += len(x_t1_initial_subject_preds)
            
            advs_loss.backward()
            advs_optimizer.step()
            
            running_advs_train_loss += advs_loss.item()
            
            # For efficient memory management
            del enc_feature_1, enc_feature_2, subj_pred, advs_loss
            
            # For updating the only Focal model (SSL model)
            for param in model.parameters():
                param.requires_grad = True
            for param in advs_model.parameters():
                param.requires_grad = False
            
            optimizer.zero_grad()

            enc_feature_1, enc_feature_2 = model(aug_1_modal_1, aug_1_modal_2, aug_2_modal_1, aug_2_modal_2, proj_head=True)
            
            subj_pred = advs_model(enc_feature_1, enc_feature_2) 
            subj_invariant_loss = advs_model.forward_subject_invariance_loss(subj_pred, subj_label)
            
            focal_loss = focal_loss_fn(enc_feature_1, enc_feature_2, subj_invariant_loss) # To-Do -> add regularization term about subject invariant
            focal_loss.backward()
            optimizer.step()
            
            focal_train_loss += focal_loss.item()
                
            # For efficient memory management
            del enc_feature_1, enc_feature_2, subj_pred, focal_loss
            torch.cuda.empty_cache()
            
        if ep % trainer_config['log_interval'] == 0:
            print(f"Epoch {ep} - Adversarial Loss: {running_advs_train_loss/ len(train_loader)}, \
                Focal Loss: {focal_train_loss/ len(train_loader)}")
            
            if ep % trainer_config['val_interval'] == 0:
                model.eval()
                advs_model.eval()
                focal_loss_fn.eval()
                
                focal_val_loss = 0
                
                for raw_modal_1, raw_modal_2, subj_label, sleep_label in valid_loader:
                    raw_modal_1, raw_modal_2, subj_label, sleep_label = raw_modal_1.to(device), raw_modal_2.to(device), subj_label.to(device), sleep_label.to(device)
                    
                    aug_1_modal_1, aug_2_modal_1  = raw_modal_1, raw_modal_1
                    aug_1_modal_2, aug_2_modal_2  = raw_modal_2, raw_modal_2
                    
                    with torch.no_grad():
                        # x1_represent, x2_represent = model(raw_modal_1, raw_modal_2)
                        enc_feature_1, enc_feature_2 = model(aug_1_modal_1, aug_1_modal_2, aug_2_modal_1, aug_2_modal_2, proj_head=True)
                        subj_pred = advs_model(enc_feature_1, enc_feature_2)
                        
                        focal_loss = focal_loss_fn(enc_feature_1, enc_feature_2, 0) # To-Do -> add regularization term about subject invariant
                        
                        focal_val_loss += focal_loss.item()
                        
                        # For efficient memory management
                        del enc_feature_1, enc_feature_2, subj_pred, focal_loss
                        torch.cuda.empty_cache()
                        
                print("-----"*10)
                print(f"(Validation) Epoch{ep} - Focal Loss: {focal_val_loss/ len(valid_loader)}")                    
                                
                if focal_val_loss < best_val_loss:
                    best_val_loss = focal_val_loss
                    
                    # To-do -> fix the save model format
                    # torch.save(model.state_dict(), os.path.join(args.save_dir, 'focal_model.pth'))
                    # torch.save(advs_model.state_dict(), os.path.join(args.save_dir, 'advs_model.pth'))
                    print("************* Model Saved *************")
                print("-----"*10)
                
def print_args(args):
    
    print("Data Configs:")
    for k, v in args.data_config.items():
        print(f"\t{k}: {v}")
    print("----------"*10)
    
    
    print("Focal Configs:")
    for k, v in args.focal_config.items():
        print(f"\t{k}: {v}")
    print("----------"*10)
    
    print("Subject Invariant Configs:")
    for k, v in args.subj_invariant_config.items():
        print(f"\t{k}: {v}")
    print("----------"*10)
    
    print("Trainer Configs:")
    for k, v in args.trainer_config.items():
        print(f"\t{k}: {v}")
    print("----------"*10)

def main():
    
    # Check the arguments
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # logging.basicConfig(level=print,
    #                 format='%(asctime)s %(levelname)s: %(message)s',
    #                 datefmt='%Y-%m-%d %H:%M:%S',
    #                 filename=os.pat.join(args.base_config["log_save_dir"], 
    #                                       f'focal_subj_mesa_{time}.log'),
    #                 filemode='a')
    
    print_args(args)

    train_dataset = MESAPairDataset(file_path=args.data_config['train_data_dir'],
                                    modalities=args.data_config['modalities'],
                                    subject_idx=args.data_config['subject_key'],
                                    stage=args.data_config['label_key'])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.trainer_config['batch_size'],
                                               shuffle=True,
                                               num_workers=4)
    
    print("Successfully Loaded Train Data")

    val_dataset = MESAPairDataset(file_path=args.data_config['val_data_dir'],
                                    modalities=args.data_config['modalities'],
                                    subject_idx=args.data_config['subject_key'],
                                    stage=args.data_config['label_key'])
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.trainer_config['batch_size']//3,
                                             shuffle=False,
                                             num_workers=4)
    
    print("Successfully Loaded Validation Data")    

    print("Loading the Focal Model")
    
    advs_model = AdversarialModel(args).to(args.subj_invariant_config["device"])
    advs_optimizer = torch.optim.Adam(advs_model.parameters(), lr=args.subj_invariant_config['lr'])
    print("Complete Loading the Adversarial Model")
    
    
    if str(list(args.focal_config["backbone"].keys())[0]) == "DeepSense":
        backbone = DeepSense(args).to(args.focal_config["device"])
        
    else:
        raise ValueError("Not Supported Backbone")
    
    focal_model = FOCAL(args, backbone).to(args.focal_config["device"])
    focal_optimizer = torch.optim.Adam(focal_model.parameters(), lr=args.focal_config["lr"])
    focal_loss_fn = FOCALLoss(args)
    print("Complete Loading the FOCAL Model")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    print("Start Training SA Focal Model")
    
    
    train_SA_Focal(train_loader, val_loader, focal_model, advs_model,
                   focal_optimizer, advs_optimizer, focal_loss_fn, device, args)
    
    print("Finished Training SA Focal Model")
    
    
if __name__ == '__main__':
    main()