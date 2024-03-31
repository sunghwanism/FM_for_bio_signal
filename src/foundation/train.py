import os
import sys
sys.path.append('./')

import args
import datetime
from tqdm import tqdm

import torch
import numpy as np
from models.AdversarialModel import AdversarialModel
from models.FOCALModules import FOCAL
from models.loss import FOCALLoss
from models.Backbone import DeepSense
from trainutils.metric import save_metrics

from data.Dataset import MESAPairDataset
from data.Augmentaion import init_augmenter

def train_SA_Focal(train_loader, valid_loader, model, advs_model, 
                   optimizer, advs_optimizer, focal_loss_fn, args):
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    trainer_config = args.trainer_config
    model_save_dir = args.trainer_config["model_save_dir"]
    log_save_dir = args.trainer_config["log_save_dir"]
    
    model_save_format = args.model_save_format
    model_save_format["focal_config"] = args.focal_config
    model_save_format['subj_invariant_config'] = args.subj_invariant_config
    model_save_format['trainer_config'] = args.trainer_config
    model_save_format['data_config'] = args.data_config
    
    aug_1_name = args.data_config['augmentation'][0]
    aug_1_config = args.data_config['augmenter_config'].get(aug_1_name, {})
    aug_2_name = args.data_config['augmentation'][1]
    aug_2_config = args.data_config['augmenter_config'].get(aug_2_name, {})
    
    aug_1 = init_augmenter(aug_1_name, aug_1_config)
    aug_2 = init_augmenter(aug_2_name, aug_2_config)
    
    model.train()
    best_val_loss = float('inf')
    
    train_focal_losses, val_focal_losses = [], []
    train_advs_losses = []
    train_accuracies = []
    
    for ep in tqdm(range(trainer_config['epochs'])):
        
        model.train()
        advs_model.train()
        focal_loss_fn.train()
        
        # Save Result
        focal_train_loss = 0
        running_advs_train_loss = 0
        
        correct_preds = 0
        total_preds = 0
        
        
        for raw_modal_1, raw_modal_2, subj_label, sleep_label in train_loader:
            raw_modal_1, raw_modal_2, subj_label, sleep_label = raw_modal_1.to(args.focal_config["device"]), raw_modal_2.to(args.focal_config["device"]), subj_label.to(args.focal_config["device"]), sleep_label.to(args.focal_config["device"]) # [B, 30], [B, 30*256], [B, 1]
            
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
            
            # Predict the subject
            subj_pred = advs_model(enc_feature_1, enc_feature_2) 
            advs_loss = advs_model.forward_adversarial_loss(subj_pred, subj_label)
            
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
            
            # Calculate accuracy
            preds = torch.argmax(subj_pred, dim=1)
            correct_preds += (preds == subj_label).sum().item()
            total_preds += subj_label.size(0)
            
            # For efficient memory management
            del enc_feature_1, enc_feature_2, subj_pred, focal_loss
            torch.cuda.empty_cache()
            
        # Calculate and store train accuracy and losses for plotting
        train_accuracy = correct_preds / total_preds
        train_accuracies.append(train_accuracy)
        train_advs_losses.append(running_advs_train_loss / len(train_loader))
        train_focal_losses.append(focal_train_loss / len(train_loader))
        
        print(f"Epoch {ep} - Adversarial Loss: {running_advs_train_loss / len(train_loader)}, \
            Focal Loss: {focal_train_loss / len(train_loader)}, Accuracy: {train_accuracy}")
                
        if ep % trainer_config['val_interval'] == 0:
            model.eval()
            advs_model.eval()
            focal_loss_fn.eval()
            
            focal_val_loss = 0
            
            for raw_modal_1, raw_modal_2, subj_label, sleep_label in valid_loader:
                raw_modal_1, raw_modal_2, subj_label, sleep_label = raw_modal_1.to(args.focal_config["device"]), raw_modal_2.to(args.focal_config["device"]), \
                                                                    subj_label.to(args.focal_config["device"]), sleep_label.to(args.focal_config["device"])
                
                with torch.no_grad():
                    enc_feature_1, enc_feature_2 = model(raw_modal_1, raw_modal_2, raw_modal_1, raw_modal_2, proj_head=True)                    
                    focal_loss = focal_loss_fn(enc_feature_1, enc_feature_2, 0) # To-Do -> add regularization term about subject invariant
                    focal_val_loss += focal_loss.item()
                    
                    # For efficient memory management
                    del enc_feature_1, enc_feature_2, subj_pred, focal_loss
                    torch.cuda.empty_cache()
                    
            print("-----"*20)
            print(f"(Validation) Epoch{ep} - Focal Loss: {focal_val_loss/ len(valid_loader)}")                    
            
            val_focal_losses.append(focal_val_loss / len(valid_loader))
                            
            if focal_val_loss < best_val_loss:
                best_val_loss = focal_val_loss
                
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                
                time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                focal_model_checkpoint = os.path.join(model_save_dir, f'SSL_focal_model_{time}_ep_{ep}.pth')
                
                # Save ckpt & arguments
                model_save_format["train_acc"] = train_accuracy
                model_save_format["train_loss"] = focal_train_loss / len(train_loader)
                model_save_format["val_loss"] = focal_val_loss / len(valid_loader)
                model_save_format["train_epoch"] = ep
                model_save_format["focalmodel_path"] = focal_model_checkpoint
                model_save_format["focal_state_dict"] = model.state_dict()
                model_save_format['advs_state_dict'] = advs_model.state_dict()
                
                torch.save(model_save_format, focal_model_checkpoint)                
                
                print(f"Model Saved - Focal Model: {focal_model_checkpoint}")
            print("-----"*20)
    
    finish_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGPATH = os.path.join(args.trainer_config["log_save_dir"], f'SSL_focal_log_{finish_time}.npz')
    train_log = np.array([train_focal_losses, val_focal_losses, train_accuracies, train_advs_losses])
    np.savez(LOGPATH, train_log)
    
    save_metrics(train_focal_losses, val_focal_losses, train_accuracies, train_advs_losses, finish_time)
                
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
    
    print_args(args)

    train_dataset = MESAPairDataset(file_path=args.data_config['train_data_dir'],
                                    modalities=args.data_config['modalities'],
                                    subject_idx=args.data_config['subject_key'],
                                    stage=args.data_config['label_key'])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.trainer_config['batch_size'],
                                               shuffle=True,
                                               num_workers=4)
    
    val_dataset = MESAPairDataset(file_path=args.data_config['val_data_dir'],
                                    modalities=args.data_config['modalities'],
                                    subject_idx=args.data_config['subject_key'],
                                    stage=args.data_config['label_key'])
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.trainer_config['batch_size']//3,
                                             shuffle=False,
                                             num_workers=4)
    
    print("****** Successfully Dataset ******")    
    
    advs_model = AdversarialModel(args).to(args.subj_invariant_config["device"])
    advs_optimizer = torch.optim.Adam(advs_model.parameters(), lr=args.subj_invariant_config['lr'])
    print("****** Complete Loading the Adversarial Model ******")
    
    
    if str(list(args.focal_config["backbone"].keys())[0]) == "DeepSense":
        backbone = DeepSense(args).to(args.focal_config["device"])
        
    else:
        raise ValueError("Not Supported Backbone")
    
    focal_model = FOCAL(args, backbone).to(args.focal_config["device"])
    focal_optimizer = torch.optim.Adam(focal_model.parameters(), lr=args.focal_config["lr"])
    focal_loss_fn = FOCALLoss(args)
    print("****** Complete Loading the FOCAL Model ******")
    
    print("Start Training SA Focal Model")
    
    
    train_SA_Focal(train_loader, val_loader, focal_model, advs_model,
                   focal_optimizer, advs_optimizer, focal_loss_fn, args)
    
    print("Finished Training SA Focal Model")
    
    
if __name__ == '__main__':
    main()