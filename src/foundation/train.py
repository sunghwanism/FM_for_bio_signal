import os
import sys
sys.path.append('./')

import args
import argparse
import logging

# import torch
from models.AdversarialModel import AdversarialModel
from models.FOCALModules import FOCAL
import datetime



def train_SA_Focal(train_loader, val_loader, model, advs_model, 
                   optimizer, advs_optimizer, focal_loss_fn, device, args):
    
    trainer_config = args.trainer_config
    
    model.train()
    best_val_loss = float('inf')
    
    for ep in range(trainer_config['epochs']):
        running_advs_train_loss = 0
        focal_train_loss = 0
        
        for raw_modal_1, raw_modal_2, subj in train_loader:
            raw_modal_1, raw_modal_2, subj = raw_modal_1.to(device), raw_modal_2.to(device), subj.to(device)
            
            # For updating the only advs_model (classifier)
            for param in model.parameters():
                param.requires_grad = False
            for param in advs_model.parameters():
                param.requires_grad = True
                
            advs_optimizer.zero_grad()
            
            # Using Encoder for classify the subject
            enc_modal_1, enc_modal_2 = model.encoder(raw_modal_1, raw_modal_2) # To-do -> Make encoder function in Focal Module
            
            # Predict the subject
            subj_preds = advs_model(enc_modal_1, enc_modal_2)
            
            advs_loss = advs_model.forward_adversarial_loss(subj_preds, subj_labels)
            
            # To-do for calculating the accuracy
            # num_adversary_correct_train_preds += adversarial_loss_fn.get_number_of_correct_preds(x_t1_initial_subject_preds, y)
            # total_num_adversary_train_preds += len(x_t1_initial_subject_preds)
            
            advs_loss.backward()
            advs_optimizer.step()
            
            running_advs_train_loss += advs_loss.item()
            
            # For efficient memory management
            del enc_modal_1, enc_modal_2, subj_preds, advs_loss
            
            # For updating the only Focal model (SSL model)
            for param in model.parameters():
                param.requires_grad = True
            for param in advs_model.parameters():
                param.requires_grad = False
            
            optimizer.zero_grad()

            x1_represent, x2_represent = model(raw_modal_1, raw_modal_2)
            
            x1_embd, x2_embd = model.encoder(raw_modal_1, raw_modal_2)
            subj_pred = advs_model(x1_embd, x2_embd)
            subj_invariant_loss = advs_model.forward_subject_invariance_loss(subj_pred, subj_labels, args.adversarial_weighting_factor) # DONE -> add subject_invariant function loss
            
            focal_loss = focal_loss_fn(x1_represent, x2_represent, subj_invariant_loss) # To-Do -> add regularization term about subject invariant
            focal_loss.backward()
            optimizer.step()
            
            focal_train_loss += focal_loss.item()
            
            # For efficient memory management
            del x1_represent, x2_represent, x1_embd, x2_embd, subj_pred, focal_loss
            torch.cuda.empty_cache()
            
        if ep % args.log_interval == 0:
            print(f"Epoch {ep} - Adversarial Loss: {running_advs_train_loss/ len(train_loader)}, \
                Focal Loss: {focal_train_loss/ len(train_loader)}")
            
            if ep % args.val_interval == 0:
                model.eval()
                advs_model.eval()
                
                advs_val_loss = 0
                focal_val_loss = 0
                
                for raw_modal_1, raw_modal_2, subj in val_loader:
                    raw_modal_1, raw_modal_2, subj = raw_modal_1.to(device), raw_modal_2.to(device), subj.to(device)
                    
                    with torch.no_grad():
                        x1_represent, x2_represent = model(raw_modal_1, raw_modal_2)
                        x1_embd, x2_embd = model.encoder(raw_modal_1), model.encoder(raw_modal_2)
                        subj_pred = advs_model(x1_embd, x2_embd) # output -> sigmoid value
                        
                        advs_loss = advs_model.loss_fcn(subj_pred, subj)
                        focal_loss = focal_loss_fn(x1_represent, x2_represent, subj_pred, subj)
                        
                        advs_val_loss += advs_loss.item()
                        focal_val_loss += focal_loss.item()
                        
                        # For efficient memory management
                        del x1_represent, x2_represent, x1_embd, x2_embd, subj_pred, focal_loss
                        torch.cuda.empty_cache()
                        
                print("-----"*10)
                print(f"(Validation) Epoch{ep} - Adversarial Loss: {advs_val_loss/ len(val_loader)}, \
                    Focal Loss: {focal_val_loss/ len(val_loader)}")                    
                                
                if focal_val_loss < best_val_loss:
                    best_val_loss = focal_val_loss
                    
                    # To-do -> fix the save model format
                    # torch.save(model.state_dict(), os.path.join(args.save_dir, 'focal_model.pth'))
                    # torch.save(advs_model.state_dict(), os.path.join(args.save_dir, 'advs_model.pth'))
                    print("************* Model Saved *************")
                print("-----"*10)
                
def print_args(args):
    
    logging.info("Base Configs:")
    for k, v in args.base_config.items():
        logging.info(f"\t{k}: {v}")
    logging.info("----------"*10)
    
    logging.info("Focal Configs:")
    for k, v in args.focal_config.items():
        print(f"\t{k}: {v}")
    logging.info("----------"*10)
    
    logging.info("Subject Invariant Configs:")
    for k, v in args.subj_invariant_config.items():
        print(f"\t{k}: {v}")
    logging.info("----------"*10)
    
    logging.info("Trainer Configs:")
    for k, v in args.trainer_config.items():
        print(f"\t{k}: {v}")
    logging.info("----------"*10)

def main():
    
    # Check the arguments
    print_args(args)
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=os.path.join(args.base_config["log_save_dir"], 
                                          f'focal_subj_mesa_{time}.log'),
                    filemode='a')
    
    train_dataset = None
    val_dataset = None
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.trainer_config['batch_size'], shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.trainer_config['batch_size'], shuffle=False)
    # print("Successfully Loaded Data")
    
    

    # print("Start Training SA Focal Model")
    
    # AdversarialModel = AdversarialModel(embedding_dim, num_subjects, dropout_rate=0.5)
    # advs_optimizer = torch.optim.Adam(AdversarialModel.parameters(), lr=args.lr)
    
    # FOCAL_Model = FOCAL(args, backbone)
    # focal_optimizer = torch.optim.Adam(FOCAL_Model.parameters(), lr=args.lr)
    # focal_loss_fn = FOCALLoss(args)
        
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # output = train_SA_Focal()
    
    
if __name__ == '__main__':
    
    main()