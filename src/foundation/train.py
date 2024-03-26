import os
import sys
sys.path.append('./')

import torch
from foundation.models import .



def train_SA_Focal(train_loader, val_loader, model, advs_model, 
                   optimizer, advs_optimizer, focal_loss_fn, device, args):
    
    model.train()
    best_val_loss = float('inf')
    
    for ep in range(args.epochs):
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
            
            advs_loss = advs_model.forward_adversarial_loss(subj_preds, subj)
            
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
            subj_invariant_loss = advs_model.forward_adversarial_loss(subj_pred, subj) # To-do -> add subject_invariant function loss
            
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
                


def main(args):
    
    print("Start Training SA Focal Model")
    
    # AdversaryModel = 
    # adversarial_loss_fn = SAAdersarialLoss()
    # advs_optimizer = torch.optim.Adam(AdversaryModel.parameters(), lr=args.lr)
    
    # FocalModel = 
    # total_loss_fn = FOCALLoss()
    # focal_optimizer = torch.optim.Adam(FocalModel.parameters(), lr=args.lr)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # output = train_SA_Focal()
    
    
if __name__ == '__main__':
    args
    main(args)