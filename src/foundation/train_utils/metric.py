import torch

# def get_acc_loss_from_dataloader(dataloader):
    
#     total_correct = 0
#     total_samples = 0
#     total_loss = 0

#     for subj_preds, advs_loss, subj in dataloader:       
#         _, predicted = subj_preds.max(1)
#         total_correct += (predicted == subj).sum().item()
#         total_samples += subj.size(0)
#         total_loss += advs_loss.sum().item()
 
#     acc = 100 * (total_correct / total_samples)
#     loss = total_loss / total_samples

#     return acc, loss


def get_accuracy_from_train_process(logit_arr, true_label):

    predicted_label = torch.argmax(logit_arr, dim=1)
    acc = torch.sum(predicted_label == true_label).item() / true_label.size(0)

    return acc


def get_acc_loss_from_dataloader(model, dataloder, device, criterion):
    
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    for i, data in enumerate(dataloder):
        ecg, hr, _, sleep_stage = data
        ecg = ecg.to(device)
        hr = hr.to(device)
        sleep_stage = sleep_stage.to(device)
        
        output = model(ecg, hr)
        loss = criterion(output, sleep_stage)
        
        total_loss += loss.item()
        total_correct += torch.sum(torch.argmax(output, dim=1) == sleep_stage).item()
        total_samples += sleep_stage.size(0)
        
    return total_correct / total_samples, total_loss / total_samples