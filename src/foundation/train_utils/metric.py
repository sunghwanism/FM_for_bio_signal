import torch

def get_acc_loss_from_dataloader(dataloader):
    
    total_correct = 0
    total_samples = 0
    total_loss = 0

    for subj_preds, advs_loss, subj in dataloader:       
        _, predicted = subj_preds.max(1)
        total_correct += (predicted == subj).sum().item()
        total_samples += subj.size(0)
        total_loss += advs_loss.sum().item()
 
    acc = 100 * (total_correct / total_samples)
    loss = total_loss / total_samples

    return acc, loss


def get_accuracy_from_train_process(logit_arr, true_label):

    predicted_label = torch.argmax(logit_arr, dim=1)
    acc = torch.sum(predicted_label == true_label).item() / true_label.size(0)

    return acc