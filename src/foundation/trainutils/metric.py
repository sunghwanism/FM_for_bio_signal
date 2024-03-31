import torch
import matplotlib.pyplot as plt

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



def save_metrics(train_focal_losses, val_focal_losses, train_accuracies, train_advs_losses, time):
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    ax[0].plot(train_focal_losses, label='Train')
    ax[0].plot(val_focal_losses, label='Validation')
    ax[0].set_title('Focal Loss Curve')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    ax[1].plot(train_accuracies, label='Train')
    ax[1].set_title('Adversarial Accuracy Curve')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    
    ax[2].plot(train_advs_losses, label='Train')
    ax[2].set_title('Adversarial Loss Curve')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Loss')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.suptitle(f"Focal SSL Learning Curve")
    plt.savefig(f'../asset/SSL_focal_Learning_Curve_{time}.png')    
    