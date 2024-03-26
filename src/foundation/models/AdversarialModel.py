import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialModel(nn.Module):
    def __init__(self, embedding_dim, num_subjects, dropout_rate=0.5):
        super(AdversarialModel, self).__init__()
        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)  # Assuming concatenation of embeddings
        self.model = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(embedding_dim // 2, embedding_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(embedding_dim // 2, embedding_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(embedding_dim // 2, num_subjects),
            torch.nn.Sigmoid()  # Use Softmax for multi-class classification
        )
    
    def forward(self, embeddings1, embeddings2):
        combined_embeddings = torch.cat((embeddings1, embeddings2), dim=1)
        embedding = self.fc(combined_embeddings)
        return self.model(embedding)

def forward_adversarial_loss(model, embeddings1, embeddings2, subject_labels):
    subject_outs = model(embeddings1, embeddings2)
    subject_outs = F.normalize(subject_outs, p=2, dim=1)

    BATCH_DIM = 0
    log_noise = 1e-12
    adversarial_loss = 0.
    curr_batch_size = subject_outs.size(BATCH_DIM)

    for i in range(curr_batch_size):
        j = torch.argmax(subject_labels[i, :])
        adversarial_loss += -1. * torch.log(log_noise + subject_outs[i, j])
    return adversarial_loss