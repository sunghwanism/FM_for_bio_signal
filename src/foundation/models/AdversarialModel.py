import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialModel(nn.Module):
    def __init__(self, args):
        super(AdversarialModel, self).__init__()
        embedding_dim = args.subj_invariant_config['embedding_dim']
        num_subjects = args.subj_invariant_config['num_subjects']
        dropout_rate = args.subj_invariant_config['dropout_rate']
        self.modalities = args.data_config['modalities']
        self.num_subjects = num_subjects
        
        self.fc = nn.Linear(embedding_dim * 4, embedding_dim)  # Assuming concatenation of embeddings
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, num_subjects),
            nn.Sigmoid()  # Use Softmax for multi-class classification
        )

    def forward(self, embeddings1, embeddings2):
        embeddings1_mod1 = embeddings1[self.modalities[0]] # B, embedding_dim
        embeddings1_mod2 = embeddings1[self.modalities[1]] # B, embedding_dim
        embeddings2_mod1 = embeddings2[self.modalities[0]] # B, embedding_dim
        embeddings2_mod2 = embeddings2[self.modalities[1]] # B, embedding_dim
        
        combined_embeddings = torch.cat((embeddings1_mod1, embeddings1_mod2, embeddings2_mod1, embeddings2_mod2), dim=1)
        embedding = self.fc(combined_embeddings)
        return self.model(embedding)

    def forward_adversarial_loss(self, subject_preds, subject_labels):
        subject_labels = F.one_hot(subject_labels, num_classes=self.num_subjects)
        subject_outs = F.normalize(subject_preds, p=2, dim=1)
        
        BATCH_DIM = 0
        log_noise = 1e-12
        adversarial_loss = 0.
        curr_batch_size = subject_outs.size(BATCH_DIM)

        for i in range(curr_batch_size):
            j = torch.argmax(subject_labels[i, :]) # Get the index of the true class
            adversarial_loss += -1. * torch.log(log_noise + subject_outs[i, j])
        return adversarial_loss

    def forward_subject_invariance_loss(self, subject_preds, subject_labels):
        subject_labels = F.one_hot(subject_labels, num_classes=self.num_subjects)
        subject_outs = F.normalize(subject_preds, p=2, dim=1)

        BATCH_DIM = 0
        log_noise = 1e-12
        subject_invariance_loss = 0.
        curr_batch_size = subject_outs.size(BATCH_DIM)

        for i in range(curr_batch_size):
            j = torch.argmax(subject_labels[i, :])
            subject_invariance_loss += (-1.) * torch.log(log_noise + (1. - subject_outs[i, j]))
        return subject_invariance_loss
