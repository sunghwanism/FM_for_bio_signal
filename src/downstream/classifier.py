import torch
import torch.nn as nn
import torch.nn.functional as F


class SleepStageClassifier(nn.Module):
    def __init__(self, args):
        super(SleepStageClassifier, self).__init__()
        self.args = args
        self.num_classes = args.downstream_config['num_classes']
        self.embedding_dim = args.downstream_config['embedding_dim']
        
        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 4, 128),  # Concatenated features from 4 inputs
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )
        
    def forward(self, enc_feature_1, enc_feature_2):
        
        features = []
        for modality in self.args.data_config['modalities']:
            features.append(enc_feature_1[modality])
            features.append(enc_feature_2[modality])
        
        concatenated_features = torch.cat(features, dim=1)
        out = self.classifier(concatenated_features)
        
        return out