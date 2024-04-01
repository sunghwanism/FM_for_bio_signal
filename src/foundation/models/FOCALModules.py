import torch.nn as nn
import torch

import sys
sys.path.append("../")
import args

torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
torch.cuda.manual_seed_all(args.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FOCAL(nn.Module):
    def __init__(self, args, backbone):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(FOCAL, self).__init__()

        self.args = args
        self.config = args.focal_config
        self.modalities = args.data_config["modalities"]
        self.backbone = backbone
        
        self.classifier = nn.Sequential(
            nn.Linear(self.config['embedding_dim'] * 4, 128),  # Concatenated features from 4 inputs
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )
        

    def forward(self, aug1_mod1, aug1_mod2, aug2_mod1, aug2_mod2, proj_head=True, class_head=False):
        """
        Input:
            aug1_mod1: augmented_1 input of the first modality.
            aug1_mod2: augmented_1 input of the second modality.
            aug2_mod1: augmented_2 input of the first modality.
            aug2_mod2: augmented_2 input of the second modality.
        Output:
            mod_features1: Projected mod features of the first augmentation.
            mod_features2: Projected mod features of the second augmentation.
        """
        # compute features

        mod_features1 = self.backbone(aug1_mod1, aug1_mod2, class_head=False, proj_head=proj_head)
        mod_features2 = self.backbone(aug2_mod1, aug2_mod2, class_head=False, proj_head=proj_head)
        
        if class_head:
            features = []
            for modality in self.args.data_config['modalities']:
                features.append(mod_features1[modality])
                features.append(mod_features2[modality])
            
            concatenated_features = torch.cat(features, dim=1)
            logit = self.classifier(concatenated_features)

            return logit
        else:
            return mod_features1, mod_features2


def split_features(mod_features):
    """
    Split the feature into private space and shared space.
    mod_feature: [b, seq, dim], where we use the sequence sampler
    """
    split_mod_features = {}

    for mod in mod_features:
        if mod_features[mod].ndim == 2:
            split_dim = mod_features[mod].shape[1] // 2
            split_mod_features[mod] = {
                "shared": mod_features[mod][:, 0:split_dim],
                "private": mod_features[mod][:, split_dim:],
            }
            
        else:
            b, seq, dim = mod_features[mod].shape
            split_dim = dim // 2
            split_mod_features[mod] = {
                "shared": mod_features[mod][:, :, 0:split_dim],
                "private": mod_features[mod][:, :, split_dim : 2 * split_dim],
            }

    return split_mod_features
