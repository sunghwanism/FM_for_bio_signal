
import os
import torch


# Set CUDA for GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set the root directory
root_dir = '/NFS/Users/moonsh/FM_biosignal'
data_dir = '/NFS/Users/moonsh/data/mesa/preproc'

################################################################################
# General Arguments
"""
Modalities = 'hr', 'ecg', 'activity'
"""

base_config = {'train_data_dir': os.path.join(data_dir, 'pair_small'), # 'pair' is real train data
               'val_data_dir': os.path.join(data_dir, 'pair'),
               'test_data_dir': os.path.join(data_dir, 'pair_test'),
               'subject_key': 'subject_idx',
               'train_num_subjects': 100,
               'test_num_subjects': 50,
               'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
               'log_save_dir': os.path.join(root_dir, 'logs'),
}

################################################################################
# Dataset Arguments

data_config = {'modalities': ['ecg', 'hr'],
               'label_key': 'stage',
               'Augmentation':['NoAug', 'GaussianNoise'], # 'NoAug', 'GaussianNoise', "AmplitudeScale"
               'num_classes': None,
               
               }


################################################################################
# For Focal Loss

focal_config = {'backbone': 'DeepSense',
                'tag': 'usePrivate', # 'noPrivate' for not using private loss
                'embedding_dim': 128,
                'num_subjects': 100,
                'dropout_rate': 0.5,
                'lr': 0.001,
                'shared_contrastive_loss_weight': 0.5,
                'private_contrastive_loss_weight': 0.5,
                'orthogonality_loss_weight': 0.2,
                'subject_invariant_loss_weight': 0.2,
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
}

################################################################################
# For Subject Invariant Loss

subj_invariant_config = {
    
                         'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
                         }




################################################################################
# Trainer Arguments

trainer_config = {'batch_size': 256,
                  'epochs': 100,
                  'log_interval': 5,
                  'val_interval': 10,
                  'model_save_dir': os.path.join(root_dir, 'checkpoints'),
}


