
import os
import torch


# Set CUDA for GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Set the root directory
# root_dir = '/data8/jungmin/uot_class/MIE1517_DL/FM_for_bio_signal'
# data_dir = "/data8/jungmin/uot_class/MIE1517_DL/FM_for_bio_signal/src/foundation/dataset"

root_dir = "/NFS/Users/moonsh/FM_biosignal"
data_dir = "/NFS/Users/moonsh/data/mesa/preproc/"
SEED = 42
################################################################################
# General Arguments
"""
Modalities = 'hr', 'ecg', 'activity'
"""

base_config = {'train_data_dir': os.path.join(data_dir, 'pair_train'), # 'pair' is real train data
               'val_data_dir': os.path.join(data_dir, 'pair_val'),
               'test_data_dir': os.path.join(data_dir, 'pair_test'),
               'modalities': ['ecg', 'hr'],
               'label_key': 'stage',
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
               'augmentation': ['GaussianNoise', 'AmplitudeScale'],
               'augmenter_config': {
                   'GaussianNoise': {'max_noise_std': 0.1},
                   'AmplitudeScale': {'amplitude_scale': 0.5}
                },
               'num_classes': None,
}

################################################################################
# For Focal Loss

focal_config = {'backbone': 
                          {'DeepSense': {'mod1_kernel_size': 11,
                                         'mod1_stride': 3,
                                         'mod1_padding': 0,
                                         'mod2_kernel_size': 5,
                                         'mod2_stride': 1,
                                         'mod2_padding': 2,
                                         'num_conv_layers': 2,
                                         'conv_dim': 128,
                                         'num_recurrent_layers': 2,
                                         'recurrent_dim': 128,
                                         'hidden_dim': 128, # should be same with conv_dim
                                         'mod1_linear_dim': 35840,
                                         'mod2_linear_dim': 3840,
                                         'num_classes': 5, # in SSL -> Embedding Dimension / in Supervised -> Number of Classes
                                         'fc_dim': 128,
                                         'class_in_dim': 1980,
                                         }
                             },
                'tag': 'usePrivate', # 'noPrivate' for not using private loss
                'embedding_dim': 128,
                'num_subjects': 100,
                'dropout_rate': 0.5,
                'lr': 0.0001,
                'shared_contrastive_loss_weight': 0.5,
                'private_contrastive_loss_weight': 0.5,
                'orthogonality_loss_weight': 0.2,
                'subject_invariant_loss_weight': 0.2,
                'seq_len': 16,
                'temperature': 0.5,
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

################################################################################
# For Subject Invariant Loss

subj_invariant_config = {'embedding_dim': 128,
                         'num_subjects': 100,
                         'dropout_rate': 0.5,
                         'adversarial_weighting_factor': 0.1,
                         'lr': 0.01,
                         'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                         }


################################################################################
# Trainer Arguments

trainer_config = {'batch_size': 1024,
                  'epochs': 100,
                  'log_interval': 1,
                  'val_interval': 1,
                  'model_save_dir': os.path.join(root_dir, 'checkpoints'),
}
