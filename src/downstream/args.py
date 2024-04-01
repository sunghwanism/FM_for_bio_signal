
import os
import torch


# Set CUDA for GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Set the root directory
root_dir = '/data8/jungmin/uot_class/MIE1517_DL/FM_for_bio_signal'
data_dir = "/data8/jungmin/uot_class/MIE1517_DL/FM_for_bio_signal/src/foundation/dataset"

# root_dir = "/NFS/Users/moonsh/FM_biosignal"
# data_dir = "/NFS/Users/moonsh/data/mesa/preproc/"
SEED = 42
################################################################################
# Dataset Arguments
"""
Modalities = 'hr', 'ecg', 'activity'
"""

data_config = {'train_data_dir': os.path.join(data_dir, 'pair_train'), # 'pair' is real train data
               'val_data_dir': os.path.join(data_dir, 'pair_val'),
               'test_data_dir': os.path.join(data_dir, 'pair_test'),
               'modalities': ['ecg', 'hr'],
               'label_key': 'stage',
               'subject_key': 'subject_idx',
               'augmentation': ['GaussianNoise', 'NoAugmenter'],
               'augmenter_config': {
                   'GaussianNoise': {'max_noise_std': 0.1},
                   'AmplitudeScale': {'amplitude_scale': 0.3}
                },
}

################################################################################
# For Focal Loss

focal_config = {'backbone': 
                          {'DeepSense': {'mod1_kernel_size': 11,
                                         'mod1_stride': 3,
                                         'mod1_padding': 5,
                                         ###############################
                                         'mod2_kernel_size': 5,
                                         'mod2_stride': 1,
                                         'mod2_padding': 2,
                                         ###############################
                                         'num_conv_layers': 4,
                                         'conv_dim': 512,
                                         'conv_dropout_rate': 0.3,
                                         ###############################
                                         'num_recurrent_layers': 4,
                                         'recurrent_dim': 1024,
                                         ###############################
                                         'mod1_linear_dim': 32768,
                                         'mod2_linear_dim': 30720,
                                         'fc_dim': 512,
                                         'class_in_dim': 1980,
                                         'proj_dropout_rate': 0.3,
                                         ###############################
                                         'num_classes': 4, # for classification only using DeepSense
                                         }
                             },
                'embedding_dim': 1024, # final embedding dimension -> using classifier
                'num_classes': 4,
                'lr': 0.0001,
                'shared_contrastive_loss_weight': 0.20,
                'private_contrastive_loss_weight': 0.40,
                'orthogonality_loss_weight': 0.20,
                'subject_invariant_loss_weight': 0.2*0.01,
                'temperature': 0.5,
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

################################################################################
# For Subject Invariant Loss

subj_invariant_config = {# 'embedding_dim': 128, -> it is same with embedding_dim in focal_config
                         'num_subjects': 100,
                         'dropout_rate': 0.0,
                         'lr': 0.001,
                         'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                         }


################################################################################
# Trainer Arguments

trainer_config = {'batch_size': 330,
                  'epochs': 100,
                  'log_interval': 1,
                  'val_interval': 1,
                  'model_save_dir': os.path.join(root_dir, 'checkpoints'),
                  'log_save_dir': os.path.join(root_dir, 'logs')
}

model_save_format = {"train_acc": None,
                     "val_acc": None,
                     "train_loss": None,
                     "val_loss": None,
                     "epoch": None,
                     "lr": None,
                     "model_path": None,
                     "model_state_dict": None,
                     'batch_size': None}

################################################################################
# Classifier Arguments
downstream_config = {'embedding_dim': 1024,
                     'num_classes': 4,
                     'lr': 0.001,
                     'epoch': 100,
                     'model_save_dir': os.path.join(root_dir, 'checkpoints/downstream'),
                     }