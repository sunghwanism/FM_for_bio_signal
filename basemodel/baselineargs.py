
import os
import torch

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Set the root directory
root_dir = "/NFS/Users/moonsh/FM_biosignal"
data_dir = "/NFS/Users/moonsh/data/mesa/preproc/"
SEED = 42
################################################################################
# General Arguments
"""
Modalities = 'hr', 'ecg', 'activity'
"""

SUBJECT_ID = "0558" # ["0558", "0560", "0565", "0571", "0583", "0586", "0590", "0614", "0621", "0626"]


data_config = {'train_data_dir': os.path.join(data_dir, f'pair_test_subj/subj_{SUBJECT_ID}_train'), # 'pair' is real train data
               'val_data_dir': os.path.join(data_dir, f'pair_test_subj/subj_{SUBJECT_ID}_val'),
               'test_data_dir': os.path.join(data_dir, f'pair_test_subj/subj_{SUBJECT_ID}_test'),
               'modalities': ['ecg', 'hr'],
               'label_key': 'stage',
               'subject_key': 'subject_idx',
               'augmentation': ['NoAugmenter', 'NoAugmenter'],
               'augmenter_config': {
                   'GaussianNoise': {'max_noise_std': 0.1},
                   'AmplitudeScale': {'amplitude_scale': 0.3}
                },
}


trainer_config = {'epoch': 100,
                   'batch_size': 1024,
                   'optimizer': 'Adam',
                   'lr': 0.001,
                   'weight_decay': 0.0001,
                   'model_save_dir': os.path.join(root_dir, 'checkpoints'),
                   'model_name': 
                          {'DeepSense': {'mod1_kernel_size': 11,
                                         'mod1_stride': 3,
                                         'mod1_padding': 0,
                                         ############################
                                         'mod2_kernel_size': 3,
                                         'mod2_stride': 1,
                                         'mod2_padding': 1,
                                         ############################
                                         'num_conv_layers': 2,
                                         'conv_dim': 512,
                                         'conv_dropout_rate': 0.5,
                                         ############################
                                         'num_recurrent_layers': 2,
                                         'recurrent_dim': 512,
                                         ############################
                                         'mod1_linear_dim': 17920,
                                         'mod2_linear_dim': 1920,
                                         ############################
                                         'embedding_dim': 512,
                                         'class_layer_dim': 158720,
                                         'fc_dim': 256,
                                         'proj_dropout_rate': 0.5,
                                         'num_classes': 4,
                                         }
                             },
                     'val_interval': 1,
                     'model_save_dir': os.path.join(root_dir, f'checkpoints_subj_{SUBJECT_ID}_ind'),
                     'log_save_dir': os.path.join(root_dir, f'logs_subj_{SUBJECT_ID}_ind'),
                     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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