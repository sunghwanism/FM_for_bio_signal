
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

data_config = {'train_data_dir': os.path.join(data_dir, 'pair_train'), # 'pair' is real train data
               'val_data_dir': os.path.join(data_dir, 'pair_val'),
               'test_data_dir': os.path.join(data_dir, 'pair_test'),               
}


pretrain_config = {'epoch': 100,
                   'batch_size': 256,
                   'optimizer': 'Adam',
                   'lr': 0.001,
                   'weight_decay': 0.0001,
                   'model_save_dir': os.path.join(root_dir, 'logs'),
                   'val_freq': 1,
                   'model_name': "LSTM"
                   }


model_save_format = {"train_acc": None,
                     "val_acc": None,
                     "train_loss": None,
                     "val_loss": None,
                     "epoch": None,
                     "lr": None,
                     "model_path": None,
                     "model_state_dict": None,}