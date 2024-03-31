
import os
import torch


# Set the root directory
# root_dir = '/data8/jungmin/uot_class/MIE1517_DL/FM_for_bio_signal'
# data_dir = "/data8/jungmin/uot_class/MIE1517_DL/FM_for_bio_signal/src/foundation/dataset"

root_dir = "/NFS/Users/moonsh/FM_biosignal"
data_dir = "/NFS/Users/moonsh/data/mesa/preproc/"
SEED = 42

################################################################################
# Dataset Arguments
"""
Modalities = 'hr', 'ecg', 'activity'
"""
SUBJECT_ID = 0


data_config = {'train_data_dir': os.path.join(data_dir, f'pair_test/subj_{SUBJECT_ID}_train'), # 'pair' is real train data
               'val_data_dir': os.path.join(data_dir, f'pair_test/subj_{SUBJECT_ID}_val'),
               'test_data_dir': os.path.join(data_dir, f'pair_test/subj_{SUBJECT_ID}_test'),
               'modalities': ['ecg', 'hr'],
               'label_key': 'stage',
               'subject_key': 'subject_idx',
               'augmentation': ['NoAugmenter', 'NoAugmenter'],
               'augmenter_config': {
                   'GaussianNoise': {'max_noise_std': 0.1},
                   'AmplitudeScale': {'amplitude_scale': 0.3}
                },
}


trainer_config = {'batch_size': 1024,
                  'epochs': 50,
                  'log_interval': 1,
                  'val_interval': 1,
                  'model_save_dir': os.path.join(root_dir, 'checkpoints_down'),
                  'log_save_dir': os.path.join(root_dir, 'logs_down')
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