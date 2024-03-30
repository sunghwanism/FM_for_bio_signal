
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
base_config = {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}

data_config = {'train_data_dir': os.path.join(data_dir, 'pair_train'), # 'pair' is real train data
               'val_data_dir': os.path.join(data_dir, 'pair_val'),
               'test_data_dir': os.path.join(data_dir, 'pair_test'), 
               'modalities': ['ecg','hr'],           
}


pretrain_config = {'epoch': 100,
                   'batch_size': 256,
                   'optimizer': 'Adam',
                   'lr': 0.001,
                   'weight_decay': 0.0001,
                   'model_save_dir': os.path.join(root_dir, 'checkpoints'),
                   'val_freq': 1,
                   'model_name': 
                          {'DeepSense': {'mod1_kernel_size': 11,
                                         'mod1_stride': 3,
                                         'mod1_padding': 0,
                                         'mod2_kernel_size': 3,
                                         'mod2_stride': 1,
                                         'mod2_padding': 1,
                                         'num_conv_layers': 2,
                                         'conv_dim': 64,
                                         'num_recurrent_layers': 2,
                                         'recurrent_dim': 64,
                                         'hidden_dim': 64,
                                         'mod1_linear_dim': 17920,
                                         'mod2_linear_dim': 1920,
                                         'num_classes': 5, # in SSL -> Embedding Dimension / in Supervised -> Number of Classes
                                         'fc_dim': 64
                                         }
                             },
                   }


model_save_format = {"train_acc": None,
                     "val_acc": None,
                     "train_loss": None,
                     "val_loss": None,
                     "epoch": None,
                     "lr": None,
                     "model_path": None,
                     "model_state_dict": None,}


focal_config = {'backbone': 
                          {'DeepSense': {'mod1_kernel_size': 11,
                                         'mod1_stride': 3,
                                         'mod1_padding': 0,
                                         'mod2_kernel_size': 3,
                                         'mod2_stride': 1,
                                         'mod2_padding': 1,
                                         'num_conv_layers': 2,
                                         'conv_dim': 64,
                                         'num_recurrent_layers': 2,
                                         'recurrent_dim': 64,
                                         'hidden_dim': 64,
                                         'mod1_linear_dim': 17920,
                                         'mod2_linear_dim': 1920,
                                         'num_classes': 4, # in SSL -> Embedding Dimension / in Supervised -> Number of Classes
                                         'fc_dim': 64
                                         }
                             },
                'tag': 'usePrivate', # 'noPrivate' for not using private loss
                'embedding_dim': 64,
                'num_subjects': 100,
                'dropout_rate': 0.5,
                'lr': 0.001,
                'shared_contrastive_loss_weight': 0.5,
                'private_contrastive_loss_weight': 0.5,
                'orthogonality_loss_weight': 0.2,
                'subject_invariant_loss_weight': 0.2,
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

focal_config = {'backbone': 
                          {'DeepSense': {'mod1_kernel_size': 11,
                                         'mod1_stride': 3,
                                         'mod1_padding': 0,
                                         'mod2_kernel_size': 3,
                                         'mod2_stride': 1,
                                         'mod2_padding': 1,
                                         'num_conv_layers': 2,
                                         'conv_dim': 64,
                                         'num_recurrent_layers': 2,
                                         'recurrent_dim': 64,
                                         'hidden_dim': 64,
                                         'mod1_linear_dim': 17920,
                                         'mod2_linear_dim': 1920,
                                         'num_classes': 5, # in SSL -> Embedding Dimension / in Supervised -> Number of Classes
                                         'fc_dim': 64
                                         }
                             },
                'tag': 'usePrivate', # 'noPrivate' for not using private loss
                'embedding_dim': 64,
                'num_subjects': 100,
                'dropout_rate': 0.5,
                'lr': 0.001,
                'shared_contrastive_loss_weight': 0.5,
                'private_contrastive_loss_weight': 0.5,
                'orthogonality_loss_weight': 0.2,
                'subject_invariant_loss_weight': 0.2,
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}