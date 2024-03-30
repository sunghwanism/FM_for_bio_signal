import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")
import args

torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
torch.cuda.manual_seed_all(args.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, drop_out=0.5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.drop_out = nn.Dropout(drop_out)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class RecurrentBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RecurrentBlock, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        """
        x: (Batch, Length, Features)
        """
        x, _ = self.gru(x)
        
        return x

class DeepSense(nn.Module):
    def __init__(self, args):
        super(DeepSense, self).__init__()
        
        self.args = args
        self.config = args.focal_config["backbone"]["DeepSense"]
        self.device = args.base_config["device"]
        self.modalities = args.data_config["modalities"]
        
        # Intialize the encoder
        self.init_encoder()
        
    def init_encoder(self):
        
        self.sample_dim = self.config["recurrent_dim"] * 2 * len(self.modalities)
        self.modality_extractors = nn.ModuleDict()
        
        dims = []
        num_layers = self.config["num_conv_layers"]
        if num_layers > 0:
            for i in range(num_layers):
                dims.append(int(self.config["hidden_dim"]/2**(i+1)))
                
            dims.append(1)
            dims.reverse()
        
        for mod in self.modalities:
            if mod == "ecg":
                if self.config["num_conv_layers"] is None:
                    self.modality_extractors[mod] = ConvBlock(in_channels=1, 
                                                            out_channels=int(self.config["conv_dim"]/2), 
                                                            kernel_size=self.config["mod1_kernel_size"], 
                                                            stride=self.config["mod1_stride"],
                                                            padding=self.config["mod1_padding"],
                                                            )
                else:   
                    conv_blocks = []           
                    for i in range(self.config["num_conv_layers"]):
                        conv_blocks.append(ConvBlock(in_channels=dims[i], 
                                                    out_channels=dims[i+1],
                                                    kernel_size=self.config["mod1_kernel_size"], 
                                                    stride=self.config["mod1_stride"],
                                                    padding=self.config["mod1_padding"],
                                                    ))
                    conv_blocks.append(ConvBlock(in_channels=int(self.config["conv_dim"]/2),
                                                out_channels=self.config["conv_dim"],
                                                kernel_size=self.config["mod1_kernel_size"],
                                                stride=self.config["mod1_stride"],
                                                padding=self.config["mod1_padding"],
                                                ))
                    self.modality_extractors[mod] = nn.Sequential(*conv_blocks)
            else:
                if self.config["num_conv_layers"] is None:
                    self.modality_extractors[mod] = ConvBlock(in_channels=1, 
                                                            out_channels=self.config["conv_dim"], 
                                                            kernel_size=self.config["mod2_ernel_size"], 
                                                            stride=self.config["mod2_stride"],
                                                            padding=self.config["mod2_padding"],
                                                            )
                else:
                    conv_blocks = []           
                    for i in range(self.config["num_conv_layers"]):
                        conv_blocks.append(ConvBlock(in_channels=dims[i], 
                                                    out_channels=dims[i+1],
                                                    kernel_size=self.config["mod2_kernel_size"], 
                                                    stride=self.config["mod2_stride"],
                                                    padding=self.config["mod2_padding"],
                                                    ))
                    conv_blocks.append(ConvBlock(in_channels=int(self.config["conv_dim"]/2),
                                                out_channels=self.config["conv_dim"],
                                                kernel_size=self.config["mod2_kernel_size"],
                                                stride=self.config["mod2_stride"],
                                                padding=self.config["mod2_padding"],
                                                ))
                    self.modality_extractors[mod] = nn.Sequential(*conv_blocks)
                    
            
            # print(f"{mod} extractor is initialized.")
                    
        # Setting GRU
        
        self.recurrent_layer = nn.ModuleDict()
        for mod in self.modalities:
            self.recurrent_layer[mod] = RecurrentBlock(input_size=self.config["conv_dim"], 
                                                        hidden_size=self.config["recurrent_dim"], 
                                                        num_layers=self.config["num_recurrent_layers"])
            
            # print(f"{mod} recurrent layer is initialized.")
            
        
        self.class_layer = nn.Sequential(nn.Linear(self.config["class_in_dim"], self.config["fc_dim"]),
                                        nn.GELU(),
                                        nn.Linear(self.config["fc_dim"], self.config["num_classes"]))
        
        out_dim = self.args.focal_config["embedding_dim"]
        self.mod_projectors = nn.ModuleDict()
        
        
        for mod in self.modalities:
            if mod == "ecg":
                self.mod_projectors[mod] = nn.Sequential(
                nn.Linear(self.config['mod1_linear_dim'], self.config["recurrent_dim"] * 2), # To-do: calculate the linear dim automatically
                nn.Linear(self.config["recurrent_dim"] * 2, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            else:
                self.mod_projectors[mod] = nn.Sequential(
                    nn.Linear(self.config['mod2_linear_dim'], self.config["recurrent_dim"] * 2), # To-do: calculate the linear dim automatically
                    nn.Linear(self.config["recurrent_dim"] * 2, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
        
        print("** Finished Initializing DeepSense Backbone **")
        

    def forward_encoder(self, mod1, mod2, class_head=True, proj_head=False):
        """
        mod1: (Batch length) -> after augmentation
        mod2: (Batch length) -> after augmentaiton
        Augmentation is applied with same method for both modalities.
        """
        # Adding channel dimension: (Batch length) -> (Bacth channel(1) length)
        mod1 = mod1.unsqueeze(1)
        mod2 = mod2.unsqueeze(1)
        
        mod_name_1 = self.args.data_config["modalities"][0]
        mod_name_2 = self.args.data_config["modalities"][1]
        
        # Pass through modality extractors (CNN)
        modality_1_features = self.modality_extractors[mod_name_1](mod1)
        modality_2_features = self.modality_extractors[mod_name_2](mod2)
        
        # print(f"mod1 cnn feature shape: {modality_1_features.shape}", f"mod2 cnn feature shape: {modality_2_features.shape}")
    
        # (Batch, Features, Length) -> (Batch, Length, Features)
        modality_1_features = modality_1_features.transpose(1,2)
        modality_2_features = modality_2_features.transpose(1,2)
        
        # Pass through GRU
        recurrent_1 = self.recurrent_layer[mod_name_1](modality_1_features).flatten(start_dim=1)
        recurrent_2 = self.recurrent_layer[mod_name_2](modality_2_features).flatten(start_dim=1)
        
        
        # print(f"mod1 rnn feature shape: {recurrent_1.shape}", f"mod2 rnn feature shape: {recurrent_2.shape}")
        modality_features = [recurrent_1, recurrent_2]
        
        if not class_head:
            if proj_head:
                sample_features = {}
                for i, mod in enumerate(self.modalities):
                    sample_features[mod] = self.mod_projectors[mod](modality_features[i])
                    
                return sample_features
            
            else:
                return dict(zip(self.modalities, modality_features))
        
        else:
            sample_features = torch.cat(modality_features, dim=1)
            logits = self.class_layer(sample_features)
            
            return logits
        
    
    def forward(self, mod1, mod2, class_head=False, proj_head=True):
        if class_head:
            """Finetuning the classifier"""
            logits = self.forward_encoder(mod1, mod2, class_head)
            return logits
        
        else:
            """Pretraining the framework"""
            enc_mod_features = self.forward_encoder(mod1, mod2, class_head, proj_head)
            
            return enc_mod_features