import torch
import torch.nn as nn
import numpy as np
from input_utils.mixup_utils import Mixup

class NoAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """None missing modality generator"""
        super().__init__()
        self.args = args

    def forward(self, loc_inputs, labels=None):
        """
        Fake forward function of the no miss modality generator.
        x: loc --> mod --> [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        return loc_inputs, None, labels



class MixupAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """mixup and cutmix augmentation, does nothing if both has alpha 0"""
        super().__init__()
        self.args = args
        self.config = args.dataset_config["mixup"]
        self.config["num_classes"] = args.dataset_config[args.task]["num_classes"]
        self.mixup_func = Mixup(**args.dataset_config["mixup"])

        if "regression" in args.task:
            raise Exception("Mixup is not supported for regression task.")

    def forward(self, org_loc_inputs, labels=None):
        """
        Fake forward function of the no miss modality generator.
        x: [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        # TODO: Contrastive learning mixup, mixup function with no labels
        aug_loc_inputs, aug_labels = self.mixup_func(org_loc_inputs, labels, self.args.dataset_config)

        return aug_loc_inputs, None, aug_labels
    
    def _mix_pair(self, x, args):
        """Original timm implementation + location and modality integration"""
        lam_batches = []
        for loc in args["location_names"]:
            for mod in args["modality_names"]:
                batch_size = len(x[loc][mod])
                lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
                x_orig = x[loc][mod].clone()  # need to keep an unmodified original for mixing source
                for i in range(batch_size // 2):
                    j = batch_size - i - 1
                    lam = lam_batch[i]
                    if lam != 1.0:
                        if use_cutmix[i]:
                            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                                x[loc][mod][i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam
                            )
                            x[loc][mod][i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                            x[loc][mod][j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                            lam_batch[i] = lam
                        else:
                            x[loc][mod][i] = x[loc][mod][i] * lam + x_orig[j] * (1 - lam)
                            x[loc][mod][j] = x[loc][mod][j] * lam + x_orig[i] * (1 - lam)
                lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
                lam_batch = torch.tensor(lam_batch, device=x.device, dtype=x[loc][mod].dtype).unsqueeze(1)
                lam_batches.append(lam_batch)
                
        return torch.mean(lam_batches, axis=0), None
