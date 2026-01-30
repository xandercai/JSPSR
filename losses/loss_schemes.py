import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(name):
    """Return loss function for a specific task"""
    if name.lower() == "l1":
        loss_fn = torch.nn.L1Loss()
    elif name.lower() == "l2" or name.lower() == "mse":
        loss_fn = torch.nn.MSELoss()
    elif name.lower() == "vanilla" or name.lower() == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif name.lower() == "edge" or name.lower() == "grad":
        from losses.loss_functions import EdgeLoss

        loss_fn = EdgeLoss()
    elif name.lower() == "berhu":
        from losses.loss_functions import BerhuLoss

        loss_fn = BerhuLoss()
    elif name.lower() == "norm":
        from losses.loss_functions import SurfaceNormalLoss

        loss_fn = SurfaceNormalLoss()
    elif name.lower() == "ssim":
        from losses.loss_functions import SSIMLoss

        loss_fn = SSIMLoss()

    else:
        raise NotImplementedError(f"Undefined loss: {name}")
    return loss_fn


class SingleLoss(nn.Module):
    def __init__(self, **loss_dict: dict):
        super(SingleLoss, self).__init__()
        self.loss, loss_kwargs = loss_dict.popitem()
        self.loss_fn = loss_kwargs["loss_fn"]
        self.out = {}

    def forward(self, pred, gt):
        self.out = {self.loss: self.loss_fn(pred, gt)}
        self.out["Total"] = self.out[self.loss]
        return self.out

    def reset(self):
        self.out = {}

    def __str__(self):
        return f"{self.__class__.__name__}:: {self.loss}:: {self.loss_fn}"


class MultiLoss(nn.Module):
    def __init__(self, **loss_dict: dict):
        super(MultiLoss, self).__init__()
        self.loss_dict = loss_dict
        self.out = {}

    def forward(self, pred, gt):
        self.out = {
            loss: self.loss_dict[loss]["loss_fn"](pred, gt)
            for loss in self.loss_dict.keys()
        }
        self.out.pop("Total", None)  # remove total if it exists, to avoid duplicates
        self.out["Total"] = torch.sum(
            torch.stack(
                [self.loss_dict[loss]["weight"] * self.out[loss] for loss in self.out]
            )
        )
        return self.out

    def reset(self):
        self.out = {}

    def __str__(self):
        return (
            f"{self.__class__.__name__}:: "
            f"{[self.loss_dict.keys()]}, "
            f"{[d['weight'] for d in self.loss_dict.values()]}, "
            f"{[d['loss_fn'] for d in self.loss_dict.values()]}"
        )
