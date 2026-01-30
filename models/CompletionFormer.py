"""
CompletionFormer
======================================================================

CompletionFormer implementation
"""

from models.components.nlspn import NLSPN
from models.components.completion_former_backbone import Backbone
import torch
import torch.nn as nn
from models import JSPSR


class Model(nn.Module):
    def __init__(self, args: dict):
        super(Model, self).__init__()
        self.in_channels = args.input_channels

        # for TorchInfo

        self.flag_dem_img = self.in_channels.get("image", 0) > 0
        self.flag_dem_msk = self.in_channels.get("mask", 0) > 0
        self.flag_dem_coord = self.in_channels.get("coord", 0) > 0
        self.flag_dem_canopy = self.in_channels.get("canopy", 0) > 0

        self.args = args
        self.prop_time = self.args.prop_time
        self.num_neighbors = self.args.prop_kernel * self.args.prop_kernel - 1

        self.backbone = Backbone(args, mode="rgbd")

        if self.prop_time > 0:
            self.prop_layer = NLSPN(
                args, self.num_neighbors, 1, 3, self.args.prop_kernel
            )

    def forward(self, *in_tensor):
        # input order is important: [dem, img, msk, coord]
        dep, rgb, _, _, _ = JSPSR.Model.parse_input(
            self.flag_dem_img,
            self.flag_dem_msk,
            self.flag_dem_coord,
            self.flag_dem_canopy,
            *in_tensor
        )

        pred_init, guide, confidence = self.backbone(rgb, dep)
        pred_init = pred_init + dep

        # Diffusion
        y_inter = [
            pred_init,
        ]
        conf_inter = [
            confidence,
        ]
        if self.prop_time > 0:
            y, y_inter, offset, aff, aff_const = self.prop_layer(
                pred_init, guide, confidence, dep, rgb
            )
        else:
            y = pred_init
            offset, aff, aff_const = (
                torch.zeros_like(y),
                torch.zeros_like(y),
                torch.zeros_like(y).mean(),
            )

        #  # Remove negative depth
        #  y = torch.clamp(y, min=0)
        #  # best at first
        #  y_inter.reverse()
        #  conf_inter.reverse()
        #  if not self.args.conf_prop:
        #      conf_inter = None

        #  output = {
        #      "pred": y,
        #      "pred_init": pred_init,
        #      "pred_inter": y_inter,
        #      "guidance": guide,
        #      "offset": offset,
        #      "aff": aff,
        #      "gamma": aff_const,
        #      "confidence": conf_inter,
        #  }

        return y
