import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from models.components.basics import Basic2d


# Weight/Affinity and Offset generation
class Generator(nn.Module):
    def __init__(self, in_channels, kernel_size, block, bc=16, leaky=False):
        super().__init__()
        self.kernel_size = kernel_size
        # Assume zero offset for center pixels
        self.num = kernel_size * kernel_size - 1
        self.idx_ref = self.num // 2

        self.convd1 = Basic2d(
            1, bc * 2, bn=False, kernel_size=3, padding=1, relu=True, leaky=leaky
        )
        self.convd2 = Basic2d(
            bc * 2, bc * 2, bn=False, kernel_size=3, padding=1, relu=True, leaky=leaky
        )

        self.convf1 = Basic2d(
            in_channels,
            bc * 2,
            bn=False,
            kernel_size=3,
            padding=1,
            relu=True,
            leaky=leaky,
        )
        self.convf2 = Basic2d(
            bc * 2, bc * 2, bn=False, kernel_size=3, padding=1, relu=True, leaky=leaky
        )

        self.conv = Basic2d(
            bc * 4, bc * 4, bn=False, kernel_size=3, padding=1, relu=True, leaky=leaky
        )
        self.block = block(bc * 4, bc * 4)

        self.conv_weight = nn.Sequential(
            nn.Conv2d(bc * 4, self.kernel_size**2, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.conv_offset = Basic2d(
            bc * 4,
            2 * (self.kernel_size**2 - 1),
            kernel_size=1,
            padding=0,
            bn=False,
            relu=False,  # no relu is slightly better
        )

    def forward(self, dem, context):
        B, _, H, W = dem.shape

        d1 = self.convd1(dem)
        d2 = self.convd2(d1)

        f1 = self.convf1(context)
        f2 = self.convf2(f1)

        input_feature = torch.cat((d2, f2), dim=1)
        input_feature = self.conv(input_feature)
        feature = self.block(input_feature)
        weight = self.conv_weight(feature)
        offset = self.conv_offset(feature)

        # Add zero reference offset
        offset = offset.view(B, self.num, 2, H, W)
        list_offset = list(torch.chunk(offset, self.num, dim=1))
        list_offset.insert(self.idx_ref, torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

        return weight, offset


# Post-processing to refine the DEM using the generated weight and offset
class PostProcessor(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        residual=True,
        scale=1.0,
    ):
        super().__init__()
        self.residual = residual
        self.w = nn.Parameter(torch.ones((1, 1, kernel_size, kernel_size)))
        self.b = nn.Parameter(torch.zeros(1))
        self.stride = (1, 1)
        self.padding = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)
        self.dilation = (1, 1)
        self.scale = scale
        if self.scale != 1:
            print(
                "Warning: The scale factor is not 1. This may lead to unexpected results."
            )

    def forward(self, init_dem, weight, offset):
        if self.residual:
            weight = weight - torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight = weight / torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        refined_dem = deform_conv2d(
            init_dem,
            offset,
            weight=self.w,
            bias=self.b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=weight,
        )
        out = refined_dem
        if self.residual:
            out = out + self.scale * init_dem
        return out
