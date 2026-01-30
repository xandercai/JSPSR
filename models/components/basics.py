import torch
import torch.nn as nn
from models.components.resnet_cbam import ChannelAttention


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=(1, 1),
        bias=False,
    )


class Basic2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        bn=True,
        relu=True,
        camb=False,
        leaky=False,
    ):
        super().__init__()
        if camb:
            self.camb = ChannelAttention(in_channels, ratio=16)
        bias = not bn
        conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )
        self.conv = nn.Sequential(conv2d)
        if bn:
            self.conv.add_module("bn", nn.BatchNorm2d(out_channels))
        if relu:
            if leaky:
                self.conv.add_module("relu", nn.LeakyReLU(0.2, inplace=True))
            else:
                self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        if hasattr(self, "camb"):
            x = self.camb(x) * x
        out = self.conv(x)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, camb=False):
        super().__init__()
        conv = Basic2d(
            in_channels, out_channels, kernel_size=3, padding=1, bn=bn, camb=camb
        )
        dconv2d = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=not bn,
        )
        self.dconv = nn.Sequential(conv, dconv2d)
        if bn:
            self.dconv.add_module("bn", nn.BatchNorm2d(out_channels))
        self.dconv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.dconv(x)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        act=True,
        scale=1.0,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act
        self.scale = scale

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out * self.scale + residual
        if self.act:
            out = self.relu(out)
        return out


class Guide(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, cat_only=True):
        super().__init__()
        if not cat_only:
            self.conv = Basic2d(
                in_channels, out_channels, kernel_size=3, padding=1, bn=bn
            )

    def forward(self, *feat):
        out = torch.cat(tuple(feat), dim=1)
        if hasattr(self, "conv"):
            out = self.conv(out)
        return out
