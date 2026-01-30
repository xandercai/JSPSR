# https://github.com/rstar000/super-resolution-resnet

from torch import nn
import math
from models.components.spn import Generator, PostProcessor
from models.components.basics import BasicBlock


def get_conv(n_feat, kernel_size, bias=True):
    return nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    """
    A residual block. The block has two convolutions with padding of 1. The size of resulting images is the same.
    Original images are added to the result (thus it is calles "residual").
    Agruments:
        n_feat - the number of features in a convolution
        kernel_size - the kernel size, as you may guess
        bias - boolean. Use a bias or not?
        bn - boolean. Use batch norm? Preferrably not because it is useless and consumes a lot of memory
        act - the activation function
    """

    def __init__(
        self, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1.0
    ):
        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(get_conv(n_feat, kernel_size, bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upscaler(nn.Sequential):
    """
    An upscaler that uses data from convolutions to reshape the tensor.
    Only argument is the number of features in the previous layer.
    The resulting tensor will have the same number of features, but the images will be twice the size.
    """

    def __init__(self, n_feat, scale, bias=True):
        modules = []
        if scale == 2:
            modules.append(nn.Conv2d(n_feat, n_feat * 4, 3, padding=1, bias=bias))
            modules.append(nn.PixelShuffle(2))
        if scale == 4:
            modules.append(nn.Conv2d(n_feat, n_feat * 4, 3, padding=1, bias=bias))
            modules.append(nn.PixelShuffle(2))

        super(Upscaler, self).__init__(*modules)


class EDSR(nn.Module):
    """
    Enhanced resnet model from EDSR+ paper.
    Uses simplified residual blocks. Upscales the image 2X. The input is an image in range [0,1].
    Arguments:
        n_resblocks: The number of residual blocks in the network
        n_features: The number of features in a convolution. All convolutions have the
                    same, high number of features
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_resblocks=16,
        n_features=64,
        scale=2,
        res_scale=0.1,
        spn=False,
    ):
        super().__init__()
        self.url = r"./models/pretrained/EDSR-b32f128x2.bin"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_scale = res_scale
        self.spn = spn

        self.entry = nn.Conv2d(self.in_channels, n_features, 3, padding=1, bias=True)

        resblocks = []
        for i in range(n_resblocks):
            resblocks.append(ResBlock(n_features, 3, res_scale=self.res_scale))
        resblocks.append(nn.Conv2d(n_features, n_features, 3, padding=1, bias=True))
        self.encoder = nn.Sequential(*resblocks)
        if scale > 1:
            self.decoder = Upscaler(n_features, scale)

        if self.spn:
            self.generator = Generator(
                n_features, 3, block=BasicBlock, bc=n_features // 2, leaky=False
            )
            self.post_layer = PostProcessor(3, True)
        else:
            self.head = nn.Conv2d(
                n_features, self.out_channels, 3, padding=1, bias=True
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.spn:
            x_copy = x.clone().detach()
            dem = x_copy[:, 0:1, :, :]
        xs = self.entry(x)  # skip connection
        x = self.encoder(xs)
        x = x + self.res_scale * xs
        if hasattr(self, "decoder"):
            x = self.decoder(x)
        if self.spn:
            weight, offset = self.generator(dem, x)
            x = self.post_layer(dem, weight, offset)
        else:
            x = self.head(x)
        return x
