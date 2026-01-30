import torch
import torch.nn as nn
from scipy.stats import truncnorm
import math
from models.components.basics import Basic2d, BasicBlock, Basic2dTrans, Guide, conv1x1
from models.components.spn import Generator, PostProcessor


class Model(nn.Module):
    def __init__(
        self,
        in_channels: dict,
        out_channels: int = 1,
        num_feature: int = 32,
        layers: tuple = (2, 2, 2, 2),
        res_scale: tuple = (1, 1, 1, 1),
        spn: bool = True,
        spn_scale: int = 1,
    ):
        super().__init__()
        self.name = "JSPSR"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spn = spn
        self.spn_scale = spn_scale
        assert len(in_channels) > 1, "At least 2 input data are required"

        self.cat_only = True  # experimental option for guided image fusion
        generator_leaky = False  # experimental option for activation layer in SPN

        block = BasicBlock
        guide = Guide

        dem_channels = self.in_channels["lr_dem"]
        num_branch = 1

        img_channels = None
        self.flag_dem_img = False
        if "image" in self.in_channels:
            img_channels = self.in_channels["image"]
            num_branch += 1
            self.flag_dem_img = True

        msk_channels = None
        self.flag_dem_msk = False
        if "mask" in self.in_channels:
            msk_channels = self.in_channels["mask"]
            num_branch += 1
            self.flag_dem_msk = True

        canopy_channels = None
        self.flag_dem_canopy = False
        if "canopy" in self.in_channels:
            canopy_channels = self.in_channels["canopy"]
            num_branch += 1
            self.flag_dem_canopy = True

        # coordinate channels is experimental option and dose not demonstrated in the paper
        coord_channels = None
        self.flag_dem_coord = False
        if "coord" in self.in_channels:
            coord_channels = self.in_channels["coord"]
            num_branch += 1
            self.flag_dem_coord = True

        self.conv_dem = Basic2d(
            dem_channels, num_feature, kernel_size=5, padding=2, bn=False
        )
        self.conv_img = (
            Basic2d(img_channels, num_feature, kernel_size=5, padding=2, bn=True)
            if self.flag_dem_img
            else None
        )
        if self.flag_dem_msk:
            self.conv_aux = Basic2d(
                msk_channels, num_feature, kernel_size=5, padding=2, bn=False
            )
        elif self.flag_dem_canopy:
            self.conv_aux = Basic2d(
                canopy_channels, num_feature, kernel_size=5, padding=2, bn=False
            )
        elif self.flag_dem_coord:
            self.conv_aux = Basic2d(
                coord_channels, num_feature, kernel_size=5, padding=2, bn=False
            )
        else:
            self.conv_aux = None

        self.inplanes = num_feature
        self.layer1_dem, self.layer1_img, self.layer1_aux = self._make_layer(
            block,
            num_feature * 2,
            layers[0],
            stride=1,
            res_scale=res_scale[0],
            num_branch=1,
            cat_only=self.cat_only,
        )  # 32 -> 64
        self.guide1 = guide(
            num_feature * 2 * num_branch, num_feature * 2, cat_only=self.cat_only
        )

        self.inplanes = num_feature * 2 * block.expansion
        self.layer2_dem, self.layer2_img, self.layer2_aux = self._make_layer(
            block,
            num_feature * 4,
            layers[1],
            stride=2,
            res_scale=res_scale[1],
            num_branch=num_branch,
            cat_only=self.cat_only,
        )  # 64 -> 128
        self.guide2 = guide(
            num_feature * 4 * num_branch, num_feature * 4, cat_only=self.cat_only
        )

        self.inplanes = num_feature * 4 * block.expansion
        self.layer3_dem, self.layer3_img, self.layer3_aux = self._make_layer(
            block,
            num_feature * 8,
            layers[2],
            stride=2,
            res_scale=res_scale[2],
            num_branch=num_branch,
            cat_only=self.cat_only,
        )  # 128 -> 256
        self.guide3 = guide(
            num_feature * 8 * num_branch, num_feature * 8, cat_only=self.cat_only
        )

        self.inplanes = num_feature * 8 * block.expansion
        self.layer4_dem, self.layer4_img, self.layer4_aux = self._make_layer(
            block,
            num_feature * 16,
            layers[3],
            stride=2,
            res_scale=res_scale[3],
            num_branch=num_branch,
            cat_only=self.cat_only,
        )  # 256 -> 512
        self.guide4 = guide(
            num_feature * 16 * num_branch, num_feature * 16, cat_only=self.cat_only
        )

        c4_channels = (
            num_feature * 16 * num_branch * block.expansion
            if self.cat_only
            else num_feature * 16 * block.expansion
        )
        self.layer3d = Basic2dTrans(c4_channels, num_feature * 8, camb=self.cat_only)

        c3_channels = (
            num_feature * 8 + num_feature * 8 * num_branch
            if self.cat_only
            else num_feature * 8
        )
        self.layer2d = Basic2dTrans(c3_channels, num_feature * 4, camb=self.cat_only)

        c2_channels = (
            num_feature * 4 + num_feature * 4 * num_branch
            if self.cat_only
            else num_feature * 4
        )
        self.layer1d = Basic2dTrans(c2_channels, num_feature * 2, camb=self.cat_only)

        c1_channels = (
            num_feature * 2 + num_feature * 2 * num_branch
            if self.cat_only
            else num_feature * 2
        )
        c0_channels = num_feature * 2 if self.cat_only else num_feature
        self.conv0 = Basic2d(
            c1_channels,
            c0_channels,
            kernel_size=3,
            padding=1,
            bn=True,
            relu=True,
            camb=self.cat_only,
        )
        bc = num_feature if self.cat_only else num_feature // 2

        if self.spn:
            self.generator = Generator(
                in_channels=c0_channels,
                kernel_size=3,
                block=BasicBlock,  #
                bc=bc,
                leaky=generator_leaky,
            )

            self.postprocessor = PostProcessor(
                kernel_size=3, residual=True, scale=self.spn_scale
            )
        else:
            self.generator = None
            self.postprocessor = Basic2d(
                c0_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bn=False,
                relu=False,
            )

        self._initialize_weights()

    def forward(self, *in_tensor):
        # input order is important: [dem, img, msk, canopy, coord]
        # dem is required, and
        # at least one of img, msk or canopy should be provided
        dem, img, msk, canopy, coord = self.parse_input(
            self.flag_dem_img,
            self.flag_dem_msk,
            self.flag_dem_canopy,
            self.flag_dem_coord,
            *in_tensor,
        )

        c0_dem = self.conv_dem(dem)  # 1 -> 32, 1 -> 1
        c0_img = self.conv_img(img) if self.conv_img else None
        c0_msk = self.conv_aux(msk) if self.conv_aux and self.flag_dem_msk else None
        c0_canopy = (
            self.conv_aux(canopy) if self.conv_aux and self.flag_dem_canopy else None
        )
        c0_coord = (
            self.conv_aux(coord) if self.conv_aux and self.flag_dem_coord else None
        )

        c1_dem = self.layer1_dem(c0_dem)  # 32 -> 64, 1 -> 1
        c1_img = self.layer1_img(c0_img) if self.layer1_img else None
        c1_msk = (
            self.layer1_aux(c0_msk) if self.layer1_aux and self.flag_dem_msk else None
        )
        c1_canopy = (
            self.layer1_aux(c0_canopy)
            if self.layer1_aux and self.flag_dem_canopy
            else None
        )
        c1_coord = (
            self.layer1_aux(c0_coord)
            if self.layer1_aux and self.flag_dem_coord
            else None
        )

        if self.flag_dem_img and self.flag_dem_msk:
            c1_fuse = self.guide1(c1_dem, c1_img, c1_msk)  # 64 * 3,  1 -> 1
        elif self.flag_dem_img and self.flag_dem_canopy:
            c1_fuse = self.guide1(c1_dem, c1_img, c1_canopy)  # 64 * 3,  1 -> 1
        elif self.flag_dem_img and self.flag_dem_coord:
            c1_fuse = self.guide1(c1_dem, c1_img, c1_coord)  # 64 * 3,  1 -> 1
        elif self.flag_dem_img:
            c1_fuse = self.guide1(c1_dem, c1_img)  # 64 * 2, 1 -> 1
        elif self.flag_dem_msk:
            c1_fuse = self.guide1(c1_dem, c1_msk)  # 64 * 2, 1 -> 1
        elif self.flag_dem_canopy:
            c1_fuse = self.guide1(c1_dem, c1_canopy)  # 64 * 2, 1 -> 1
        else:
            raise NotImplementedError

        c2_dem = self.layer2_dem(c1_fuse)  # 64*2(3) -> 128, 1 -> 1/2
        c2_img = self.layer2_img(c1_img) if self.layer2_img else None
        c2_msk = (
            self.layer2_aux(c1_msk) if self.layer2_aux and self.flag_dem_msk else None
        )
        c2_canopy = (
            self.layer2_aux(c1_canopy)
            if self.layer2_aux and self.flag_dem_canopy
            else None
        )
        c2_coord = (
            self.layer2_aux(c1_coord)
            if self.layer2_aux and self.flag_dem_coord
            else None
        )

        if self.flag_dem_img and self.flag_dem_msk:
            c2_fuse = self.guide2(c2_dem, c2_img, c2_msk)  # 128 * 3, 1/2 -> 1/2
        elif self.flag_dem_img and self.flag_dem_canopy:
            c2_fuse = self.guide2(c2_dem, c2_img, c2_canopy)  # 128 * 3, 1/2 -> 1/2
        elif self.flag_dem_img and self.flag_dem_coord:
            c2_fuse = self.guide2(c2_dem, c2_img, c2_coord)  # 128 * 3, 1/2 -> 1/2
        elif self.flag_dem_img:
            c2_fuse = self.guide2(c2_dem, c2_img)  # 128 * 2, 1/2 -> 1/2
        elif self.flag_dem_msk:
            c2_fuse = self.guide2(c2_dem, c2_msk)  # 128 * 2, 1/2 -> 1/2
        elif self.flag_dem_canopy:
            c2_fuse = self.guide2(c2_dem, c2_canopy)  # 128 * 2, 1/2 -> 1/2
        else:
            raise NotImplementedError

        c3_dem = self.layer3_dem(c2_fuse)  # 128*2(3) -> 256, 1/2 -> 1/4
        c3_img = self.layer3_img(c2_img) if self.layer3_img else None
        c3_msk = (
            self.layer3_aux(c2_msk) if self.layer3_aux and self.flag_dem_msk else None
        )
        c3_canopy = (
            self.layer3_aux(c2_canopy)
            if self.layer3_aux and self.flag_dem_canopy
            else None
        )
        c3_coord = (
            self.layer3_aux(c2_coord)
            if self.layer3_aux and self.flag_dem_coord
            else None
        )

        if self.flag_dem_img and self.flag_dem_msk:
            c3_fuse = self.guide3(c3_dem, c3_img, c3_msk)  # 256 * 3, 1/4 -> 1/4
        elif self.flag_dem_img and self.flag_dem_canopy:
            c3_fuse = self.guide3(c3_dem, c3_img, c3_canopy)  # 256 * 3, 1/4 -> 1/4
        elif self.flag_dem_img and self.flag_dem_coord:
            c3_fuse = self.guide3(c3_dem, c3_img, c3_coord)  # 256 * 3, 1/4 -> 1/4
        elif self.flag_dem_img:
            c3_fuse = self.guide3(c3_dem, c3_img)  # 256 * 2, 1/4 -> 1/4
        elif self.flag_dem_msk:
            c3_fuse = self.guide3(c3_dem, c3_msk)  # 256 * 2, 1/4 -> 1/4
        elif self.flag_dem_canopy:
            c3_fuse = self.guide3(c3_dem, c3_canopy)  # 256 * 2, 1/4 -> 1/4
        else:
            raise NotImplementedError

        c4_dem = self.layer4_dem(c3_fuse)  # 256*2(3) -> 512, 1/4 -> 1/8
        c4_img = self.layer4_img(c3_img) if self.layer4_img else None
        c4_msk = (
            self.layer4_aux(c3_msk) if self.layer4_aux and self.flag_dem_msk else None
        )
        c4_canopy = (
            self.layer4_aux(c3_canopy)
            if self.layer4_aux and self.flag_dem_canopy
            else None
        )
        c4_coord = (
            self.layer4_aux(c3_coord)
            if self.layer4_aux and self.flag_dem_coord
            else None
        )

        if self.flag_dem_img and self.flag_dem_msk:
            c4 = self.guide4(c4_dem, c4_img, c4_msk)  # 512 * 3, 1/8 -> 1/8
        elif self.flag_dem_img and self.flag_dem_canopy:
            c4 = self.guide4(c4_dem, c4_img, c4_canopy)  # 512 * 3, 1/8 -> 1/8
        elif self.flag_dem_img and self.flag_dem_coord:
            c4 = self.guide4(c4_dem, c4_img, c4_coord)  # 512 * 3, 1/8 -> 1/8
        elif self.flag_dem_img:
            c4 = self.guide4(c4_dem, c4_img)  # 512 * 2, 1/8 -> 1/8
        elif self.flag_dem_msk:
            c4 = self.guide4(c4_dem, c4_msk)  # 512 * 2, 1/8 -> 1/8
        elif self.flag_dem_canopy:
            c4 = self.guide4(c4_dem, c4_canopy)  # 512 * 2, 1/8 -> 1/8
        else:
            raise NotImplementedError

        dc3 = self.layer3d(c4)
        if self.cat_only:
            c3 = torch.cat((dc3, c3_fuse), dim=1)
        else:
            c3 = dc3 + c3_fuse
        dc2 = self.layer2d(c3)
        if self.cat_only:
            c2 = torch.cat((dc2, c2_fuse), dim=1)
        else:
            c2 = dc2 + c2_fuse
        dc1 = self.layer1d(c2)
        if self.cat_only:
            c1 = torch.cat((dc1, c1_fuse), dim=1)
        else:
            c1 = dc1 + c1_fuse
        c0 = self.conv0(c1)

        if self.spn:
            dem = dem.detach()
            weight, offset = self.generator(dem, c0)

            output = self.postprocessor(dem, weight, offset)

        else:
            output = self.postprocessor(c0)

        return output

    def _make_layer(
        self,
        block,
        planes,
        layers,
        stride=1,
        act=True,
        res_scale=1.0,
        num_branch=2,
        cat_only=True,
    ):
        dem_downsample, img_downsample, aux_downsample = None, None, None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dem_downsample = nn.Sequential(
                conv1x1(
                    self.inplanes * num_branch if cat_only else self.inplanes,
                    planes * block.expansion,
                    stride,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
            img_downsample = (
                nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
                if self.flag_dem_img
                else None
            )
            aux_downsample = (
                nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
                if self.flag_dem_msk or self.flag_dem_coord or self.flag_dem_canopy
                else None
            )

        dem_layers = [
            block(
                self.inplanes * num_branch if cat_only else self.inplanes,
                planes,
                stride,
                dem_downsample,
                act=act,
                scale=res_scale,
            )
        ]
        img_layers = [
            block(
                self.inplanes,
                planes,
                stride,
                img_downsample,
                act=act,
                scale=res_scale,
            )
        ]
        aux_layers = [
            block(
                self.inplanes,
                planes,
                stride,
                aux_downsample,
                act=act,
                scale=res_scale,
            )
        ]

        for _ in range(1, layers):
            dem_layers.append(
                block(
                    planes,
                    planes,
                    stride=1,
                    downsample=None,
                    act=act,
                    scale=res_scale,
                )
            )
            img_layers.append(
                block(
                    planes,
                    planes,
                    stride=1,
                    downsample=None,
                    act=act,
                    scale=res_scale,
                )
            )
            aux_layers.append(
                block(
                    planes,
                    planes,
                    stride=1,
                    downsample=None,
                    act=act,
                    scale=res_scale,
                )
            )

        ret = [nn.Sequential(*dem_layers)]
        if self.flag_dem_img:
            ret.append(nn.Sequential(*img_layers))
        else:
            ret.append(None)
        if self.flag_dem_msk or self.flag_dem_coord or self.flag_dem_canopy:
            ret.append(nn.Sequential(*aux_layers))
        else:
            ret.append(None)
        return tuple(ret)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0.0, std=1.0):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm(
                (lower - mean) / std, (upper - mean) / std, loc=mean, scale=std
            )
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(
                    m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2.0 / n)
                )
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def parse_input(
        flag_dem_img, flag_dem_msk, flag_dem_canopy, flag_dem_coord, *in_tensor
    ):
        dem, img, msk, canopy, coord = None, None, None, None, None
        assert (
            flag_dem_msk or flag_dem_img or flag_dem_canopy
        ), "At least one of image or mask is required"
        if len(in_tensor) == 3:
            if flag_dem_img and flag_dem_msk:
                dem, img, msk = in_tensor
            elif flag_dem_img and flag_dem_canopy:
                dem, img, canopy = in_tensor
            elif flag_dem_img and flag_dem_coord:
                dem, img, coord = in_tensor
            else:
                raise NotImplementedError
        elif len(in_tensor) == 2:
            if flag_dem_img:
                dem, img = in_tensor
            elif flag_dem_msk:
                dem, msk = in_tensor
            elif flag_dem_canopy:
                dem, canopy = in_tensor
            elif flag_dem_coord:
                dem, coord = in_tensor
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return dem, img, msk, canopy, coord
