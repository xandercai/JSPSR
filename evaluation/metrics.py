from typing import Union
import math
import torch
import numpy as np
import piq
import skimage.metrics
from data.data_utils import RGB2YCbCr
from kornia.filters import spatial_gradient
from torch.autograd import Variable
import torch.nn.functional as F
import richdem as rd
from hide_warnings import hide_warnings

# from pyproj import CRS
from data.data_utils import ToDEM, ToTensor

"""SSIM Loss function taken from: https://github.com/Po-Hsun-Su/pytorch-ssim"""


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            math.exp(-(x - window_size // 2) * 2 / float(2 * sigma * 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(pred, gt, border=0):
    """
    Calculate PSNR between prediction and ground truth, input 0 to 1 only
    :param pred: prediction [0,1]
    :param gt: ground truth [0,1]
    :param border: border to crop from the image
    """
    h, w = pred.shape[:2]
    pred = pred[border : h - border, border : w - border]
    gt = gt[border : h - border, border : w - border]
    mse = np.mean(
        (np.array(gt, dtype=np.float32) - np.array(pred, dtype=np.float32)) ** 2
    )
    if mse == 0:
        return 100
    return 20 * np.log10(1 / (np.sqrt(mse)))


# PyTorch implementation of Sobel Operator by https://github.com/chaddy1004/sobel-operator-pytorch/
class Sobel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = torch.nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = torch.nn.Parameter(G, requires_grad=False)

    def forward(self, tensor):
        x = self.filter(tensor)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


class MeterBase(object):
    def __init__(self, tensor_range="[0, 1]", border=0):
        self.tensor_range = tensor_range
        self.border = border

    def _prepare(
        self,
        pred,
        gt,
        target: Union[type(torch.Tensor), type(np.ndarray)] = torch.Tensor,
    ):
        if target == torch.Tensor and not isinstance(pred, torch.Tensor):
            _pred = torch.from_numpy(pred)
            if _pred.dim() == 3:
                _pred = _pred.permute(2, 0, 1)
                _pred = _pred.unsqueeze(0)
        elif target == np.ndarray and not isinstance(pred, np.ndarray):
            _pred = pred.cpu().numpy()
        else:
            _pred = pred
        if target == torch.Tensor and not isinstance(gt, torch.Tensor):
            _gt = torch.from_numpy(gt)
            if _gt.dim() == 3:
                _gt = _gt.permute(2, 0, 1)
                _gt = _gt.unsqueeze(0)
        elif target == np.ndarray and not isinstance(gt, np.ndarray):
            _gt = gt.cpu().numpy()
        else:
            _gt = gt
        assert _pred.shape == _gt.shape, f"{_pred.shape} {_gt.shape}"

        if self.border != 0:
            h, w = _pred.shape[-2:]
            _pred = _pred[
                ...,
                int(h * self.border) : h - int(h * self.border),
                int(w * self.border) : w - int(w * self.border),
            ]
            _gt = _gt[
                ...,
                int(h * self.border) : h - int(h * self.border),
                int(w * self.border) : w - int(w * self.border),
            ]

        # convert to [0, 1] if necessary
        if self.tensor_range == "[-1, 1]":
            _pred = (_pred + 1) / 2.0
            _gt = (_gt + 1) / 2.0
        if self.tensor_range == "[0, 255]":
            _pred = _pred / 255.0
            _gt = _gt / 255.0

        if isinstance(_pred, torch.Tensor):
            _pred = _pred.clamp(0.0, 1.0)
        else:
            _pred = np.clip(_pred, 0.0, 1.0)

        return _pred, _gt


class MeterPSNR(MeterBase):
    def __init__(
        self,
        package,
        psnr_type="y",
        tensor_range="[0, 1]",
        border=0.0,
        value_min=0.0,
        value_max=1.0,
        verbose=True,
    ):
        super().__init__()
        self.package = package
        self.psnr_type = psnr_type
        self.tensor_range = tensor_range
        self.border = border
        self.total_psnr = 0.0
        self.total_n = 0
        self.name = "PSNR"
        self.verbose = verbose
        self.value_min = value_min
        self.value_max = value_max

    def update(self, pred, gt, meta=None, base_elev=0, elev_log=False):
        _pred, _gt = self._prepare(pred, gt)

        if self.package == "piq":
            _psnr = piq.psnr(
                _gt,
                _pred,
                data_range=1.0,
                convert_to_greyscale=(self.psnr_type == "y"),
                reduction="mean",
            )
        elif self.package == "skimage":
            if self.psnr_type == "y":
                _pred = _pred
                _gt = RGB2YCbCr(_gt)
            _psnr = skimage.metrics.peak_signal_noise_ratio(
                _gt.cpu().numpy(),
                _pred.cpu().numpy(),
                data_range=1.0,
            )
        elif self.package == "local":
            if self.psnr_type == "y":
                _pred = RGB2YCbCr(_pred)
                _gt = RGB2YCbCr(_gt)
            _psnr = psnr(_gt.cpu().numpy(), _pred.cpu().numpy())
        else:
            raise NotImplementedError

        if isinstance(_psnr, torch.Tensor):
            self.total_psnr += _psnr.item()
        else:
            self.total_psnr += _psnr

        self.total_n += 1

    def reset(self):
        self.total_psnr = 0.0
        self.total_n = 0

    def get_score(self):
        eval_result = self.total_psnr / self.total_n
        if self.verbose:
            print(
                f"{self.package} {self.name} {1-self.border}\t"
                f"{'Y' if self.psnr_type=='y' else ''}\t{eval_result:6.4f}"
            )

        return eval_result


class MeterSSIM(MeterBase):
    def __init__(
        self,
        package,
        tensor_range="[0, 1]",
        border=0.0,
        value_min=0.0,
        value_max=1.0,
        verbose=True,
    ):
        super().__init__()
        self.package = package
        self.tensor_range = tensor_range
        self.border = border
        self.total_ssim = 0.0
        self.total_n = 0
        self.name = "SSIM"
        self.value_min = value_min
        self.value_max = value_max
        self.verbose = verbose

    def update(self, pred, gt, meta=None, base_elev=0, elev_log=False):
        _pred, _gt = self._prepare(pred, gt)

        if self.package == "piq":
            _ssim = piq.ssim(
                _gt, _pred, data_range=1.0, reduction="mean", downsample=False
            )
        elif self.package == "skimage":
            _ssim_list = []
            for i in range(_pred.shape[1]):
                _tmp = skimage.metrics.structural_similarity(
                    _gt.cpu().numpy()[:, i, :, :].squeeze(),
                    _pred.cpu().numpy()[:, i, :, :].squeeze(),
                    channel_axis=0,
                    data_range=1.0,
                )
                _ssim_list.append(_tmp)
            _ssim = sum(_ssim_list) / len(_ssim_list)
        elif self.package == "local":
            _ssim = ssim(_gt, _pred, size_average=True)
        else:
            raise NotImplementedError

        if isinstance(_ssim, torch.Tensor):
            self.total_ssim += _ssim.item()
        else:
            self.total_ssim += _ssim

        self.total_n += 1

    def reset(self):
        self.total_ssim = 0.0
        self.total_n = 0

    def get_score(self):
        eval_result = self.total_ssim / self.total_n
        if self.verbose:
            print(f"{self.package} {self.name} {1-self.border}\t\t{eval_result:.6e}")

        return eval_result


class MeterRMSE(MeterBase):
    def __init__(
        self,
        package,
        tensor_range="[0, 1]",
        border=0.0,
        value_min=0.0,
        value_max=1.0,
        verbose=True,
    ):
        super().__init__()
        self.package = package
        self.tensor_range = tensor_range
        self.border = border
        self.value_min = value_min
        self.value_max = value_max
        self.total_rmse = 0.0
        self.total_n = 0
        self.sample_rmse = []
        self.sample_id = []
        self.name = "RMSE"
        self.verbose = verbose

    def update(self, pred, gt, meta=None, base_elev=0, elev_log=False):
        _pred, _gt = self._prepare(pred, gt)
        # batch size is 1
        _subset = [m["subset"] for m in meta][0].split("_")[0]
        _subset = _subset if len(_subset) < 6 else _subset[:7]
        _id = [m["id"] for m in meta][0].split("-")
        _id = "-".join((_id[2], _id[3]))
        _id = "_".join((_subset, _id))

        # _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max) + base_elev
        # _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max) + base_elev
        _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max, elev_log)
        _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max, elev_log)

        if self.package == "local":
            assert (
                _pred.shape == _gt.shape
                and _pred.dim() == _gt.dim() == 4
                and _pred.size(1) == _gt.size(1) == 1
            ), f"{_pred.shape} {_gt.shape}"
            _tmp = _pred - _gt
            _tmp = torch.pow(_tmp, 2)
            _tmp = _tmp.sum() / _tmp.numel()
            _rmse = torch.sqrt(_tmp)

        else:
            raise NotImplementedError

        if isinstance(_rmse, torch.Tensor):
            self.total_rmse += _rmse.item()
        else:
            raise ValueError

        self.sample_rmse.append(_rmse.item())
        self.sample_id.append(_id)
        self.total_n += 1

    def reset(self):
        self.total_rmse = 0.0
        self.total_n = 0
        self.sample_rmse = []
        self.sample_id = []

    def get_score(self):
        eval_result = self.total_rmse / self.total_n
        if self.total_n > 3:
            # get worst n-1 samples name, currently n=3
            sample_worst = {}
            for i in range(3):
                idx = np.argmax(self.sample_rmse)
                sample_worst[self.sample_id[idx]] = self.sample_rmse[idx]
                self.sample_rmse.pop(idx)
                self.sample_id.pop(idx)
            string_worst = ", ".join([f"{k} {v:.2f}" for k, v in sample_worst.items()])
            if self.verbose:
                print(
                    f"{self.package[:3]} {self.name} {1-self.border}\t\t{eval_result:5.4f}, {string_worst}"
                )

        return eval_result


class MeterMedian(MeterBase):
    def __init__(
        self,
        package,
        tensor_range="[0, 1]",
        border=0.0,
        value_min=0.0,
        value_max=1.0,
        verbose=True,
    ):
        super().__init__()
        self.package = package
        self.tensor_range = tensor_range
        self.border = border
        self.value_min = value_min
        self.value_max = value_max
        self.total_median = 0.0
        self.total_n = 0
        self.name = "Median"
        self.verbose = verbose

    def update(self, pred, gt, meta=None, base_elev=0, elev_log=False):
        _pred, _gt = self._prepare(pred, gt)

        # _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max) + base_elev
        # _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max) + base_elev
        _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max, elev_log)
        _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max, elev_log)

        if self.package == "local":
            _median = torch.median(_pred - _gt)
        else:
            raise NotImplementedError

        if isinstance(_median, torch.Tensor):
            self.total_median += _median.item()
        else:
            raise ValueError

        self.total_n += 1

    def reset(self):
        self.total_median = 0.0
        self.total_n = 0

    def get_score(self):
        eval_result = self.total_median / self.total_n
        if self.verbose:
            print(
                f"{self.package[:3]} {self.name} {1-self.border}\t\t{eval_result:5.4f}"
            )

        return eval_result


class MeterNMAD(MeterBase):
    def __init__(
        self,
        package,
        tensor_range="[0, 1]",
        border=0.0,
        value_min=0.0,
        value_max=1.0,
        verbose=True,
    ):
        super().__init__()
        self.package = package
        self.tensor_range = tensor_range
        self.border = border
        self.value_min = value_min
        self.value_max = value_max
        self.total_nmad = 0.0
        self.total_n = 0
        self.name = "NMAD"
        self.verbose = verbose

    def update(self, pred, gt, meta=None, base_elev=0, elev_log=False):
        _pred, _gt = self._prepare(pred, gt)

        # _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max) + base_elev
        # _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max) + base_elev
        _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max, elev_log)
        _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max, elev_log)

        if self.package == "local":
            _dh = _pred - _gt
            _mdh = torch.median(_dh)
            _nmad = 1.4826 * torch.median(torch.abs(_dh - _mdh))
        else:
            raise NotImplementedError

        if isinstance(_nmad, torch.Tensor):
            self.total_nmad += _nmad.item()
        else:
            raise ValueError

        self.total_n += 1

    def reset(self):
        self.total_nmad = 0.0
        self.total_n = 0

    def get_score(self):
        eval_result = self.total_nmad / self.total_n
        if self.verbose:
            print(
                f"{self.package[:3]} {self.name} {1-self.border}\t\t{eval_result:5.4f}"
            )

        return eval_result


class MeterLE95(MeterBase):
    def __init__(
        self,
        package,
        tensor_range="[0, 1]",
        border=0.0,
        value_min=0.0,
        value_max=1.0,
        verbose=True,
    ):
        super().__init__()
        self.package = package
        self.tensor_range = tensor_range
        self.border = border
        self.value_min = value_min
        self.value_max = value_max
        self.total_le95 = 0.0
        self.total_n = 0
        self.name = "LE95"
        self.verbose = verbose

    def update(self, pred, gt, meta=None, base_elev=0, elev_log=False):
        _pred, _gt = self._prepare(pred, gt)

        # _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max) + base_elev
        # _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max) + base_elev
        _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max, elev_log)
        _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max, elev_log)

        if self.package == "local":
            _dh = _pred - _gt
            # https://gist.github.com/sailfish009/28b54c8aa6398148a6358b8f03c0b611
            _k = 1 + round(float(0.95) * (_dh.numel() - 1))
            _le95 = torch.kthvalue(torch.abs(_dh).flatten(), _k).values
        else:
            raise NotImplementedError

        if isinstance(_le95, torch.Tensor):
            self.total_le95 += _le95.item()
        else:
            raise ValueError

        self.total_n += 1

    def reset(self):
        self.total_le95 = 0.0
        self.total_n = 0

    def get_score(self):
        eval_result = self.total_le95 / self.total_n
        if self.verbose:
            print(
                f"{self.package[:3]} {self.name} {1-self.border}\t\t{eval_result:5.4f}"
            )

        return eval_result


# three methods to calculate slope, richdem, kornia, local, respectively
# please note the results of the three methods are different.
class MeterSlope(MeterBase):
    def __init__(
        self,
        package="local",
        tensor_range="[0, 1]",
        border=0,
        value_min=0,
        value_max=1,
        verbose=True,
    ):
        super().__init__()
        self.package = package
        self.tensor_range = tensor_range
        self.border = border
        self.value_range = [value_min, value_max]
        self.total_rmse = 0.0
        self.total_n = 0
        self.name = "Slop"
        self.value_min = value_min
        self.value_max = value_max
        self.verbose = verbose
        self.sobel = Sobel().cuda()

    # https://github.com/r-barnes/richdem/issues/20
    # convert numpy array to rdarray
    def np2rdarray(self, in_array, no_data=None, projection=None, geotransform=None):
        assert (
            len(in_array.shape) == 2
        ), f"Input array must be 2D but get {in_array.shape}"
        gt1, gt5 = in_array.shape
        if no_data is None:
            no_data = np.nan
        if projection is None:
            # projection = CRS.from_epsg(2154)
            projection = "EPSG:2154"
        if geotransform is None:
            geotransform = (0.0, gt1, 0.0, 0.0, 0.0, gt5)

        out_array = rd.rdarray(in_array, no_data=no_data)
        out_array.projection = projection
        out_array.geotransform = geotransform
        return out_array

    @hide_warnings
    def get_slope(self, dem, attrib="slope_riserun"):
        return rd.TerrainAttribute(dem, attrib=attrib)

    def update(self, pred, gt, meta=None, base_elev=0, elev_log=False):
        if self.package.lower() == "richdem":
            _pred, _gt = self._prepare(pred, gt, target=np.ndarray)
            base_elev = base_elev.cpu().numpy()
        else:
            _pred, _gt = self._prepare(pred, gt)

        # _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max) + base_elev
        # _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max) + base_elev
        _pred = ToDEM.descale_data(_pred, self.value_min, self.value_max, elev_log)
        _gt = ToDEM.descale_data(_gt, self.value_min, self.value_max, elev_log)

        if self.package.lower() == "richdem":
            _pred_rd = self.np2rdarray(_pred.squeeze())
            _gt_rd = self.np2rdarray(_gt.squeeze())
            _slop_pred = self.get_slope(_pred_rd, attrib="slope_riserun")
            _slop_gt = self.get_slope(_gt_rd, attrib="slope_riserun")
            _slop_rmse = np.linalg.norm(_slop_pred - _slop_gt) / np.sqrt(
                len(_slop_pred.flatten())
            )
        elif self.package.lower() == "kornia":
            _pred_slope = spatial_gradient(_pred)
            _gt_slope = spatial_gradient(_gt)
            _slop_rmse = torch.sqrt(torch.mean(((_pred_slope - _gt_slope) ** 2)))
        elif self.package.lower() == "local":
            _pred_slope = self.sobel(_pred)
            _gt_slope = self.sobel(_gt)
            _slop_rmse = torch.sqrt(torch.mean(((_pred_slope - _gt_slope) ** 2)))
        else:
            raise NotImplementedError

        if isinstance(_slop_rmse, torch.Tensor):
            self.total_rmse += _slop_rmse.item()
        else:
            self.total_rmse += _slop_rmse

        self.total_n += 1

    def reset(self):
        self.total_rmse = 0.0
        self.total_n = 0

    def get_score(self):
        eval_result = self.total_rmse / self.total_n
        if self.verbose:
            print(
                f"{self.package[:3]} {self.name} {1-self.border}\t\t{eval_result:5.4f}"
            )

        return eval_result
