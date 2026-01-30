import numpy.random as random
from math import ceil, log
import numpy as np
import torch
import torchvision
from affine import Affine


class RandomFlipRotate90(object):
    """Flip and/or rotate the image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):
        if random.random() < 0.5:
            angle = random.choice([1, 2, 3])
            do_lr = random.choice([True, False])
            do_ud = random.choice([True, False])
            for elem in sample.keys():
                if "meta" in elem:
                    sample["meta"]["augmentation"] = {
                        "rot90": angle,
                        "flip_lr": do_lr,
                        "flip_ud": do_ud,
                    }
                else:
                    tmp = sample[elem]
                    tmp = np.rot90(tmp, angle)
                    tmp = np.fliplr(tmp) if do_lr else tmp
                    tmp = np.flipud(tmp) if do_ud else tmp
                    sample[elem] = tmp
        return sample

    def __str__(self):
        return "RandomFlipRotate90"


class RandomCrop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, crop_size=128, scale=None):
        self.crop_size = crop_size
        self.scale = scale

    def __call__(self, sample):
        if "image" in sample:
            h, w, _ = sample["image"].shape
        elif "lr_img" in sample:
            h, w, _ = sample["lr_img"].shape
        elif "lr_dem" in sample:
            h, w, _ = sample["lr_dem"].shape
        else:
            raise ValueError(f"the sample do not contains image: {sample.keys()}")

        if self.crop_size > h or self.crop_size > w or (self.crop_size == h == w):
            return sample
        else:
            _h = random.randint(0, h - self.crop_size - 1)
            _w = random.randint(0, w - self.crop_size - 1)

        for elem in sample:
            if elem == "hr_img" and self.scale is not None:
                assert self.scale % 2 == 0, "scale must be even"
                tmp = sample[elem]
                tmp = tmp[
                    _h * self.scale : (_h + self.crop_size) * self.scale,
                    _w * self.scale : (_w + self.crop_size) * self.scale,
                    :,
                ]
                sample[elem] = tmp
            elif "meta" in elem:
                sample[elem]["bbox"] = (
                    _h,
                    _w,
                    _h + self.crop_size,
                    _w + self.crop_size,
                )
            else:
                tmp = sample[elem]
                tmp = tmp[_h : _h + self.crop_size, _w : _w + self.crop_size, :]
                sample[elem] = tmp

        return sample

    def __str__(self):
        return "RandomCrop"


class TileCrop(object):
    """Crop the image in a sample in a tile cover mode."""

    def __init__(self, crop_size=128, scale=None, n_tile=None):
        self.crop_size = crop_size
        self.n_tile = n_tile
        self.scale = scale
        self._row = 0
        self._col = 0
        self._stride = 0

    def __call__(self, sample):
        if "image" in sample:
            h, w, _ = sample["image"].shape
        elif "lr_img" in sample:
            h, w, _ = sample["lr_img"].shape
        elif "lr_dem" in sample:
            h, w, _ = sample["lr_dem"].shape
        else:
            raise ValueError(f"Input invalid: {sample.keys()}")

        if self.crop_size > h or self.crop_size > w or (self.crop_size == h == w):
            return sample

        if self._row == 0 and self._col == 0:
            # calculate stride
            self._stride, self.n_tile = self.get_tile(w, self.crop_size, self.n_tile)

        for elem in sample:
            if elem == "hr_img" and self.scale is not None:
                assert self.scale % 2 == 0, "scale must be even"
                tmp = sample[elem]
                tmp = tmp[
                    self._stride
                    * self._row
                    * self.scale : (self._stride * self._row + self.crop_size),
                    self._stride
                    * self._col
                    * self.scale : (self._stride * self._col + self.crop_size)
                    * self.scale,
                    :,
                ]
                sample[elem] = tmp
            elif "meta" in elem:
                sample[elem]["bbox"] = (
                    self._stride * self._col,
                    self._stride * self._row,
                    self._stride * self._col + self.crop_size,
                    self._stride * self._row + self.crop_size,
                )
                _profile = sample[elem]["profile"]
                _res = _profile["transform"][0]
                _xy = _profile["transform"] * (
                    self._stride * self._col,
                    self._stride * self._row,
                )
                _profile["transform"] = Affine(_res, 0.0, _xy[0], 0.0, -_res, _xy[1])
                _profile["width"] = self.crop_size
                _profile["height"] = self.crop_size
                sample[elem]["profile"] = _profile
            else:
                tmp = sample[elem]
                tmp = tmp[
                    self._stride * self._row : self._stride * self._row
                    + self.crop_size,
                    self._stride * self._col : self._stride * self._col
                    + self.crop_size,
                    :,
                ]
                sample[elem] = tmp

        self._col += 1
        if self._col == self.n_tile**0.5:
            self._row += 1
            if self._row == self.n_tile**0.5 and self._col == self.n_tile**0.5:
                self._row = 0
                self._stride = 0
                self.n_tile = None
            self._col = 0
        # print(f"TileCrop 1: {self._row} {self._col} {self.n_tile} {self._stride}")

        return sample

    @staticmethod
    def get_tile(w, k, n=None):
        """
        Calculate the tile parameters.
        only support square image and tile,
        n must be a square number
        no padding

        :param w: input image size (h or w)
        :param n: number of tiles
        :param k: tile size (h or w)
        :return: row, col, stride
        """
        if n is None:
            n_x = (w - w % k) / k + 1  # number of tiles in row or column
        else:
            n_x = ceil(n**0.5)
        assert (
            n_x % 1 == 0
        ), "cannot divide the image into n_tile tiles, check the input."
        stride = (w - k) / (n_x - 1)
        assert (
            stride % 1 == 0
        ), "no padding for cropping to tile evenly, check the input."
        return int(stride), int(n_x**2)

    def __str__(self):
        return "TileCrop"


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(
        self, normalize_list=None, mask_channel=None, relative=False, **kwargs
    ):
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize_list = normalize_list if normalize_list else []
        self.image_range = kwargs.get("image_range", None) if kwargs else None
        self.label_range = kwargs.get("label_range", None) if kwargs else None
        self.elev_min = kwargs.get("min", None) if kwargs else None
        self.elev_max = kwargs.get("max", None) if kwargs else None
        self.elev_log = kwargs.get("log", False) if kwargs else False
        self.relative = relative
        self.scale_mask = kwargs.get("scale_mask", False) if kwargs else False
        self.mask_channel = mask_channel if mask_channel else [*range(15)]

    def __call__(self, sample):
        base_elev = sample["meta"]["base"] if self.relative else 0.0
        sample_id = sample["meta"]["id"]

        for elem in sample:
            if "meta" in elem:
                continue
            tmp = sample[elem]
            if "img" in elem or "image" in elem:
                # [0, 1]
                sample[elem] = self.to_tensor(tmp.astype(np.uint8))
                if self.label_range is not None and elem == "hr_img":
                    if self.label_range == "[-1, 1]":
                        sample[elem] = 2.0 * sample[elem] - 1.0
                    elif self.label_range == "[0, 255]":
                        sample[elem] = sample[elem] / 255.0
                if self.image_range is not None and elem in {"lr_img", "image"}:
                    if self.image_range == "[-1, 1]":
                        sample[elem] = 2.0 * sample[elem] - 1.0
                    elif self.image_range == "[0, 255]":
                        sample[elem] = sample[elem] / 255.0
            else:
                tmp = tmp.transpose((2, 0, 1)).astype(np.float32)
                if "dem" in elem and elem not in self.normalize_list:
                    assert (
                        self.elev_min is not None and self.elev_max is not None
                    ), f"{elem} {self.elev_min} {self.elev_max}"
                    _min = tmp.min()  # min elevation before scale
                    _max = tmp.max()  # max elevation before scale
                    tmp = self.scale_data(
                        tmp,
                        self.elev_min,
                        self.elev_max,
                        self.elev_log,
                        base_elev=base_elev,
                    )
                    assert (
                        tmp.min() >= 0 and tmp.max() <= 1
                    ), f"{sample_id} {elem}, {tmp.min()} {tmp.max()} {_min} {_max}, {base_elev} {self.elev_min} {self.elev_max}"
                    if self.label_range is not None and elem == "hr_dem":
                        if self.label_range == "[-1, 1]":
                            tmp = tmp * 2 - 1
                    if self.image_range is not None and elem in {"lr_dem"}:
                        if self.image_range == "[-1, 1]":
                            tmp = tmp * 2 - 1
                if "mask" in elem and self.scale_mask:
                    # each channel has unique value and max value smaller than 1
                    for i in range(tmp.shape[0]):
                        tmp[i] = tmp[i] * (i + 1) / (len(self.mask_channel) + 1)
                if "canopy" in elem:
                    tmp = tmp / 68  # 68 is the max canopy height
                    # scale canopy height to the same range as DEM
                    # tmp = self.scale_data(
                    #     tmp,
                    #     self.elev_min,
                    #     self.elev_max,
                    #     elev_log=self.elev_log,
                    #     base_elev=0,
                    # )

                assert (
                    tmp.min() >= 0 and tmp.max() <= 1
                ), f"{sample_id} {elem}, {_min} {_max}"

                sample[elem] = torch.from_numpy(tmp).type(torch.float32).contiguous()

        # output: [c:h:w] [r:g:b] [0:1] by default
        return sample

    def __str__(self):
        return "ToTensor"

    @staticmethod
    def scale_data(data, elev_min, elev_max, elev_log=False, base_elev=0.0):
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        if base_elev != 0:  # relative elevation
            data = data - base_elev
        if elev_log:  # log minmax normalization
            if isinstance(data, np.ndarray):
                assert (
                    np.min(data) - elev_min >= 1
                ), f"elev_min must smaller than (data - 1) for [0, 1] range: {np.min(data)} {elev_min}"
                data = np.log(data - elev_min) / np.log(elev_max - elev_min) + 1e-8
            elif isinstance(data, torch.Tensor):
                assert (
                    torch.min(data) - elev_min >= 1
                ), f"elev_min must smaller than (data - 1) for [0, 1] range: {torch.min(data)} {elev_min}"
                data = (
                    torch.log(data - elev_min) / torch.log(elev_max - elev_min) + 1e-8
                )
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        else:  # minmax normalization
            data = (data - elev_min) / (elev_max - elev_min)
        return data


# do not use Normalization due to negative effect on training for DEM
class Normalize(object):
    def __init__(self, normalize_list=None, resolution=None):
        # DFC2022+ stats, the distribution of the dataset is not normal, so normalization is not recommended
        # r8
        # RGB Mean:         [104.5478121  113.53916278  91.06393941] 	STD [48.61966393 36.84840044 33.2264289 ] 	Min 1 	Max 255
        # HR DEM Mean:      200.50319 	STD 386.5053 	Min -52.41804 	Max 3018.4194
        # Log HR DEM Mean:  5.0787544 	STD 0.83917964 	Min 2.1496623 	Max 8.032496
        # LR DEM Mean:      201.49762 	STD 386.18207 	Min -59.902428 	Max 3024.011
        # Log LR DEM Mean:  5.0840178 	STD 0.83918 	Min 0.093100764 Max 8.03431

        # r3
        # RGB Mean:         [104.55297366 113.54333935  91.0669583 ] 	STD [50.76874938 38.8785096  34.9372223 ] 	Min 1 	Max 255
        # HR DEM Mean:      200.49414 	STD 386.50452 	Min -52.48448 	Max 3020.159
        # Log HR DEM Mean:  5.078724 	STD 0.8391652 	Min 2.14189 	Max 8.033061
        # LR DEM Mean:      201.48833 	STD 386.1985 	Min -58.85854 	Max 3024.1736
        # Log LR DEM Mean:  5.083961 	STD 0.83916867 	Min 0.761488 	Max 8.034363

        # for div2k
        _mean = {
            "lr_img": [114.45135577, 111.47021502, 103.02948559],
            "hr_img": [114.45135577, 111.47021502, 103.02948559],
        }
        _std = {
            "lr_img": [-10.67384323, -5.43743541, 0.44979045],
            "hr_img": [-10.67384323, -5.43743541, 0.44979045],
        }
        # for DFC30
        _mean_8m = {
            "image": [104.5478121, 113.53916278, 91.06393941],
            "lr_dem": [201.49762],
            "hr_dem": [200.50319],
        }
        _std_8m = {
            "image": [48.61966393, 36.84840044, 33.2264289],
            "lr_dem": [386.18207],
            "hr_dem": [386.5053],
        }
        _mean_3m = {
            "image": [104.55297366, 113.54333935, 91.0669583],
            "lr_dem": [201.48833],
            "hr_dem": [200.49414],
        }
        _std_3m = {
            "image": [50.76874938, 38.8785096, 34.9372223],
            "lr_dem": [386.1985],
            "hr_dem": [386.50452],
        }
        self.normalize_list = normalize_list
        self.resolution = resolution
        if self.resolution == 8:
            self.mean = _mean_8m
            self.std = _std_8m
        elif self.resolution == 3:
            self.mean = _mean_3m
            self.std = _std_3m
        else:
            self.mean = _mean
            self.std = _std

    def __call__(self, sample):
        for elem in sample.keys():
            if elem not in self.normalize_list:
                continue
            tmp = sample[elem]
            _mean = self.mean[elem]
            _std = self.std[elem]
            if isinstance(tmp, torch.Tensor):
                normalizer = torchvision.transforms.Normalize(tuple(_mean), tuple(_std))
                sample[elem] = normalizer(tmp)
            else:
                sample[elem] = self.normalizer(tmp, _mean, _std)
        return sample

    def __str__(self):
        return f"Normalize(\nmean{self.mean}, \nstd{self.std})"

    @staticmethod
    def normalizer(arr, mean, std):
        mean = np.array(mean).astype(np.float32)
        std = np.array(std).astype(np.float32)
        arr = (arr - mean) / std
        return arr


class ToImage(object):
    """
    Convert Tensors [0-1] back to image nparray.
    """

    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            data = data.to("cpu").numpy()
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        # data = (255.0 * (data - data.min()) / (data.max() - data.min() + 1e-10)).astype(int)
        assert data.min() >= 0 and data.max() <= 1, f"{data.min()} {data.max()}"
        data = (255.0 * data).astype(int)
        return data

    def __str__(self):
        return "ToImage"


class ToDEM(object):
    """
    Convert Tensors [0-1] back to DEM nparray.
    """

    def __init__(self, elev_min, elev_max, elev_log=False):
        self.elev_min = elev_min
        self.elev_max = elev_max
        self.elev_log = elev_log

    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            data = data.to("cpu").numpy()
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        assert data.min() >= 0 and data.max() <= 1, f"{data.min()} {data.max()}"
        data = self.descale_data(data, self.elev_min, self.elev_max, self.elev_log)
        return data

    def __str__(self):
        return "ToDEM"

    @staticmethod
    def descale_data(data, elev_min, elev_max, elev_log=False):
        if elev_log:
            if isinstance(data, np.ndarray):
                data = (
                    np.exp(data.astype(np.float32) * np.log(elev_max - elev_min))
                    + elev_min
                )
            elif isinstance(data, torch.Tensor):
                data = torch.exp(data * log(elev_max - elev_min)) + elev_min
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        else:
            if isinstance(data, np.ndarray):
                data = data.astype(np.float32)
            data = data * (elev_max - elev_min) + elev_min
        return data


class RGB2YCbCr(object):
    def __init__(self, y_channel_only=False):
        self.y_channel_only = y_channel_only

    def __call__(self, sample):
        for elem in sample.keys():
            tmp = sample[elem]
            if "img" in elem or "image" in elem:
                sample[elem] = self.rgb2ycbcr(tmp, self.y_channel_only)
            else:
                continue
        return sample

    def __str__(self):
        if self.y_channel_only:
            return "RGB2YCbCr channel Y only"
        else:
            return "RGB2YCbCr channel Y Cb CR"

    # https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/color_util.py
    @staticmethod
    def rgb2ycbcr(img: np.ndarray, y_only: bool) -> np.ndarray:
        """Convert a RGB image to YCbCr image.

        This function produces the same results as Matlab's `rgb2ycbcr` function.
        It implements the ITU-R BT.601 conversion for standard-definition
        television. See more details in
        https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

        It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
        In OpenCV, it implements a JPEG conversion. See more details in
        https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

        Args:
            img (ndarray): The input image. It accepts:
                1. np.uint8 type with range [0, 255];
                2. np.float32 type with range [0, 1].
            y_only (bool): Whether to only return Y channel. Default: False.

        Returns:
            ndarray: The converted YCbCr image. The output image has the same type
                and range as input image.
        """
        assert isinstance(img, np.ndarray), (
            f"img type is not np.ndarray. Got {type(img)}" f"Expect np.ndarray."
        )
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        if y_only:
            out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
        else:
            out_img = np.matmul(
                img,
                [
                    [65.481, -37.797, 112.0],
                    [128.553, -74.203, -93.786],
                    [24.966, 112.0, -18.214],
                ],
            ) + [16, 128, 128]
        return out_img

    @staticmethod
    def ycbcr2rgb(img: np.ndarray) -> np.ndarray:
        """Convert a YCbCr image to RGB image.

        This function produces the same results as Matlab's ycbcr2rgb function.
        It implements the ITU-R BT.601 conversion for standard-definition
        television. See more details in
        https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

        It differs from a similar function in cv2.cvtColor: `YCrCb <-> RGB`.
        In OpenCV, it implements a JPEG conversion. See more details in
        https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

        Args:
            img (ndarray): The input image. It accepts:
                1. np.uint8 type with range [0, 255];
                2. np.float32 type with range [0, 1].

        Returns:
            ndarray: The converted RGB image. The output image has the same type
                and range as input image.
        """
        assert isinstance(img, np.ndarray), (
            f"img type is not np.ndarray. Got {type(img)}" f"Expect np.ndarray."
        )
        if img.dtype == np.float32:
            img = (img * 255.0).astype(np.uint8)

        out_img = np.matmul(
            img,
            [
                [0.00456621, 0.00456621, 0.00456621],
                [0, -0.00153632, 0.00791071],
                [0.00625893, -0.00318811, 0],
            ],
        ) * 255.0 + [
            -222.921,
            135.576,
            -276.836,
        ]  # noqa: E126

        return out_img
