import cv2
from pathlib import Path
import tifffile  # faster than rasterio for reading tiff files
import rasterio
import numpy as np
import torch
import torch.utils.data as data
from natsort import natsorted


class DFC30(data.Dataset):
    """
    DFC30 dataset
    """

    def __init__(
        self,
        split="valid",  # options: "train", "tra", "valid", "val", "test", "trainval", "all"
        transform=None,
        **kwargs,
    ):
        super(DFC30, self).__init__()
        self.DFC30_bounds = [
            100000,
            6200000,
            1100000,
            7120000,
        ]  # minx, miny, maxx, maxy
        self.p = kwargs
        self.transform = transform
        self.path = self.p.get("dataset_path", r"../datasets/DFC30_8m")
        self.resolution = self.p.get("resolution", 8)
        self.input_data = self.p.get("input_data")
        self.mask_channel = (
            self.p.get("mask_channel")
            if self.p.get("mask_channel") is not None
            else [*range(15)]
        )
        self.coord_mode = self.p.get("coord_mode", None)
        self.patches_per_image = (
            self.p.get("patches_per_image")
            if self.p.get("patches_per_image") is not None
            else 1
        )
        if isinstance(split, str):
            self.split = [split]
        else:
            self.split = natsorted(split)
        assert all(
            [
                x in ["train", "tra", "valid", "val", "test", "trainval", "all"]
                for x in self.split
            ]
        ), f"{self.split} contains invalid split name(s)"
        self.relative = self.p.get("relative", False)

        self.id = []
        self.subset = []
        self.lr_dem = []
        self.image = []
        self.hr_dem = []
        if self.input_data.get("mask"):
            self.mask = []
        if self.input_data.get("canopy"):
            self.canopy = []

        # for loading data
        self.last_image_index = None
        self.last_lr_dem_index = None
        self.last_hr_dem_index = None
        self.last_mask_index = None
        self.last_canopy_index = None
        self.last_image = None
        self.last_lr_dem = None
        self.last_lr_dem_ds = None
        self.last_hr_dem = None
        self.last_mask = None
        self.last_canopy = None
        self.last_coord_size = [0, 0]
        self.last_coord = None

        _data_dir = [d for d in Path(self.path).glob("*") if d.is_dir()]

        if self.p.get("verbose"):
            print(f"Initializing dataset for DFC30 {self.resolution}m {self.split} set")

        for sp in self.split:
            sp_set = None
            if sp in ["train", "tra"]:
                sp_set = [d for d in _data_dir if d.name in self.p.get("train_set", [])]
            if sp in ["valid", "val", "test"]:
                sp_set = [d for d in _data_dir if d.name in self.p.get("valid_set", [])]
            if sp in ["trainval", "all"]:
                sp_set = [
                    d
                    for d in _data_dir
                    if d.name
                    in (self.p.get("train_set", []) + self.p.get("valid_set", []))
                ]
            assert sp_set is not None and len(sp_set) > 0, f"Invalid split {sp}"

            _size = 0
            for _, dataset in enumerate(sp_set):
                _files = natsorted(
                    [f for f in dataset.rglob("*.tif")]
                )  # sort to avoid different id/file order
                # Low resolution DEM
                if self.input_data.get("COP30") == 1:
                    _lr_dem = [f for f in _files if "COP30" == f.parent.name]
                elif self.input_data.get("FABDEM") == 1:
                    _lr_dem = [f for f in _files if "FABDEM" == f.parent.name]
                else:
                    raise ValueError("Invalid input_data configuration")
                self.lr_dem.extend(_lr_dem)
                # Image
                if self.input_data.get("image"):
                    _image = [f for f in _files if "BDORTHO" == f.parent.name]
                    self.image.extend(_image)
                # Ground Truth High Resolution DEM
                _hr_dem = [f for f in _files if "RGEALTI" == f.parent.name]
                self.hr_dem.extend(_hr_dem)
                # Land Use Mask
                if self.input_data.get("mask"):
                    # _mask = [f for f in _files if "UrbanAtlas" == f.parent.name]  # previous version
                    _mask = [f for f in _files if "UA2012" == f.parent.name]
                    self.mask.extend(_mask)
                # IDs
                if self.input_data.get("COP30") == 1:
                    self.id.extend([f.stem[: -len("_COP30")] for f in _lr_dem])
                elif self.input_data.get("FABDEM") == 1:
                    self.id.extend([f.stem[: -len("_FABDEM")] for f in _lr_dem])
                else:
                    raise ValueError("Invalid input_data configuration")
                # Canopy
                if self.input_data.get("canopy"):
                    _canopy = [f for f in _files if "CHM" == f.parent.name]
                    self.canopy.extend(_canopy)

                self.subset.extend([f.parent.parent.name for f in _lr_dem])
                _size += len(_lr_dem)
                assert len(_lr_dem) == self._check_size(dataset.name)
            sp_set = [d.name for d in sp_set]  # path to name
            assert _size == self._check_size(sp_set)

        self.lr_dem = [f.as_posix() for f in self.lr_dem]
        self.hr_dem = [f.as_posix() for f in self.hr_dem]
        if self.input_data.get("image"):
            self.image = [f.as_posix() for f in self.image]
        if self.input_data.get("mask"):
            self.mask = [f.as_posix() for f in self.mask]
        if self.input_data.get("canopy"):
            self.canopy = [f.as_posix() for f in self.canopy]

        assert len(self.id) == len(self.lr_dem)
        if self.input_data.get("image"):
            assert len(self.id) == len(self.image)
        assert len(self.id) == len(self.hr_dem)
        if self.input_data.get("mask"):
            assert len(self.id) == len(self.mask)
        if self.input_data.get("canopy"):
            assert len(self.id) == len(self.canopy)
        assert len(self.id) == len(self.subset)

        if self.patches_per_image > 1:
            self.id = [
                x + f"_{i}" for x in self.id for i in range(self.patches_per_image)
            ]
            self.subset = [
                x for x in self.subset for _ in range(self.patches_per_image)
            ]
            self.lr_dem = [
                x for x in self.lr_dem for _ in range(self.patches_per_image)
            ]
            if self.input_data.get("image"):
                self.image = [
                    x for x in self.image for _ in range(self.patches_per_image)
                ]
            self.hr_dem = [
                x for x in self.hr_dem for _ in range(self.patches_per_image)
            ]
            if self.input_data.get("mask"):
                self.mask = [
                    x for x in self.mask for _ in range(self.patches_per_image)
                ]
            if self.input_data.get("canopy"):
                self.canopy = [
                    x for x in self.canopy for _ in range(self.patches_per_image)
                ]

        # Display stats
        if self.p.get("verbose"):
            print(f"DFC30 {self.split} set sample number: {len(self.id)}")

    def __getitem__(self, index):
        sample = {}
        num_channels = 0

        _lr_dem, _lr_dem_ds = self._load_lr_dem(index)
        if self.relative:
            _lr_dem_min = np.min(_lr_dem)
        sample["lr_dem"] = _lr_dem
        num_channels += _lr_dem.shape[2]
        assert _lr_dem.dtype == np.float32

        if self.input_data.get("image"):
            _image = self._load_img(index)
            sample["image"] = _image
            num_channels += _image.shape[2]
            assert _image.shape[2] == self.input_data.image
            assert _image.dtype == np.uint8

        _hr_dem = self._load_hr_dem(index)
        sample["hr_dem"] = _hr_dem
        num_channels += _hr_dem.shape[2]
        assert _hr_dem.shape[2] == 1
        assert _hr_dem.dtype == np.float32

        if self.input_data.get("coord"):
            _coord = self._gen_coord(_lr_dem, _lr_dem_ds, self.coord_mode)
            sample["coord"] = _coord.astype(np.float32)
            num_channels += _coord.shape[2]

        if self.input_data.get("mask"):
            _mask = self._load_mask(index)
            if len(self.mask_channel) > 0:
                _mask = _mask[:, :, self.mask_channel]
            sample["mask"] = _mask
            num_channels += _mask.shape[2]
            assert _mask.shape[2] == self.input_data.mask == len(self.mask_channel)
            assert _mask.dtype == np.uint8

        if self.input_data.get("canopy"):
            _canopy = self._load_canopy(index)
            sample["canopy"] = _canopy
            num_channels += _canopy.shape[2]
            assert _canopy.shape[2] == 1
            assert _canopy.dtype == np.uint8

        sample["meta"] = {
            "id": str(self.id[index]),
            "subset": str(self.subset[index]),
            "shape": (_lr_dem.shape[0], _lr_dem.shape[1], num_channels),
            "augmentation": {"rot90": 0, "flip_lr": False, "flip_ud": False},
            "bbox": (
                0,
                0,
                _lr_dem.shape[0],
                _lr_dem.shape[1],
            ),  # minx, miny, maxx, maxy
            "base": _lr_dem_min if self.relative else 0,
            "profile": _lr_dem_ds.profile,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.id)

    def _load_lr_dem(self, index):
        if self.last_lr_dem_index != self.lr_dem[index]:
            self.last_lr_dem_index = self.lr_dem[index]
            self.last_lr_dem_ds = rasterio.open(self.lr_dem[index])
            self.last_lr_dem = np.transpose(self.last_lr_dem_ds.read(), (1, 2, 0))
        return self.last_lr_dem, self.last_lr_dem_ds

    def _load_img(self, index):
        if self.last_image_index != self.image[index]:
            self.last_image_index = self.image[index]
            _im = cv2.imread(self.image[index], cv2.IMREAD_UNCHANGED)
            self.last_image = cv2.cvtColor(_im, cv2.COLOR_BGR2RGB)
        return self.last_image

    def _load_hr_dem(self, index):
        if self.last_hr_dem_index != self.hr_dem[index]:
            self.last_hr_dem_index = self.hr_dem[index]
            self.last_hr_dem = np.expand_dims(
                cv2.imread(self.hr_dem[index], cv2.IMREAD_UNCHANGED), axis=2
            )
        return self.last_hr_dem

    def _load_canopy(self, index):
        if self.last_canopy_index != self.canopy[index]:
            self.last_canopy_index = self.canopy[index]
            self.last_canopy = np.expand_dims(
                cv2.imread(self.canopy[index], cv2.IMREAD_UNCHANGED), axis=2
            )
        return self.last_canopy

    def _gen_coord(self, dem, ds, coord_mode=None):
        if coord_mode.lower() == "local":
            w, h = dem.shape[0], dem.shape[1]
            if self.last_coord_size[0] != w or self.last_coord_size[1] != h:
                self.last_coord_size[0] = w
                self.last_coord_size[1] = h
                xx_channel, yy_channel = np.mgrid[0:w, 0:h]
                xx_channel = xx_channel.astype(np.float32) / (w - 1)
                yy_channel = yy_channel.astype(np.float32) / (h - 1)
                self.last_coord = np.concatenate(
                    (
                        np.expand_dims(xx_channel, axis=2),
                        np.expand_dims(yy_channel, axis=2),
                    ),
                    axis=2,
                )
        elif coord_mode.lower() == "global":
            w, h = ds.width, ds.height
            if self.last_coord_size[0] != w or self.last_coord_size[1] != h:
                self.last_coord_size[0] = w
                self.last_coord_size[1] = h
                xx_channel, yy_channel = ds.xy(np.arange(w), np.arange(h))
                xx_channel = sorted(xx_channel)
                yy_channel = sorted(yy_channel)
                xx_channel, yy_channel = np.meshgrid(xx_channel, yy_channel)
                xx_channel = (
                    xx_channel.astype(np.float32) - self.DFC30_bounds[0]
                ) / self.DFC30_bounds[2]
                assert (
                    (0 < xx_channel) & (xx_channel < 1)
                ).all(), "Invalid x coordinate"
                yy_channel = (
                    yy_channel.astype(np.float32) - self.DFC30_bounds[1]
                ) / self.DFC30_bounds[3]
                assert (
                    (0 < yy_channel) & (yy_channel < 1)
                ).all(), "Invalid y coordinate"
                self.last_coord = np.concatenate(
                    (
                        np.expand_dims(xx_channel, axis=2),
                        np.expand_dims(yy_channel, axis=2),
                    ),
                    axis=2,
                )

        return self.last_coord

    def _load_mask(self, index):
        # cv2 only support 1, 3, 4 channels, use tifffile instead
        # return cv2.imread(self.mask[index], cv2.IMREAD_UNCHANGED)
        if self.last_mask_index != self.mask[index]:
            self.last_mask_index = self.mask[index]
            self.last_mask = tifffile.imread(self.mask[index])
        return self.last_mask

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for PyTorch DataLoader.
        """
        new_batch = {}
        new_batch.update({"lr_dem": torch.stack([b["lr_dem"] for b in batch])})
        if all(["image" in b for b in batch]):
            new_batch.update({"image": torch.stack([b["image"] for b in batch])})
        if all(["mask" in b for b in batch]):
            new_batch.update({"mask": torch.stack([b["mask"] for b in batch])})
        if all(["canopy" in b for b in batch]):
            new_batch.update({"canopy": torch.stack([b["canopy"] for b in batch])})
        if all(["coord" in b for b in batch]):
            new_batch.update({"coord": torch.stack([b["coord"] for b in batch])})
        new_batch.update({"hr_dem": torch.stack([b["hr_dem"] for b in batch])})
        new_batch.update({"meta": [b["meta"] for b in batch]})
        return new_batch

    @staticmethod
    def _check_size(dataset):
        ref_size = {
            "Angers": 246,
            "Brest": 172,
            "Caen": 251,
            "Calais_Dunkerque": 256,
            "Cherbourg": 113,
            "Clermont-Ferrand": 300,
            "LeMans": 214,
            "Lille_Arras_Lens_Douai_Henin": 407,
            "Lorient": 120,
            "Marseille_Martigues": 309,
            "Nantes_Saint-Nazaire": 433,
            "Nice": 333,
            "Quimper": 154,
            "Rennes": 391,
            "Saint-Brieuc": 136,
            "Vannes": 146,
        }  # 3981 in total
        dataset = [dataset] if isinstance(dataset, str) else dataset
        return sum([ref_size[d] for d in dataset])

    def __str__(self):
        return f"DFC30 dataset (split={self.split}, resolution={self.resolution})"
