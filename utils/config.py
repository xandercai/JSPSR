import math
from easydict import EasyDict as edict
from pathlib import Path
import yaml
from data.data_utils import TileCrop


def create_config(config_file):
    # Read the files
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)

    # Copy all the arguments
    cfg = edict()
    for k, v in config.items():
        cfg[k] = v

    cfg.work_root = r"./" if cfg.get("work_root") is None else cfg.get("work_root")
    cfg.data_root = (
        r"../datasets" if cfg.get("data_root") is None else cfg.get("data_root")
    )

    if "dfc" in cfg.dataset.lower():
        if cfg.resolution == 8:
            cfg.patch_size = (
                128 if cfg.get("patch_size") is None else cfg.get("patch_size")
            )
            cfg.dataset_path = (Path(cfg.data_root) / "DFC30_8m").as_posix()
            cfg.patches_per_image = 1

        if cfg.resolution == 3:
            cfg.patch_size = (
                128 if cfg.get("patch_size") is None else cfg.get("patch_size")
            )
            cfg.dataset_path = (Path(cfg.data_root) / "DFC30_3m").as_posix()

        if cfg.get("crop_mode") is None:
            cfg.crop_mode = "tile"

        if (
            cfg.resolution == 3
            and cfg.crop_mode.lower() == "tile"
            and cfg.get("patches_per_image") is None
        ):
            _, n_tile = TileCrop.get_tile(334, cfg.patch_size)
            cfg.patches_per_image = n_tile

        cfg.input_data = {} if cfg.get("input_data") is None else cfg.get("input_data")
        # to be compatible with previous version config.yml
        cfg.input_data.lr_dem = 1
        if cfg.input_data.get("COP30") is None and cfg.input_data.get("FABDEM") is None:
            cfg.input_data.COP30 = 1
        assert (
            cfg.input_data.get("COP30") is not None
            or cfg.input_data.get("FABDEM") is not None
        ), "Either cop30 or fabdem have to be configured as lr_dem"
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} not implemented yet.")

    if (
        cfg.tensor_kwargs
        and cfg.tensor_kwargs.get("min") is not None
        and cfg.tensor_kwargs.get("max") is not None
    ):
        if cfg.tensor_kwargs.get("log") is True:
            assert (
                cfg.tensor_kwargs.max - cfg.tensor_kwargs.min > 1
            ), f"If log minmax normalisation on, max - min should larger than 1"
        else:
            cfg.tensor_kwargs.log = False

        for key, kwargs in cfg.metric.items():
            if kwargs.get("border") is None:
                if cfg.get("val_border") is not None:
                    cfg["metric"][key]["border"] = cfg.val_border
                else:
                    cfg["metric"][key]["border"] = 0
            if kwargs.get("tensor_range") is None:
                if cfg.tensor_kwargs.get("label_range") is not None:
                    cfg["metric"][key]["tensor_range"] = cfg.tensor_kwargs.label_range
                else:
                    cfg["metric"][key]["tensor_range"] = "[0, 1]"
            if kwargs.get("min") is None:
                cfg["metric"][key]["min"] = cfg.tensor_kwargs.min
            if kwargs.get("max") is None:
                cfg["metric"][key]["max"] = cfg.tensor_kwargs.max

    if (
        cfg.resolution == 3
        and cfg.get("val_id_visual") is not None
        and cfg.crop_mode.lower() == "tile"
    ):
        cfg.val_id_visual = cfg.val_id_visual * 9 + 4

    if cfg.model_kwargs.get("spn") is None:
        if cfg.model_name.lower() == "edsr":
            cfg.model_kwargs.spn = False
        if cfg.model_name.lower() == "jspsr":
            cfg.model_kwargs.spn = True

    # for cleaning config file --->
    if cfg.get("scale") is None:
        cfg.scale = None

    if cfg.get("normalize") is None:
        cfg.normalize = False

    if cfg.optimizer_kwargs.get("diff_lr") is None:
        cfg.optimizer_kwargs.diff_lr = False

    if cfg.get("train_num_visual") is None:
        cfg.train_num_visual = 0

    if cfg.get("monitor_value") is None:
        cfg.monitor_value = None

    if cfg.get("mask_channel") is None:
        cfg.mask_channel = None

    return cfg
