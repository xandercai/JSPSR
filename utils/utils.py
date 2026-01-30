import time
import math
from natsort import natsorted
import re
import torch
from datetime import datetime
from collections import OrderedDict
from typing import Union
import json
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from pathlib import Path
from data.data_utils import RGB2YCbCr
from data.data_utils import ToDEM, ToTensor
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import rasterio
from rasterio.features import dataset_features
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import multiprocessing
from itertools import repeat
import geopandas as gpd
import mapply
from data.data_utils import TileCrop
from torchinfo import summary


mapply.init(
    n_workers=multiprocessing.cpu_count(),
    chunk_size=1,
    max_chunks_per_worker=1,
    progressbar=True,
)


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -9999_9999_9999
        self.min = 9999_9999_9999

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)

    def __str__(self):
        fmtstr = (
            "{name}: Count {count"
            + "}\tMin {min"
            + self.fmt
            + "} ms\tMax {max"
            + self.fmt
            + "} ms\tAvg {avg"
            + self.fmt
            + "} ms"
        )
        return fmtstr.format(**self.__dict__)


def get_model_summary(cfg, model):
    if cfg.verbose:
        B, C, H, W = 2, 3, cfg.patch_size, cfg.patch_size
        if hasattr(model, "in_channels"):
            if isinstance(model.in_channels, int):
                input_size = (B, model.in_channels, H, W)
            elif isinstance(model.in_channels, list):
                input_size = [(B, c, H, W) for c in model.in_channels]
            elif isinstance(model.in_channels, dict):
                if cfg.model_name.lower() in {"jspsr", "lrru"}:
                    input_size = [
                        (B, model.in_channels.get("lr_dem"), H, W),
                    ]
                    if "image" in model.in_channels:
                        input_size.append((B, model.in_channels.get("image"), H, W))
                    if "mask" in model.in_channels:
                        input_size.append((B, model.in_channels.get("mask"), H, W))
                    if "canopy" in model.in_channels:
                        input_size.append((B, model.in_channels.get("canopy"), H, W))
                    if "coord" in model.in_channels:
                        input_size.append((B, model.in_channels.get("coord"), H, W))
                if cfg.model_name.lower() in {"completionformer"}:
                    input_size = [
                        (B, model.in_channels.get("lr_dem"), H, W),
                        (
                            B,
                            model.in_channels.get("image", 0)
                            + model.in_channels.get("mask", 0)
                            + model.in_channels.get("canopy", 0)
                            + model.in_channels.get("coord", 0),
                            H,
                            W,
                        ),
                    ]
            else:
                raise NotImplementedError
            summary(
                model,
                input_size=input_size,
                col_names=[
                    "input_size",
                    "output_size",
                    "kernel_size",
                    "num_params",
                ],
                verbose=1,
                depth=5,
            )
        else:
            summary(
                model,
                input_size=(B, C, H, W),
                col_names=["input_size", "output_size", "kernel_size", "num_params"],
                verbose=1,
                depth=5,
            )


def get_loss_monitor(losses):
    """Return dictionary with loss meters to monitor training"""
    monitor = {loss: AverageMeter(f"Loss {loss}", ":6.4f") for loss in losses.keys()}
    monitor["Total"] = AverageMeter(f"Loss Total", ":6.4f")
    return monitor


def get_output(output):
    """Convert model prediction output from range [-1, 1] to range [0, 1]"""
    inputs = output.clone()
    inputs = (inputs + 1) / 2
    return inputs


def get_batch_pair(batch, model_name=None, input_data=None, gpu=0):
    """Return input data"""
    if torch.cuda.is_available():
        # lr_dem + image + mask + canopy + coord
        if model_name and model_name.lower() in {"jspsr", "lrru"}:
            assert input_data is not None, "input_data must be specified"
            lr_dem = batch["lr_dem"].cuda(gpu)
            input_list = [lr_dem]
            if "image" in input_data:
                image = batch["image"].cuda(gpu)
                input_list.append(image)
            if "mask" in input_data:
                mask = batch["mask"].cuda(gpu)
                input_list.append(mask)
            if "canopy" in input_data:
                canopy = batch["canopy"].cuda(gpu)
                input_list.append(canopy)
            if "coord" in input_data:
                coord = batch["coord"].cuda(gpu)
                input_list.append(coord)
            hr_dem = batch["hr_dem"].cuda(gpu)
            base_elev = [d["base"] for d in batch["meta"]]
            meta_data = batch["meta"]
            assert len(base_elev) == lr_dem.shape[0], base_elev
            base_elev = torch.as_tensor(
                base_elev, dtype=torch.float32, device=torch.device(gpu)
            )
            return input_list, hr_dem, base_elev, meta_data
        else:
            if input_data:
                if model_name and model_name.lower() in {
                    "completionformer",
                }:  # (lr_dem) + (image+mask+canopy+coord)
                    lr_dem = batch["lr_dem"].cuda(gpu)
                    input_list = [lr_dem]
                    new_images = {
                        "images": torch.zeros(
                            (
                                batch["lr_dem"].shape[0],
                                (
                                    input_data.get("image", 0)
                                    + input_data.get("mask", 0)
                                    + input_data.get("canopy", 0)
                                    + input_data.get("coord", 0)
                                ),
                                batch["lr_dem"].shape[2],
                                batch["lr_dem"].shape[3],
                            ),
                            dtype=torch.float32,
                            device=torch.device("cpu"),
                        )
                    }

                    for i in range(batch["lr_dem"].shape[0]):  # batch size
                        # order is important
                        # 3 or more channels image
                        next_channel = 0
                        if "image" in input_data:
                            new_images["images"][
                                i,
                                next_channel : next_channel + input_data["image"],
                                ...,
                            ] = batch["image"][i, ...]
                            next_channel += input_data["image"]
                        # 15 channels mask
                        if "mask" in input_data:
                            new_images["images"][
                                i, next_channel : next_channel + input_data["mask"], ...
                            ] = batch["mask"][i, ...]
                            next_channel += input_data["mask"]
                        # 1 channel canopy
                        if "canopy" in input_data:
                            new_images["images"][
                                i,
                                next_channel : next_channel + input_data["canopy"],
                                ...,
                            ] = batch["canopy"][i, ...]
                            next_channel += input_data["canopy"]
                        # 2 channels coord
                        if "coord" in input_data:
                            new_images["images"][
                                i,
                                next_channel : next_channel + input_data["coord"],
                                ...,
                            ] = batch["coord"][i, ...]
                            # next_channel += input_data["coord"]
                    images = new_images["images"].cuda(gpu, non_blocking=True)
                    input_list.append(images)
                    hr_dem = batch["hr_dem"].cuda(gpu)
                    base_elev = [d["base"] for d in batch["meta"]]
                    meta_data = batch["meta"]
                    assert len(base_elev) == lr_dem.shape[0], base_elev
                    base_elev = torch.as_tensor(
                        base_elev, dtype=torch.float32, device=torch.device(gpu)
                    )
                    return input_list, hr_dem, base_elev, meta_data
                else:  # (lr_dem + image + mask + canopy + coord)
                    new_batch = {
                        "images": torch.zeros(
                            (
                                batch["lr_dem"].shape[0],
                                (
                                    1  # lr_dem
                                    + input_data.get("image", 0)
                                    + input_data.get("mask", 0)
                                    + input_data.get("canopy", 0)
                                    + input_data.get("coord", 0)
                                ),
                                batch["lr_dem"].shape[2],
                                batch["lr_dem"].shape[3],
                            ),
                            dtype=torch.float32,
                            device=torch.device("cpu"),
                        ),
                        "labels": batch["hr_dem"],
                        "meta": batch["meta"],
                    }

                    for i in range(batch["lr_dem"].shape[0]):  # batch size
                        # order is important
                        # 1 channel dem
                        new_batch["images"][i, 0 : input_data["lr_dem"], ...] = batch[
                            "lr_dem"
                        ][i, ...]
                        next_channel = input_data["lr_dem"]
                        # 3 or more channels image
                        if "image" in input_data:
                            new_batch["images"][
                                i,
                                next_channel : next_channel + input_data["image"],
                                ...,
                            ] = batch["image"][i, ...]
                            next_channel += input_data["image"]
                        # 15 channels mask
                        if "mask" in input_data:
                            new_batch["images"][
                                i, next_channel : next_channel + input_data["mask"], ...
                            ] = batch["mask"][i, ...]
                            next_channel += input_data["mask"]
                        # 1 channel canopy
                        if "canopy" in input_data:
                            new_batch["images"][
                                i,
                                next_channel : next_channel + input_data["canopy"],
                                ...,
                            ] = batch["canopy"][i, ...]
                            next_channel += input_data["canopy"]
                        # 2 channels coord
                        if "coord" in input_data:
                            new_batch["images"][
                                i,
                                next_channel : next_channel + input_data["coord"],
                                ...,
                            ] = batch["coord"][i, ...]
                            # next_channel += input_data["coord"]
                    images = new_batch["images"].cuda(gpu, non_blocking=True)
                    labels = new_batch["labels"].cuda(gpu, non_blocking=True)
                    base_elev = [d["base"] for d in batch["meta"]]
                    meta_data = batch["meta"]
                    assert len(base_elev) == images.shape[0], base_elev
                    base_elev = torch.as_tensor(
                        base_elev, dtype=torch.float32, device=torch.device(gpu)
                    )
                    return [images], labels, base_elev, meta_data
            else:
                images = batch["images"].cuda(gpu, non_blocking=True)
                labels = batch["labels"].cuda(gpu, non_blocking=True)
                return [images], labels, None, None
    else:
        return None, None, None, None


def pair_state_dict(state_dict, model):
    """Pair state dict with model by layer position"""
    # print(model.state_dict().keys())
    if len(state_dict.keys()) != len(model.state_dict().keys()):
        print(
            f"Error: state_dict size {len(state_dict.keys())} != model size {len(model.state_dict().keys())}"
        )
    new_state_dict = OrderedDict()
    for i, (key, value) in enumerate(state_dict.items()):
        new_key = list(model.state_dict().keys())[i]
        new_state_dict[new_key] = value
    return new_state_dict


def load_model_from_url(url, model):
    if Path(url).is_file():
        state_dict = torch.load(url, map_location=lambda storage, loc: storage)
    else:
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location=lambda storage, loc: storage
        )
    # print(state_dict.keys())
    state_dict = pair_state_dict(state_dict, model)
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    return model


def load_state_dict(
    model: torch.nn.Module,
    state_dict: dict,
):
    # Process parameter dictionary
    model_state_dict = model.state_dict()

    # Traverse the model parameters, load the parameters in the pre-trained model into the current model
    new_state_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_state_dict.keys() and v.size() == model_state_dict[k].size()
    }

    # update model parameters
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def load_resume_state_dict(
    model,
    optimizer,
    scheduler,
    model_weights_path,
    resume=False,
):
    """Restore training model weights"""
    # Load model weights
    checkpoint = torch.load(
        model_weights_path, map_location=lambda storage, loc: storage
    )

    # Load training node parameters
    start_epoch = checkpoint["epoch"] if resume else 0
    best_results = checkpoint["best_result"]
    state_dict = checkpoint["state_dict"]

    model = load_state_dict(model, state_dict)

    if optimizer is not None and resume:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and resume:
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        scheduler = None

    return (
        model,
        start_epoch,
        best_results,
        optimizer,
        scheduler,
    )


def load_pretrained_state_dict(
    model,
    compile_state,
    model_weights_path,
):
    """Load pre-trained model weights"""

    checkpoint = torch.load(
        model_weights_path, map_location=lambda storage, loc: storage
    )
    state_dict = checkpoint["state_dict"]
    model = load_state_dict(model, compile_state, state_dict)
    return model


def get_number_after_letter(text, letter):
    pattern = re.compile(f"{letter}(\d+)")
    match = pattern.search(text)

    if match:
        return int(match.group(1))
    else:
        return None


def get_time_span(start: datetime, end: datetime):
    span = (end - start).seconds
    days = span // 86400
    hours = span // 3600
    minutes = (span % 3600) // 60
    seconds = span % 60
    return days, hours, minutes, seconds


def serialize_json(data: dict, path: Union[str, Path] = None) -> None:
    """serialize dictionary and print or save dictionary as json."""

    def _recursive_str(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    _recursive_str(v)
                else:
                    d[k] = str(v)
        else:
            d = str(d)
        return d

    print(
        "Test case configuration:\n", json.dumps(data, default=_recursive_str, indent=2)
    )

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, default=_recursive_str, indent=2)


def colorbar(im, label):
    last_axes = plt.gca()
    ax = im.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.12)
    cbar = fig.colorbar(im, cax=cax, label=label)
    plt.sca(last_axes)
    return cbar


def display_predictions(p, sample, pred, current_epoch=0):
    """Display input data and model predictions"""
    pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred

    for k, v in sample.items():
        sample[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v

    if p.tensor_kwargs and p.tensor_kwargs.get("label_range") == "[-1, 1]":
        pred = (pred + 1.0) / 2.0
        if "labels" in sample:
            sample["labels"] = (sample["labels"] + 1.0) / 2.0
    if p.tensor_kwargs and p.tensor_kwargs.get("label_range") == "[0, 255]":
        pred = pred * 255.0
        if "labels" in sample:
            sample["labels"] = sample["labels"] * 255.0

    pred = np.clip(pred, 0, 1)

    if p.get("y_channel_only", None) is not None:
        if not p.y_channel_only:
            pred = RGB2YCbCr.ycbcr2rgb(pred)
            sample["labels"] = RGB2YCbCr.ycbcr2rgb(sample["labels"])

    if p.val_border is not None and p.val_border > 0:
        h, w = pred.shape[-2:]
        pred = pred[
            ...,
            int(h * p.val_border) : h - int(h * p.val_border),
            int(w * p.val_border) : w - int(w * p.val_border),
        ]
        # cut the p.val_border
        # _keys = list(sample.keys())
        # for i in range(len(sample)):
        #     if _keys[i] != "meta":
        #         sample[_keys[i]] = sample[_keys[i]][
        #             ...,
        #             int(h * p.val_border) : h - int(h * p.val_border),
        #             int(w * p.val_border) : w - int(w * p.val_border),
        #         ]

    if "div2k" in p.dataset.lower():
        f, axs = plt.subplots(1, 3, figsize=(15, 5))
        for x in range(len(axs)):
            axs[x].cla()
        axs[0].imshow(np.transpose(sample["labels"], (1, 2, 0)), interpolation="none")
        axs[0].set_title("Ground truth")
        axs[1].imshow(np.transpose(sample["images"], (1, 2, 0)), interpolation="none")
        axs[1].set_title("Input")
        axs[2].imshow(np.transpose(pred, (1, 2, 0)), interpolation="none")
        axs[2].set_title("Prediction")
        subset = sample["meta"]["subset"]
        id = sample["meta"]["id"]
        shape = (
            sample["meta"]["shape"][0],
            sample["meta"]["shape"][1],
            sample["meta"]["shape"][2],
        )
        bbox = (
            sample["meta"]["bbox"][0],
            sample["meta"]["bbox"][1],
            sample["meta"]["bbox"][2],
            sample["meta"]["bbox"][3],
        )

        f.suptitle(
            f"E{current_epoch}-{subset}-{id}-{shape}-{bbox}",
            fontsize=15,
        )

    elif "dfc" in p.dataset.lower():
        if (
            p.tensor_kwargs
            and p.tensor_kwargs.get("min") is not None
            and p.tensor_kwargs.get("max") is not None
        ):
            base_elev = sample["meta"]["base"]
            sample["lr_dem"] = ToDEM.descale_data(
                sample["lr_dem"], p.tensor_kwargs.min, p.tensor_kwargs.max
            )
            sample["lr_dem"] = sample["lr_dem"] + base_elev
            sample["hr_dem"] = (
                ToDEM.descale_data(
                    sample["hr_dem"], p.tensor_kwargs.min, p.tensor_kwargs.max
                )
                + base_elev
            )
            pred = (
                ToDEM.descale_data(pred, p.tensor_kwargs.min, p.tensor_kwargs.max)
                + base_elev
            )
        elev_min = np.min((sample["lr_dem"], sample["hr_dem"]))
        elev_max = np.max((sample["lr_dem"], sample["hr_dem"]))
        # norm = plt.Normalize(elev_min, elev_max)

        if p.input_data.get("mask"):
            if p.mask_channel is not None and len(p.mask_channel) == 1:
                ncols = 5
            else:
                ncols = 6  # DEM, image, and mask
        elif p.input_data.get("canopy"):
            ncols = 5
        elif p.input_data.get("image"):
            ncols = 4  # DEM and image
        else:
            ncols = 3  # DEM only
        f, axs = plt.subplots(1, ncols, figsize=(ncols * 5, 5), sharey=True)
        for x in range(len(axs)):
            axs[x].cla()
        idx = 0
        if p.input_data.get("image"):
            axs[idx].imshow(
                np.transpose(sample["image"], (1, 2, 0)), interpolation="none"
            )
            axs[idx].set_title("Image")
            idx += 1
        axs[idx].imshow(
            sample["lr_dem"].squeeze(),
            interpolation="none",
            cmap="turbo",
            # cmap=cm.gist_ncar,
            # cmap="terrain",
            # cmap=cm.gist_earth,
            vmin=elev_min,
            vmax=elev_max,
        )
        axs[idx].set_title("LR DEM")
        idx += 1
        axs[idx].imshow(
            sample["hr_dem"].squeeze(),
            interpolation="none",
            cmap="turbo",
            # cmap=cm.gist_ncar,
            # cmap="terrain",
            # cmap=cm.gist_earth,
            vmin=elev_min,
            vmax=elev_max,
        )
        axs[idx].set_title("Ground truth")
        idx += 1
        if p.input_data.get("canopy"):
            sample["canopy"] = sample["canopy"] * 68  # 68 is the max canopy height
            axs[idx].imshow(
                sample["canopy"].squeeze(),
                interpolation="none",
                cmap="YlGn",
                vmin=0,
                vmax=68,
            )
            axs[idx].set_title("Canopy")
            idx += 1

        if p.input_data.get("mask"):
            if p.mask_channel is not None and len(p.mask_channel) == 1:
                axs[idx].imshow(sample["mask"].squeeze(), interpolation="none")
                axs[idx].set_title("Mask")
                idx += 1
            else:
                _nonzero_count_list = []
                for i in range(p.input_data.get("mask")):
                    _nonzero_count_list.append(
                        np.count_nonzero(sample["mask"].squeeze()[i])
                    )
                _mask_plot_list = []
                for i in range(
                    2
                ):  # select two channels with most non-zero pixels to plot
                    _id = np.argmax(_nonzero_count_list)
                    _mask_plot_list.append(_id)
                    _nonzero_count_list.pop(_id)

                axs[idx].imshow(
                    sample["mask"].squeeze()[_mask_plot_list[0]], interpolation="none"
                )
                axs[idx].set_title(f"Mask channel {_mask_plot_list[0]}")
                idx += 1
                axs[idx].imshow(
                    sample["mask"].squeeze()[_mask_plot_list[1]], interpolation="none"
                )
                axs[idx].set_title(f"Mask channel {_mask_plot_list[1]}")
                idx += 1
        _im = axs[idx].imshow(
            pred.squeeze(),
            interpolation="none",
            cmap="turbo",
            # cmap=cm.gist_ncar,
            # cmap="terrain",
            # cmap=cm.gist_earth,
            vmin=elev_min,
            vmax=elev_max,
        )
        colorbar(_im, "Elevation (m)")
        axs[idx].set_title("Prediction")
        subset = sample["meta"]["subset"]
        id = sample["meta"]["id"]
        shape = (
            sample["meta"]["shape"][0],
            sample["meta"]["shape"][1],
            sample["meta"]["shape"][2],
        )
        bbox = (
            sample["meta"]["bbox"][0],
            sample["meta"]["bbox"][1],
            sample["meta"]["bbox"][2],
            sample["meta"]["bbox"][3],
        )

        f.suptitle(
            f"E{current_epoch}-{subset}-{id}-{shape}-{bbox}",
            fontsize=15,
        )
    else:
        raise NotImplementedError

    if p.val_save_visual:
        # path = Path(p.result_dir) / "visual" / f"E{current_epoch}_{subset}_{id}.png"
        path = Path(p.result_dir) / "visual" / f"E{current_epoch}_{subset}_{id}.tiff"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            path,
            format="tiff",
            dpi=300,
            bbox_inches="tight",
        )

    plt.tight_layout()
    plt.show()
    plt.close(f)


# double check parameter number.
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Params:{total_params}")
    return total_params


def get_diff_params(model, diff_params=None):
    if not isinstance(diff_params, list):
        diff_params = [diff_params]

    # print("Different learning rate:", diff_params)

    diff_params_list = list(
        filter(
            lambda kv: [kv[0] for param in diff_params if param in kv[0]],
            model.named_parameters(),
        )
    )

    diff_params_name_list = [p[0] for p in diff_params_list]
    base_params_list = list(
        filter(lambda kv: kv[0] not in diff_params_name_list, model.named_parameters())
    )

    print("base_params_list", len(base_params_list))
    # [print(p[0]) for p in base_params_list]
    print("diff_params_list", len(diff_params_list))
    # [print(p[0]) for p in diff_params_list]
    return base_params_list, diff_params_list


# https://stackoverflow.com/questions/45718523/pass-kwargs-to-starmap-while-using-pool-in-python
def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def gen_crop_subset(
    src_root,
    dst_root,
    dataset,
    subset,
    file_name,
    suffix,
    crs="EPSG:2154",
    geometry=None,
    nodata=None,
    buffer=0,
    overwrite=False,
):
    # print(f"\nprocessing {dataset} {subset} {file_name} ...")
    src_f = (
        Path(src_root)
        / Path(dataset)
        / Path(subset)
        / Path(file_name + f"_{suffix}.tif")
    )
    dst_f = (
        Path(dst_root).parent
        / Path(subset.lower() + "_30")
        / Path(dataset)
        / Path(file_name + f"_{suffix}.tif")
    )
    if dst_f.exists() and not overwrite:
        print(f"File {dst_f} exist, nonthing to do.")
        return
    dst_f.parent.mkdir(parents=True, exist_ok=True)

    geom_clip = geometry.buffer(buffer, join_style="mitre")

    with rxr.open_rasterio(src_f, cache=True) as rds:
        _rds = rds.rio.clip_box(*geom_clip.bounds, crs=crs)
        nodata_mask = _rds == nodata
        if nodata_mask.any():
            print(f"!!!!!!!!!!!!!!!!!!!! Nodata exists {dataset} {subset} {file_name}")
        if nodata is not None:
            rds.rio.write_nodata(nodata, inplace=True)
        _rds.rio.to_raster(dst_f, compress="LZW", tiled=True)
    # print(f"Finish {dataset} {subset} {file_name} process.")


def gen_weight_row(sda, i, patch_size):

    arr = np.ones((sda.shape[1:]))

    w_h = 334  # 3m resolution sample size
    w_l = patch_size  # prediction size
    w_l_c, _ = arr.shape  # border cropped prediction  -> 116
    w_h_c = w_h - (w_l - w_l_c)  # border cropped full prediction  -> 322
    s, n = TileCrop.get_tile(w_h_c, w_l_c)  # tile strip and tile number  -> 103, 9
    assert n == int(n**0.5) ** 2, f"n {n} is not a square number."
    p = w_l_c - s  # overlapped pixel number  -> 13

    # strip weight
    weight = np.linspace(1, 0, p + 2)[1:-1]  # remove 1 and 0

    # single size 1d weight
    arr1d_weight_1 = np.ones(w_l_c)
    arr1d_weight_1[-p:] = weight

    # both size 1d weight
    arr1d_weight_2 = np.ones(w_l_c)
    arr1d_weight_2[:p] = np.flip(weight)
    arr1d_weight_2[-p:] = weight

    if n**0.5 == 3:
        if i % n**0.5 == 0:
            arr = np.apply_along_axis(lambda row: row * arr1d_weight_1, axis=1, arr=arr)
        elif i % n**0.5 == 1:
            arr = np.apply_along_axis(lambda row: row * arr1d_weight_2, axis=1, arr=arr)
        elif i % n**0.5 == 2:
            arr = np.apply_along_axis(
                lambda row: row * np.flip(arr1d_weight_1), axis=1, arr=arr
            )
        else:
            raise NotImplementedError(f"n {n} is not 9, but {i % n**0.5}.")
    elif n**0.5 == 2:
        if i % n**0.5 == 0:
            arr = np.apply_along_axis(lambda row: row * arr1d_weight_1, axis=1, arr=arr)
        elif i % n**0.5 == 1:
            arr = np.apply_along_axis(
                lambda row: row * np.flip(arr1d_weight_1), axis=1, arr=arr
            )
        else:
            raise NotImplementedError(f"n {n} is not 4, but {i % n**0.5}.")
    else:
        raise NotImplementedError(f"n {n} is not 9 or 4, but {i % n**0.5}.")

    return arr


def gen_weight_col(sda, i, patch_size):

    arr = np.ones((sda.shape[1:]))

    w_h = 334  # 3m resolution sample size
    w_l = patch_size  # prediction size
    w_l_c, _ = arr.shape  # border cropped prediction  -> 116
    w_h_c = w_h - (w_l - w_l_c)  # border cropped full prediction  -> 322
    s, n = TileCrop.get_tile(w_h_c, w_l_c)  # tile strip and tile number  -> 103, 9
    p = w_l_c - s  # overlapped pixel number  -> 13

    # strip weight
    weight = np.linspace(1, 0, p + 2)[1:-1]  # remove 1 and 0

    # single size 1d weight
    arr1d_weight_1 = np.ones(w_l_c)
    arr1d_weight_1[-p:] = weight

    # both size 1d weight
    arr1d_weight_2 = np.ones(w_l_c)
    arr1d_weight_2[:p] = np.flip(weight)
    arr1d_weight_2[-p:] = weight

    if n**0.5 == 3:
        if i % 3 == 0:
            arr = np.apply_along_axis(lambda col: col * arr1d_weight_1, axis=0, arr=arr)
        elif i % 3 == 1:
            arr = np.apply_along_axis(lambda col: col * arr1d_weight_2, axis=0, arr=arr)
        elif i % 3 == 2:
            arr = np.apply_along_axis(
                lambda col: col * np.flip(arr1d_weight_1), axis=0, arr=arr
            )
        else:
            raise NotImplementedError(f"n {n} is not 9, but {i % n**0.5}.")
    elif n**0.5 == 2:
        if i % 2 == 0:
            arr = np.apply_along_axis(lambda col: col * arr1d_weight_1, axis=0, arr=arr)
        elif i % 2 == 1:
            arr = np.apply_along_axis(
                lambda col: col * np.flip(arr1d_weight_1), axis=0, arr=arr
            )
        else:
            raise NotImplementedError(f"n {n} is not 4, but {i % n**0.5}.")
    else:
        raise NotImplementedError(f"n {n} is not 9 or 4, but {i % n**0.5}.")

    return arr


def copyto_add(merged_data, new_data, merged_mask, new_mask, **kwargs):
    # print(merged_data.shape, new_data.shape, merged_mask.shape, new_mask.shape)
    # print(merged_mask[:, :1, :])
    # print(new_mask[:, :1, :])
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    merged_data[mask] = np.add(merged_data, new_data, where=mask)[mask]
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def merge_dem(file_list, border=0.0, crs="EPSG:2154", method="first", save=None):
    xds_list = []
    # gdf = gpd.GeoDataFrame()
    for i, f in enumerate(file_list):
        with rasterio.open(f) as src:
            patch_size = src.meta["width"]
            _gdf = gpd.GeoDataFrame.from_features(
                dataset_features(
                    src,
                    bidx=1,
                    as_mask=True,
                    geographic=False,
                    band=False,
                    with_nodata=True,
                )
            )
            _gdf.crs = crs
            # _gdf["name"] = f.stem
            _gdf["geometry"] = _gdf["geometry"].buffer(
                -int(src.meta["width"] * border) * src.meta["transform"][0],
                join_style="mitre",
            )
            # _gdf = _gdf[["name", "geometry"]]
            _gdf = _gdf[["geometry"]]
            # gdf = pd.concat([gdf, _gdf], ignore_index=True)
        with rxr.open_rasterio(f) as xds:
            xds = xds.rio.clip_box(*_gdf.total_bounds, crs=crs)
            weight = gen_weight_row(xds, i, patch_size)
            xds = np.multiply(xds, weight)
            xds_list.append(xds)

    xds_row_list = []
    for i in range(len(xds_list)):
        if i % len(xds_list) ** 0.5 == 0:
            xds = merge_arrays(
                xds_list[i : i + int(len(xds_list) ** 0.5)],
                nodata=-99999,
                method=method,
            )
            xds_row_list.append(xds)
    xds_col_list = []
    for i, xds in enumerate(xds_row_list):
        weight = gen_weight_col(xds, i, patch_size)
        xds = np.multiply(xds, weight)
        xds_col_list.append(xds)
    xds = merge_arrays(xds_col_list, nodata=-99999, method=method)

    if save is not None:
        save_path = Path(file_list[0]).parent / (
            Path(file_list[0]).stem.split("_")[0] + ".tif"
        )
        xds.rio.to_raster(save_path, driver="GTiff", compress="LZW")

    return xds.to_numpy().squeeze()


def summarise_evaluation(
    p, sr_dir, plot_path="", inference=False, online=False, plot=True
):
    """Validate the evaluation directories
    Note that evaluation metrics calculate the average metrics of each sample in validation set,
    while here it calculate the overall metrics of the whole validation set. The results may be different.
    """
    assert not (
        inference == True and online == True
    ), "inference and online cannot be True at the same time"

    if "sr_dem" in Path(sr_dir).as_posix():
        sr_dir = Path(sr_dir)
    else:
        sr_dir = Path(sr_dir) / "sr_dem"
    assert sr_dir.is_dir(), f"{sr_dir} is not exist"
    dataset_path = Path(p.dataset_path)
    assert dataset_path.is_dir(), f"{dataset_path} is not exist"

    if not inference and not online and p.resolution == 3:
        sr_list = natsorted([f.as_posix() for f in sr_dir.rglob("*_0.tif")])
    else:
        sr_list = natsorted([f.as_posix() for f in sr_dir.rglob("*.tif")])

    sr_dataset_list = set([Path(f).parent.name for f in sr_list])

    file_list = [
        f
        for f in dataset_path.rglob("*.tif")
        if f.parent.parent.name in sr_dataset_list
    ]
    gt_list = natsorted([f.as_posix() for f in file_list if "RGEALTI" == f.parent.name])
    cop_list = natsorted([f for f in file_list if "COP30" == f.parent.name])
    fab_list = natsorted([f for f in file_list if "FABDEM" == f.parent.name])
    fat_list = natsorted([f for f in file_list if "FATHOM" == f.parent.name])

    if online:
        online_rmse_list = []
        online_median_list = []
        online_nmad_list = []
        online_le95_list = []
        online_psnr_list = []

        if p.resolution == 3:
            cop_list = [x for x in cop_list for _ in range(p.patches_per_image)]
            fab_list = [x for x in fab_list for _ in range(p.patches_per_image)]
            fat_list = [x for x in fat_list for _ in range(p.patches_per_image)]
            gt_list = [x for x in gt_list for _ in range(p.patches_per_image)]

        assert (
            len(sr_list)
            == len(gt_list)
            == len(cop_list)
            == len(fab_list)
            == len(fat_list)
        ), f"{len(sr_list)} {len(gt_list)} {len(cop_list)} {len(fab_list)} {len(fat_list)}"

        cnt_0_rmse_cop = 0
        cnt_0_rmse_fab = 0
        cnt_0_rmse_fat = 0
        cnt_0_rmse_sr = 0
        cnt_crop_col = 0  # for online mode only
        cnt_crop_row = 0  # for online mode only
        stride = 0
        n_tile = 0
        for cop, fab, fat, sr, gt in zip(
            cop_list, fab_list, fat_list, sr_list, gt_list
        ):
            arr_cop = rasterio.open(cop).read(1)
            arr_fab = rasterio.open(fab).read(1)
            arr_fat = rasterio.open(fat).read(1)
            arr_gt = rasterio.open(gt).read(1)
            arr_sr = rasterio.open(sr).read(1)

            if p.resolution == 3:
                h, w = arr_gt.shape
                if cnt_crop_row == 0 and cnt_crop_col == 0:
                    # calculate stride
                    stride, n_tile = TileCrop.get_tile(w, p.patch_size)
                arr_cop = arr_cop[
                    stride * cnt_crop_row : stride * cnt_crop_row + p.patch_size,
                    stride * cnt_crop_col : stride * cnt_crop_col + p.patch_size,
                ]
                arr_fab = arr_fab[
                    stride * cnt_crop_row : stride * cnt_crop_row + p.patch_size,
                    stride * cnt_crop_col : stride * cnt_crop_col + p.patch_size,
                ]
                arr_fat = arr_fat[
                    stride * cnt_crop_row : stride * cnt_crop_row + p.patch_size,
                    stride * cnt_crop_col : stride * cnt_crop_col + p.patch_size,
                ]
                arr_gt = arr_gt[
                    stride * cnt_crop_row : stride * cnt_crop_row + p.patch_size,
                    stride * cnt_crop_col : stride * cnt_crop_col + p.patch_size,
                ]

                # step over the image
                cnt_crop_col += 1
                if cnt_crop_col == n_tile**0.5:
                    cnt_crop_row += 1
                    if cnt_crop_row == n_tile**0.5 and cnt_crop_col == n_tile**0.5:
                        cnt_crop_row = 0
                        stride = 0
                        n_tile = None
                    cnt_crop_col = 0

            if p.val_border > 0:
                h0, w0 = p.patch_size, p.patch_size
                h1, w1 = arr_gt.shape

                arr_cop = arr_cop[
                    int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                    int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
                ].flatten()
                arr_fab = arr_fab[
                    int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                    int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
                ].flatten()
                arr_fat = arr_fat[
                    int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                    int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
                ].flatten()
                arr_gt = arr_gt[
                    int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                    int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
                ].flatten()

                if arr_sr.shape == (h1, w1):
                    arr_sr = arr_sr[
                        int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                        int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
                    ].flatten()
                else:
                    arr_sr = arr_sr.flatten()
            else:
                arr_cop = arr_cop.flatten()
                arr_fab = arr_fab.flatten()
                arr_fat = arr_fat.flatten()
                arr_sr = arr_sr.flatten()
                arr_gt = arr_gt.flatten()

            assert (
                arr_cop.shape
                == arr_fab.shape
                == arr_fat.shape
                == arr_sr.shape
                == arr_gt.shape
            ), f"{arr_cop.shape} {arr_fab.shape} {arr_fat.shape} {arr_sr.shape} {arr_gt.shape}"

            _cop_error = arr_cop - arr_gt
            _fab_error = arr_fab - arr_gt
            _fat_error = arr_fat - arr_gt
            _sr_error = arr_sr - arr_gt

            _cop_sqrd_error = _cop_error**2
            _fab_sqrd_error = _fab_error**2
            _fat_sqrd_error = _fat_error**2
            _sr_sqrd_error = _sr_error**2

            _cop_rmse = math.sqrt(np.mean(_cop_sqrd_error))
            if _cop_rmse < 0.01:
                cnt_0_rmse_cop += 1
                # print("0 COP RMSE", cnt_0_rmse, _cop_rmse, cop)
            _fab_rmse = math.sqrt(np.mean((_fab_sqrd_error)))
            if _fab_rmse < 0.01:
                cnt_0_rmse_fab += 1
                # print("0 FAB RMSE", cnt_0_rmse, _fab_rmse, fab)
            _fat_rmse = math.sqrt(np.mean((_fat_sqrd_error)))
            if _fat_rmse < 0.01:
                cnt_0_rmse_fat += 1
                # print("0 FAT RMSE", cnt_0_rmse, _fat_rmse, fat)
            _sr_rmse = math.sqrt(np.mean((_sr_sqrd_error)))
            if _sr_rmse < 0.01:
                cnt_0_rmse_sr += 1
                print("0 SR RMSE", cnt_0_rmse_sr, _sr_rmse, sr)
            _cop_median = np.median(_cop_error)
            _fab_median = np.median(_fab_error)
            _fat_median = np.median(_fat_error)
            _sr_median = np.median(_sr_error)
            _cop_nmad = 1.4826 * np.median(np.abs(_cop_error - _cop_median))
            _fab_nmad = 1.4826 * np.median(np.abs(_fab_error - _fab_median))
            _fat_nmad = 1.4826 * np.median(np.abs(_fat_error - _fat_median))
            _sr_nmad = 1.4826 * np.median(np.abs(_sr_error - _sr_median))
            _cop_le95 = np.percentile(np.abs(_cop_error), 95)
            _fab_le95 = np.percentile(np.abs(_fab_error), 95)
            _fat_le95 = np.percentile(np.abs(_fat_error), 95)
            _sr_le95 = np.percentile(np.abs(_sr_error), 95)
            _cop_psnr = 20 * np.log10(p.tensor_kwargs.max / (_cop_rmse + 1e-8))
            _fab_psnr = 20 * np.log10(p.tensor_kwargs.max / (_fab_rmse + 1e-8))
            _fat_psnr = 20 * np.log10(p.tensor_kwargs.max / (_fat_rmse + 1e-8))
            _sr_psnr = 20 * np.log10(p.tensor_kwargs.max / _sr_rmse)
            online_rmse_list.append([_cop_rmse, _fab_rmse, _fat_rmse, _sr_rmse])
            online_median_list.append(
                [_cop_median, _fab_median, _fat_median, _sr_median]
            )
            online_nmad_list.append([_cop_nmad, _fab_nmad, _fat_nmad, _sr_nmad])
            online_le95_list.append([_cop_le95, _fab_le95, _fat_le95, _sr_le95])
            online_psnr_list.append([_cop_psnr, _fab_psnr, _fat_psnr, _sr_psnr])

        online_cop_rmse = np.mean([i[0] for i in online_rmse_list])
        online_fab_rmse = np.mean([i[1] for i in online_rmse_list])
        online_fat_rmse = np.mean([i[2] for i in online_rmse_list])
        online_sr_rmse = np.mean([i[3] for i in online_rmse_list])
        online_cop_median = np.mean([i[0] for i in online_median_list])
        online_fab_median = np.mean([i[1] for i in online_median_list])
        online_fat_median = np.mean([i[2] for i in online_median_list])
        online_sr_median = np.mean([i[3] for i in online_median_list])
        online_cop_nmad = np.mean([i[0] for i in online_nmad_list])
        online_fab_nmad = np.mean([i[1] for i in online_nmad_list])
        online_fat_nmad = np.mean([i[2] for i in online_nmad_list])
        online_sr_nmad = np.mean([i[3] for i in online_nmad_list])
        online_cop_le95 = np.mean([i[0] for i in online_le95_list])
        online_fab_le95 = np.mean([i[1] for i in online_le95_list])
        online_fat_le95 = np.mean([i[2] for i in online_le95_list])
        online_sr_le95 = np.mean([i[3] for i in online_le95_list])
        online_cop_psnr = np.mean([i[0] for i in online_psnr_list])
        online_fab_psnr = np.mean([i[1] for i in online_psnr_list])
        online_fat_psnr = np.mean([i[2] for i in online_psnr_list])
        online_sr_psnr = np.mean([i[3] for i in online_psnr_list])

        print("\n")
        print(
            f"Online COP30\tvs\tGT\t{p.val_border}::\tRMSE: {online_cop_rmse:.4f} Median: {online_cop_median:.4f} NMAD: {online_cop_nmad:.4f} LE95: {online_cop_le95:.4f} PSNR: {online_cop_psnr:.4f}"
        )
        print(
            f"Online FABDEM\tvs\tGT\t{p.val_border}::\tRMSE: {online_fab_rmse:.4f} Median: {online_fab_median:.4f} NMAD: {online_fab_nmad:.4f} LE95: {online_fab_le95:.4f} PSNR: {online_fab_psnr:.4f}"
        )
        print(
            f"Online FATHOM\tvs\tGT\t{p.val_border}::\tRMSE: {online_fat_rmse:.4f} Median: {online_fat_median:.4f} NMAD: {online_fat_nmad:.4f} LE95: {online_fat_le95:.4f} PSNR: {online_fat_psnr:.4f}"
        )
        print(
            f"Online SR    \tvs\tGT\t{p.val_border}::\tRMSE: {online_sr_rmse:.4f} Median: {online_sr_median:.4f} NMAD: {online_sr_nmad:.4f} LE95: {online_sr_le95:.4f} PSNR: {online_sr_psnr:.4f}"
        )
        print(
            f"0 RMSE count: COP {cnt_0_rmse_cop}, FAB {cnt_0_rmse_fab}, FAT {cnt_0_rmse_fat}, SR {cnt_0_rmse_sr}"
        )
        print("\n")

    # offline
    if "sr_dem" in Path(sr_dir).as_posix():
        sr_dir = Path(sr_dir)
    else:
        sr_dir = Path(sr_dir) / "sr_dem"
    assert sr_dir.is_dir(), f"{sr_dir} is not exist"
    dataset_path = Path(p.dataset_path)
    assert dataset_path.is_dir(), f"{dataset_path} is not exist"

    if not inference and p.resolution == 3:
        sr_list = natsorted([f.as_posix() for f in sr_dir.rglob("*_0.tif")])
    else:
        sr_list = natsorted([f.as_posix() for f in sr_dir.rglob("*.tif")])

    sr_dataset_list = set([Path(f).parent.name for f in sr_list])

    file_list = [
        f
        for f in dataset_path.rglob("*.tif")
        if f.parent.parent.name in sr_dataset_list
    ]
    gt_list = natsorted([f.as_posix() for f in file_list if "RGEALTI" == f.parent.name])
    cop_list = natsorted([f for f in file_list if "COP30" == f.parent.name])
    fab_list = natsorted([f for f in file_list if "FABDEM" == f.parent.name])
    fat_list = natsorted([f for f in file_list if "FATHOM" == f.parent.name])

    assert (
        len(sr_list) == len(gt_list) == len(cop_list) == len(fab_list) == len(fat_list)
    ), f"{len(sr_list)} {len(gt_list)} {len(cop_list)} {len(fab_list)} {len(fat_list)}"

    tmp_sr_error = np.array([]).astype(np.float32)
    tmp_cop_error = np.array([]).astype(np.float32)
    tmp_fab_error = np.array([]).astype(np.float32)
    tmp_fat_error = np.array([]).astype(np.float32)
    tmp_sr_sqrd_error = np.array([]).astype(np.float32)
    tmp_cop_sqrd_error = np.array([]).astype(np.float32)
    tmp_fab_sqrd_error = np.array([]).astype(np.float32)
    tmp_fat_sqrd_error = np.array([]).astype(np.float32)

    for cop, fab, fat, sr, gt in zip(cop_list, fab_list, fat_list, sr_list, gt_list):
        arr_cop = rasterio.open(cop).read(1)
        arr_fab = rasterio.open(fab).read(1)
        arr_fat = rasterio.open(fat).read(1)
        arr_gt = rasterio.open(gt).read(1)

        if p.resolution == 3:
            assert arr_gt.shape[0] > p.patch_size, f"{arr_gt.shape} {p.patch_size}"

            _name, _index = Path(sr).stem.split("_")
            assert int(_index) == 0, f"{_index}"
            assert (
                _name in Path(cop).stem
                and _name in Path(fab).stem
                and _name in Path(fat).stem
                and _name in Path(gt).stem
            ), f"{_name}, {cop}, {fab}, {fat}, {gt}"
            _sr_list = [
                (
                    Path(sr).parent / Path(Path(sr).stem.split("_")[0] + f"_{i}.tif")
                ).as_posix()
                for i in range(p.patches_per_image)
            ]
            # merge method is average for overlapping area and removed border
            # arr_sr = merge_dem(_sr_list, p.val_border, method=copyto_avg)
            arr_sr = merge_dem(_sr_list, p.val_border, method=copyto_add, save=True)
        else:
            arr_sr = rasterio.open(sr).read(1)

        if p.val_border > 0:
            h0, w0 = p.patch_size, p.patch_size
            h1, w1 = arr_gt.shape

            arr_cop = arr_cop[
                int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
            ].flatten()

            arr_fab = arr_fab[
                int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
            ].flatten()

            arr_fat = arr_fat[
                int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
            ].flatten()

            arr_gt = arr_gt[
                int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
            ].flatten()

            if arr_sr.shape == (h1, w1):
                arr_sr = arr_sr[
                    int(h0 * p.val_border) : h1 - int(h0 * p.val_border),
                    int(w0 * p.val_border) : w1 - int(w0 * p.val_border),
                ].flatten()
            else:
                arr_sr = arr_sr.flatten()

            assert (
                arr_cop.shape
                == arr_fab.shape
                == arr_fat.shape
                == arr_sr.shape
                == arr_gt.shape
            ), f"{arr_cop.shape} {arr_fab.shape} {arr_fat.shape} {arr_sr.shape} {arr_gt.shape}"

        else:
            arr_cop = arr_cop.flatten()
            arr_fab = arr_fab.flatten()
            arr_fat = arr_fat.flatten()
            arr_sr = arr_sr.flatten()
            arr_gt = arr_gt.flatten()

        _cop_error = arr_cop - arr_gt
        _fab_error = arr_fab - arr_gt
        _fat_error = arr_fat - arr_gt
        _sr_error = arr_sr - arr_gt

        tmp_cop_error = np.concatenate((tmp_cop_error, _cop_error))
        tmp_fab_error = np.concatenate((tmp_fab_error, _fab_error))
        tmp_fat_error = np.concatenate((tmp_fat_error, _fat_error))
        tmp_sr_error = np.concatenate((tmp_sr_error, _sr_error))
        tmp_cop_sqrd_error = np.concatenate((tmp_cop_sqrd_error, _cop_error**2))
        tmp_fab_sqrd_error = np.concatenate((tmp_fab_sqrd_error, _fab_error**2))
        tmp_fat_sqrd_error = np.concatenate((tmp_fat_sqrd_error, _fat_error**2))
        tmp_sr_sqrd_error = np.concatenate((tmp_sr_sqrd_error, _sr_error**2))

    cop_median = np.median(tmp_cop_error)
    fab_median = np.median(tmp_fab_error)
    fat_median = np.median(tmp_fat_error)
    sr_median = np.median(tmp_sr_error)
    cop_rmse = math.sqrt(np.mean(tmp_cop_sqrd_error))
    fab_rmse = math.sqrt(np.mean(tmp_fab_sqrd_error))
    fat_rmse = math.sqrt(np.mean(tmp_fat_sqrd_error))
    sr_rmse = math.sqrt(np.mean(tmp_sr_sqrd_error))
    cop_nmad = 1.4826 * np.median(np.abs(tmp_cop_error - cop_median))
    fab_nmad = 1.4826 * np.median(np.abs(tmp_fab_error - fab_median))
    fat_nmad = 1.4826 * np.median(np.abs(tmp_fat_error - fat_median))
    sr_nmad = 1.4826 * np.median(np.abs(tmp_sr_error - sr_median))
    cop_le95 = np.percentile(np.abs(tmp_cop_error), 95)
    fab_le95 = np.percentile(np.abs(tmp_fab_error), 95)
    fat_le95 = np.percentile(np.abs(tmp_fat_error), 95)
    sr_le95 = np.percentile(np.abs(tmp_sr_error), 95)
    cop_psnr = 20 * np.log10(p.tensor_kwargs.max / cop_rmse)
    fab_psnr = 20 * np.log10(p.tensor_kwargs.max / fab_rmse)
    fat_psnr = 20 * np.log10(p.tensor_kwargs.max / fat_rmse)
    sr_psnr = 20 * np.log10(p.tensor_kwargs.max / sr_rmse)

    print(
        f"COP30\tvs\tGT\t{p.val_border}::\tRMSE: {cop_rmse:.4f} Median: {cop_median:.4f} NMAD: {cop_nmad:.4f} LE95: {cop_le95:.4f} PSNR: {cop_psnr:.4f}"
    )
    print(
        f"FABDEM\tvs\tGT\t{p.val_border}::\tRMSE: {fab_rmse:.4f} Median: {fab_median:.4f} NMAD: {fab_nmad:.4f} LE95: {fab_le95:.4f} PSNR: {fab_psnr:.4f}"
    )
    print(
        f"FATHOM\tvs\tGT\t{p.val_border}::\tRMSE: {fat_rmse:.4f} Median: {fat_median:.4f} NMAD: {fat_nmad:.4f} LE95: {fat_le95:.4f} PSNR: {fat_psnr:.4f}"
    )
    print(
        f"SR  \tvs\tGT\t{p.val_border}::\tRMSE: {sr_rmse:.4f} Median: {sr_median:.4f} NMAD: {sr_nmad:.4f} LE95: {sr_le95:.4f} PSNR: {sr_psnr:.4f}"
    )

    if plot:
        df_metric = pd.DataFrame(
            [
                ["COP30", "RMSE", cop_rmse],
                ["COP30", "Median", cop_median],
                ["COP30", "NMAD", cop_nmad],
                ["COP30", "LE95", cop_le95],
                ["FABDEM", "RMSE", fab_rmse],
                ["FABDEM", "Median", fab_median],
                ["FABDEM", "NMAD", fab_nmad],
                ["FABDEM", "LE95", fab_le95],
                ["FATHOM", "RMSE", fat_rmse],
                ["FATHOM", "Median", fat_median],
                ["FATHOM", "NMAD", fat_nmad],
                ["FATHOM", "LE95", fat_le95],
                ["JSPSR", "RMSE", sr_rmse],
                ["JSPSR", "Median", sr_median],
                ["JSPSR", "NMAD", sr_nmad],
                ["JSPSR", "LE95", sr_le95],
            ],
            columns=["Dataset", "Metric", "Error"],
        )

        # remove outliers, here define outliers are outside [-10, 10]m
        tmp_cop_error = tmp_cop_error[(tmp_cop_error >= -5) & (tmp_cop_error <= 5)]
        tmp_fab_error = tmp_fab_error[(tmp_fab_error >= -5) & (tmp_fab_error <= 5)]
        tmp_fat_error = tmp_fat_error[(tmp_fat_error >= -5) & (tmp_fat_error <= 5)]
        tmp_sr_error = tmp_sr_error[(tmp_sr_error >= -5) & (tmp_sr_error <= 5)]

        df_cop = pd.DataFrame(tmp_cop_error, columns=["Error"]).astype("float16")
        df_fab = pd.DataFrame(tmp_fab_error, columns=["Error"]).astype("float16")
        df_fat = pd.DataFrame(tmp_fat_error, columns=["Error"]).astype("float16")
        df_sr = pd.DataFrame(tmp_sr_error, columns=["Error"]).astype("float16")
        df_err = pd.concat(
            [
                df_cop.assign(Dataset="COP30"),
                df_fab.assign(Dataset="FABDEM"),
                df_fat.assign(Dataset="FATHOM"),
                df_sr.assign(Dataset="JSPSR"),
            ]
        )
        del df_cop, df_fab, df_fat, df_sr

        plot_settings = {
            "ytick.labelsize": 16,
            "xtick.labelsize": 16,
            "font.size": 22,
            "figure.figsize": (10, 10),
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "lines.linewidth": 2,
            "lines.markersize": 3,
            "legend.fontsize": 11,
            "mathtext.fontset": "stix",
            "axes.titley": 1,
            "axes.xmargin": 0,
            # 'font.family': 'STIXGeneral'
            "font.family": "sans-serif",
        }
        plt.style.use(plot_settings)
        colors = mpl.colormaps["Set1"].colors

        fig, axs = plt.subplots(
            1,
            2,
            figsize=(17, 5),
            sharex=False,
            sharey=False,
            gridspec_kw={"width_ratios": [3, 1]},
            dpi=600,
        )

        sns.kdeplot(
            df_err,
            x="Error",
            hue="Dataset",
            bw_adjust=1,
            cut=0.5,
            fill=False,
            common_norm=False,
            linewidth=1,
            ax=axs[0],
        )
        # axs[0].get_legend().set_visible(False)
        axs[0].set(xlabel="Elevation Error [m]", ylabel="Density")
        axs[0].set_title("Elevation Error Distribution in [-5, 5] m", weight="bold")

        sns.barplot(
            df_metric,
            x="Error",
            y="Metric",
            hue="Dataset",
            legend=True,
            errorbar=None,
            orient="y",
            ax=axs[1],
        )
        axs[1].set(xlabel="Metric Value [m]")
        axs[1].set_title("Metrics", weight="bold")

        # fig.suptitle("Elevation Error and Metrics by DEM Dataset", weight="bold", size=24)
        # fig.supxlabel("Elevation Error [m]", y=0.05, size=16)

        sns.despine()
        plt.tight_layout()

        if plot_path == "":
            if not inference:
                plot_path = Path(sr_dir).parent / "visual" / f"final_dist.tiff"
            else:
                plot_path = (
                    Path(sr_dir).parent / "visual" / f"final_dist_inference.tiff"
                )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            print("Save plot to", plot_path)

        # plt.savefig(plot_path, dpi=300, transparent=True, bbox_inches="tight")
        plt.savefig(
            plot_path,
            transparent=True,
            dpi=300,
            bbox_inches="tight",
            format="tiff",
            # pil_kwargs={"compression": "tiff_lzw"},
        )

        plt.show()


# ref: https://github.com/rstar000/super-resolution-resnet/blob/master/upscale.py
def add_padding(img, n_pixels):
    """
    Add a mirrored border to the image
    """
    h, w, c = img.shape
    img_with_border = np.empty((h + 2 * n_pixels, w + 2 * n_pixels, c), np.float32)
    img_with_border[n_pixels : n_pixels + h, n_pixels : n_pixels + w, :] = img
    left_border = img[:, 0:n_pixels, :]
    right_border = img[:, -n_pixels:, :]

    img_with_border[n_pixels : n_pixels + h, 0:n_pixels, :] = left_border[:, ::-1, :]
    img_with_border[n_pixels : n_pixels + h :, -n_pixels:, :] = right_border[:, ::-1, :]

    top_border = img_with_border[n_pixels : 2 * n_pixels, :, :]
    btm_border = img_with_border[-2 * n_pixels - 1 : -n_pixels - 1, :, :]

    img_with_border[0:n_pixels, :, :] = top_border[::-1, :, :]
    img_with_border[-n_pixels:, :, :] = btm_border[::-1, :, :]

    return img_with_border


def remove_padding(img, pad):
    """
    Removes the border
    """
    h, w, c = img.shape

    h -= 2 * pad
    w -= 2 * pad
    return img[pad : pad + h, pad : pad + w, :]


def cal_pad(img):
    """
    Calculate the border size
    max size is 1024x1204 pixels
    """
    h_pad = 0
    w_pad = 0
    h, w, _ = img.shape
    # check if the image is already a power of 2
    if int.bit_count(h) == 1 and int.bit_count(w) == 1:
        return 0

    for i in range(1, 10):
        if 2**i > h:
            h_pad = (2**i - h) // 2
            w_pad = (2**i - w) // 2
            break
    assert h_pad == w_pad

    return h_pad


@torch.no_grad()
def upscale_dem(model, sample, p):
    """
    Upscales the dem.
    """
    smaple_to_tensor = sample.copy()

    pad = cal_pad(sample["lr_dem"])
    # print("Padding size", pad)

    if pad > 0:
        dem = sample["lr_dem"]
        img = sample["image"] if "image" in sample else None
        msk = sample["mask"] if "mask" in sample else None

        dem = add_padding(dem, pad)
        smaple_to_tensor["lr_dem"] = dem
        if "image" in sample:
            img = add_padding(img, pad)
            smaple_to_tensor["image"] = img
        if "mask" in sample:
            msk = add_padding(msk, pad)
            smaple_to_tensor["mask"] = msk

    to_tensor = ToTensor(
        mask_channel=p.mask_channel,
        relative=p.relative,
        **p.tensor_kwargs if p.tensor_kwargs else {},
    )

    sample_to_tensor = to_tensor(smaple_to_tensor)

    dem = sample_to_tensor["lr_dem"].unsqueeze(0).cuda()
    img = sample_to_tensor["image"].unsqueeze(0).cuda() if "image" in sample else None
    msk = sample_to_tensor["mask"].unsqueeze(0).cuda() if "mask" in sample else None

    inputs = []
    if p.model_name.lower() in {"jspsr", "lrru"}:
        if img is None and msk is None:
            inputs = [dem]
        if img is not None and msk is None:
            inputs = [dem, img]
        if img is None and msk is not None:
            inputs = [dem, msk]
        if img is not None and msk is not None:
            inputs = [dem, img, msk]
    else:
        new_dem = torch.zeros(
            (
                dem.shape[0],
                (
                    1  # lr_dem
                    + p.input_data.get("image", 0)
                    + p.input_data.get("mask", 0)
                ),
                dem.shape[2],
                dem.shape[3],
            ),
            dtype=torch.float32,
            device=torch.device("cuda"),
        )

        if img is None and msk is None:
            inputs = [dem]
        if img is not None and msk is None:
            new_dem[:, 0:1, ...] = dem
            new_dem[:, 1:4, ...] = img
            inputs = [new_dem]
        if img is not None and msk is not None:
            new_dem[:, 0:1, ...] = dem
            new_dem[:, 1:4, ...] = img
            new_dem[:, 4:, ...] = msk
            inputs = [dem, img, msk]

    # record inference time and memory
    torch.cuda.reset_peak_memory_stats(device=None)
    t_start = time.time_ns()
    y = model(*inputs)
    torch.cuda.current_stream().synchronize()
    t_infer = (time.time_ns() - t_start) // 1000 / 1000  # unit ms
    m_infer = torch.cuda.max_memory_allocated(device=None) / 1024 / 1024  # unit MB
    # print(t_infer, "ms")
    # print(m_infer, "MB")

    y = y.data[0]
    y = y.cpu().numpy()
    y = y.transpose(1, 2, 0)

    if pad > 0:
        y = remove_padding(y, pad)

    # plt.imshow(
    #     y.squeeze(),
    #     interpolation="none",
    #     cmap="turbo",
    # )
    # plt.show()

    return y, t_infer, m_infer
