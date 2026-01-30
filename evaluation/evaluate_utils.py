import math
from pathlib import Path
from natsort import natsorted
import numpy as np
import torch
import torch.nn.functional as F
from data.data_utils import ToDEM
from utils.utils import (
    get_loss_monitor,
    get_batch_pair,
    display_predictions,
    summarise_evaluation,
)
from evaluation.metrics import (
    MeterPSNR,
    MeterSSIM,
    MeterRMSE,
    MeterMedian,
    MeterNMAD,
    MeterLE95,
    MeterSlope,
)
import rasterio


class PerformanceMeter(object):
    """A general performance meter which shows performance"""

    def __init__(self, metrics_kwargs):
        self.metrics = [m for m in metrics_kwargs.keys()]
        self.meters = {m: get_meter(m, metrics_kwargs[m]) for m in self.metrics}

    def reset(self):
        for m in self.metrics:
            self.meters[m].reset()

    def update(self, pred, gt, meta=None, base_elev=0, elev_log=False):
        for m in self.metrics:
            self.meters[m].update(
                pred, gt, meta, base_elev=base_elev, elev_log=elev_log
            )

    def get_score(self):
        eval_dict = {}
        for m in self.metrics:
            eval_dict[m] = self.meters[m].get_score()
        return eval_dict


def get_meter(metric, kwargs):
    """Retrieve a meter to measure the performance"""
    package = kwargs.get("package") if kwargs.get("package") is not None else "local"
    psnr_type = (
        kwargs.get("psnr_type") if kwargs.get("psnr_type") is not None else "rgb"
    )
    tensor_range = (
        kwargs.get("tensor_range")
        if kwargs.get("tensor_range") is not None
        else "[0, 1]"
    )
    border = kwargs.get("border") if kwargs.get("border") is not None else 0

    value_min = kwargs.get("min") if kwargs.get("min") is not None else 0
    value_max = kwargs.get("max") if kwargs.get("max") is not None else 1

    if metric.lower() == "psnr":
        return MeterPSNR(
            package=package,
            psnr_type=psnr_type,
            tensor_range=tensor_range,
            border=border,
            value_min=value_min,
            value_max=value_max,
        )
    elif metric.lower() == "ssim":
        return MeterSSIM(package=package, tensor_range=tensor_range, border=border)
    elif metric.lower() == "rmse":
        return MeterRMSE(
            package=package,
            tensor_range=tensor_range,
            border=border,
            value_min=value_min,
            value_max=value_max,
        )
    elif metric.lower() == "median":
        return MeterMedian(
            package=package,
            tensor_range=tensor_range,
            border=border,
            value_min=value_min,
            value_max=value_max,
        )
    elif metric.lower() == "nmad":
        return MeterNMAD(
            package=package,
            tensor_range=tensor_range,
            border=border,
            value_min=value_min,
            value_max=value_max,
        )
    elif metric.lower() == "le95":
        return MeterLE95(
            package=package,
            tensor_range=tensor_range,
            border=border,
            value_min=value_min,
            value_max=value_max,
        )
    elif metric.lower() == "slope":
        return MeterSlope(
            package=package,
            tensor_range=tensor_range,
            border=border,
            value_min=value_min,
            value_max=value_max,
        )
    else:
        raise NotImplementedError


def validate_results(current, reference, best_metric=None):
    """
    Compare the results between the current eval dict and a reference eval dict.
    Returns a tuple (boolean, eval_dict).
    The boolean is true if the current eval dict has higher performance compared
    to the reference eval dict.
    The returned eval dict is the one with the highest performance.
    """
    assert set(current.keys()) == set(
        reference.keys()
    ), f"{current.keys()}, {reference.keys()}"

    if isinstance(best_metric, str):
        best_metric = [best_metric]

    if not best_metric or all([x not in list(current.keys()) for x in best_metric]):
        keys = list(current.keys())
    else:
        keys = best_metric

    comparison = []
    for k in keys:
        if k.lower() in {"rmse"}:
            comparison.append(current[k] < reference[k] or reference[k] == 0)
        if k.lower() in {"psnr", "ssim"}:
            comparison.append(current[k] > reference[k] or reference[k] == 0)

    if all(comparison):
        return True, current
    else:
        return False, reference


def get_visual_id(num_visual, num_sample, batch_size, id_visual=None):
    if num_visual < 0:
        _list = list(range(num_sample))
    elif num_visual == 0:
        return []
    else:
        _list = list(np.random.choice(np.arange(num_sample), size=num_visual))

    if id_visual is not None:
        if num_visual == 1:
            _list = [id_visual]
        else:
            _list.pop()
            _list.append(id_visual)
            _list = [*{*_list}]  # remove duplicate

    _list = natsorted(_list)

    batch_list = [i // batch_size for i in _list]
    item_list = [i % batch_size for i in _list]

    return list(zip(batch_list, item_list))


def get_sample_from_batch(batch, n):
    """
    Get a sample from a batch
    :param batch: batch data
    :param n: index of the sample in the batch
    """
    sample = {}
    for k, v in batch.items():
        sample[k] = v[n]
    return sample


def display_predictions_one_epoch(
    p,
    plt_list,
    epoch_id,
    batch_id,
    batch,
    pred,
):
    if isinstance(plt_list, list) and len(plt_list) > 0:
        to_plt = [n for m, n in plt_list if m == batch_id]
        if pred.requires_grad:
            pred = pred.detach()
        for n in to_plt:
            # print(i, n)
            sample = get_sample_from_batch(batch, n)
            display_predictions(p, sample, pred[n], current_epoch=epoch_id)
            if plt_list[0][0] == batch_id and plt_list[0][1] == n:
                plt_list.pop(0)
    return plt_list


def do_eval(
    epochs,
    current_epoch,
    start_epochs,
    warmup_epochs,
    val_interval,
    val_start_epoch=1,
):
    """Return True if it is time to perform evaluation"""
    if val_interval is None:
        val_interval = epochs // 10

    # Always validate the last 3 epochs
    if current_epoch + 1 >= epochs - 3:
        return True

    # Always validate the first 1 epochs after warmup
    if (
        start_epochs + warmup_epochs
        < current_epoch + 1
        <= start_epochs + warmup_epochs + 1
    ):
        return True

    # Validate every val_interval epochs if current epoch larger than val_start_epoch
    if current_epoch + 1 >= val_start_epoch and (current_epoch + 1) % val_interval == 0:
        return True

    return False


def save_prediction_to_disk(p, meta, pred):
    """Save the prediction to disk"""
    name_list = [m["id"] for m in meta]
    subset_list = [m["subset"] for m in meta]
    prof_list = [m["profile"] for m in meta]
    base_list = [m["base"] for m in meta]

    save_dir = Path(p.result_dir) / "sr_dem"

    for name, subset, prof, base, arr in zip(
        name_list, subset_list, prof_list, base_list, pred
    ):
        save_path = save_dir / subset / Path(name + ".tif")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        assert pred.dim() == 4, f"Prediction should have 4 dimensions, got {pred.dim()}"
        pred = pred.cpu().numpy().squeeze().squeeze()
        pred = np.clip(pred, 0, 1)
        pred = (
            ToDEM.descale_data(
                pred,
                p.tensor_kwargs.min,
                p.tensor_kwargs.max,
                p.tensor_kwargs.log,
            )
            + base
        )
        with rasterio.open(save_path.as_posix(), "w", **prof) as dst:
            dst.write(pred, 1)

    return save_dir


@torch.no_grad()
def eval_model(
    p,
    val_loader,
    criterion,
    model,
    current_epoch=0,
    compair_input=False,
    save_prediction=False,
    summarise=False,
):
    """Evaluate model on the validation set"""
    plt_list = get_visual_id(
        p.val_num_visual,
        p.num_val_sample,
        p.valid_batch_size,
        id_visual=p.val_id_visual,
    )
    save_dir = None
    # print(plt_list)
    performance_meter = PerformanceMeter(p.metric)
    performance_input = PerformanceMeter(p.metric) if compair_input else None
    loss_monitor = get_loss_monitor(p.loss)

    model.eval()
    for i, batch in enumerate(val_loader):
        criterion.reset()

        # Forward pass
        inputs, gt, base_elev, meta = get_batch_pair(batch, p.model_name, p.input_data)

        pred = model(*inputs)

        # Save prediction to disk
        if save_prediction:
            save_dir = save_prediction_to_disk(p, meta, pred)

        # Measure performance and loss
        loss_dict = criterion(pred, gt)
        for k, v in loss_dict.items():
            loss_monitor[k].update(v.item(), gt.size(0))
        performance_meter.update(
            pred,
            gt,
            meta=meta,
            base_elev=base_elev,
            elev_log=p.tensor_kwargs.log,
        )

        # Measure performance of Bicubic
        if compair_input:
            if p.get("input_data") is None:
                data_input = inputs[0][:, 0:3, :, :]
            elif "COP30" in p.input_data or "FABDEM" in p.input_data:
                data_input = inputs[0][:, 0:1, :, :]
            else:
                raise NotImplementedError
            if data_input.size(2) != gt.size(2):
                data_input = F.interpolate(data_input, size=gt.size(2), mode="bicubic")
            performance_input.update(
                data_input,
                gt,
                meta=meta,
                base_elev=base_elev,
                elev_log=p.tensor_kwargs.log,
            )

        # Visualize predictions
        plt_list = display_predictions_one_epoch(
            p, plt_list, current_epoch, i, batch, pred
        )

    if compair_input:
        print(f"E{current_epoch} Bicubic score:")
        performance_input.get_score()
    print(f"E{current_epoch} Prediction score:")
    eval_results = performance_meter.get_score()

    # calculate and plot evaluation summary
    if summarise:
        summarise_evaluation(p, save_dir, online=True)

    # print(f"Total loss\t\t\t{loss_monitor['Total'].avg:5.3e}")
    return eval_results, loss_monitor["Total"].avg
