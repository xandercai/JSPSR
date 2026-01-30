import torch
from sympy.logic.algorithms.dpll import pl_true_int_repr
from tqdm import tqdm
from collections import OrderedDict
from evaluation.evaluate_utils import (
    get_visual_id,
    display_predictions_one_epoch,
)
from utils.utils import get_loss_monitor, get_batch_pair


class EarlyStopper:
    def __init__(
        self,
        patience,
        min_delta=1e-6,
        monitor="val_loss",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.min_val_loss = float("inf")
        self.val_loss = None
        self.train_loss = None

    def __call__(self, val_loss=None, train_loss=None, eval_result=None):
        # ignore if patience is not defined
        if self.patience is None:
            return False

        if self.monitor == "val_loss":
            self.val_loss = val_loss
        elif self.monitor == "trainval_loss":
            self.val_loss = val_loss
            self.train_loss = train_loss
        elif (
            self.monitor == "val_psnr"
            and isinstance(eval_result, dict)
            and eval_result.get("PSNR")
        ):
            self.val_loss = eval_result["PSNR"]
        elif (
            self.monitor == "val_ssim"
            and isinstance(eval_result, dict)
            and eval_result.get("SSIM")
        ):
            self.val_loss = eval_result["PSNR"]
        elif (
            self.monitor == "val_rmse"
            and isinstance(eval_result, dict)
            and eval_result.get("RMSE")
        ):
            self.val_loss = eval_result["RMSE"]
        else:
            raise NotImplementedError

        # ignore if val_loss is not computed
        if self.val_loss is None:
            return False

        if self.monitor == "trainval_loss":
            assert self.train_loss is not None, "train_loss must be provided"
            if val_loss > train_loss + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            else:
                self.counter = 0
        elif self.monitor in {"val_loss", "val_psnr", "val_ssim", "val_rmse"}:
            if val_loss > self.min_val_loss + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            else:
                self.min_val_loss = val_loss
                self.counter = 0
        else:
            raise NotImplementedError

        return False


def get_tensor_range(tensor, tensor_range=None):
    """
    Get the range of input tensor, and update the tensor_range if it is provided.
    tensor: torch.Tensor
    tensor_range: [min, max]
    """
    output_min, output_max = torch.min(tensor), torch.max(tensor)
    if tensor_range is not None:
        tensor_range[0] = min(output_min, tensor_range[0])
        tensor_range[1] = max(output_max, tensor_range[1])
    else:
        tensor_range = [output_min, output_max]
    return tensor_range


def get_tensor_mean(tensor_list):
    """
    Get the mean value of the tensors
    """
    tensor_list = tensor_list if isinstance(tensor_list, list) else [tensor_list]

    mean_list = []
    for tensor in tensor_list:
        mean_list.append(torch.mean(tensor).item())

    return mean_list


def get_tensor_real(tensor_list):
    """
    Get the sigmoid value zero/one count of the tensors
    """
    tensor_list = tensor_list if isinstance(tensor_list, list) else [tensor_list]

    one_list = []
    for tensor in tensor_list:
        # one = tensor.sigmoid().gt(0.5).sum().item()
        one = tensor.gt(0).sum().item()
        one_list.append(one)

    return one_list


@torch.no_grad()
def get_gradient_range(model, param_range=None):
    """
    Get the range of gradient in the model
    """
    param_min = 999
    param_max = -999
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_min = min(param_min, param.grad.min())
            param_max = max(param_max, param.grad.max())
    if param_range is not None:
        param_range[0] = min(param_min, param_range[0])
        param_range[1] = max(param_max, param_range[1])
    else:
        param_range = [param_min, param_max]
    return param_range


# @torch.no_grad()
# def clip_gradient(optimizer, grad_clip):
#     """
#     Clips gradients computed during backpropagation to avoid explosion of gradients.
#
#     :param optimizer: optimizer with the gradients to be clipped
#     :param grad_clip: clip value
#     """
#     for group in optimizer.param_groups:
#         for param in group["params"]:
#             if param.grad is not None:
#                 param.grad.data.clamp_(-grad_clip, grad_clip)


def train_one_epoch(
    gpu,
    p,
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    epoch,
):
    """Training loop for one epoch"""
    if isinstance(epoch, tuple):
        current_epoch, epochs = epoch
    elif isinstance(epoch, int):
        current_epoch = epoch
        epochs = "Unknown"
    else:
        raise NotImplementedError

    # setup monitors
    loss_monitor = get_loss_monitor(p.loss)
    # performance_meter = PerformanceMeter(p.metric)
    # for recording the last range of input, gt, pred, and gradinat
    # range_input = [1, -1]
    # range_gt = [1, -1]
    # range_pred = [1, -1]
    # range_grad = [1, -1]

    model.train()

    plt_list = None
    if p.train_num_visual:
        plt_list = get_visual_id(
            p.train_num_visual, p.num_train_sample, p.train_batch_size
        )

    pbar = tqdm(
        total=len(train_loader),
        unit="batch",
        ncols=220,
        position=gpu,
        desc=f"E{current_epoch:03d}/{epochs:d}",
        bar_format="{desc:<9}{percentage:3.0f}%|{bar:10}{r_bar}",
    )

    for i, batch in enumerate(train_loader):
        criterion.reset()

        inputs, gt, _, _ = get_batch_pair(batch, p.model_name, p.input_data, gpu)

        model.zero_grad(set_to_none=True)
        pred = model(*inputs)

        # Measure loss and performance
        loss_dict = criterion(pred, gt)

        # Backward
        loss_dict["Total"].backward()

        optimizer.step()

        # Visualize predictions for observation if needed
        if p.train_num_visual:
            display_predictions_one_epoch(p, plt_list, current_epoch, i, batch, pred)

        for k, v in loss_dict.items():
            loss_monitor[k].update(v.item(), gt.size(0))
            # loss_string += f"{k}:{loss_monitor[k].avg:5.3e} "
        loss_string = f"{loss_monitor['Total'].avg:5.3e}"
        # performance_meter.update(pred, gt)

        lr = float(optimizer.param_groups[0]["lr"])
        if p.optimizer_kwargs.diff_lr:
            lr_string = "{:4.2e}, {:4.2e}".format(
                lr,
                float(optimizer.param_groups[-1]["lr"]),
            )
        else:
            lr_string = "{:4.2e}".format(lr)

        # Update progress bar
        postfix = OrderedDict(
            loss=loss_string,
            lr=lr_string,
        )
        if p.monitor_value:
            if "grad" in p.monitor_value:
                # range_grad = get_gradient_range(model, range_param)
                range_grad = get_gradient_range(model)
                postfix.update(
                    grad="({:6.4f} {:6.4f})".format(range_grad[0], range_grad[1]),
                )
            if "input" in p.monitor_value:
                # range_input = get_tensor_range(torch.stack(inputs), range_input)
                # range_gt = get_tensor_range(gt, range_gt)
                range_input = get_tensor_range(inputs[0])
                range_gt = get_tensor_range(gt)
                postfix.update(
                    input="({:6.4f} {:6.4f})".format(range_input[0], range_input[1]),
                    gt="({:6.4f} {:6.4f})".format(range_gt[0], range_gt[1]),
                )
            if "pred" in p.monitor_value:
                # range_pred = get_tensor_range(pred.detach(), range_pred)
                range_pred = get_tensor_range(pred.detach())
                postfix.update(
                    pred="({:6.4f} {:6.4f})".format(range_pred[0], range_pred[1])
                )
        pbar.set_postfix(postfix)
        pbar.update(1)

    scheduler.step()

    pbar.close()
    # eval_results = performance_meter.get_score()
    del batch, inputs, gt, pred
    # return eval_results, loss_monitor["Total"].avg
    return loss_monitor["Total"].avg, lr
