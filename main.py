import sys
import random
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo  # python>=3.9
from pathlib import Path
import numpy as np
import torch

import utils.utils
from utils.config import create_config
from utils.common_config import (
    get_dataset,
    get_transformations,
    get_dataloader,
    get_optimizer,
    get_model,
    get_scheduler,
    get_criterion,
)
from utils.logger import Logger
from train.train_utils import EarlyStopper, train_one_epoch
from evaluation.evaluate_utils import eval_model, validate_results, do_eval
from utils.utils import (
    load_resume_state_dict,
    get_time_span,
    get_model_summary,
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Parser
parser = argparse.ArgumentParser(description="Vanilla Training")
parser.add_argument("--config", required=True, help="Config file")
parser.add_argument("--val", action="store_true", help="Validate model weights only")
args = parser.parse_args()

if not args.config:
    raise ValueError("Please provide a config file using --config")

configs = create_config(args.config)
if args.val:
    configs["val_weight"] = True


def main(gpu, p):
    p.start = datetime.now(ZoneInfo("Pacific/Auckland"))
    p.start_string = p.start.strftime("%m%d_%H%M")
    p.result_dir = (Path(p.work_root) / "results" / p.start_string).as_posix()
    Path(p.result_dir).mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(gpu)

    # Random seed to maintain reproducible results
    seed = random.randint(0, 9999)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    p.random_seed = seed

    # https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Retrieve config file
    sys.stdout = Logger(Path(p.result_dir) / "train.log")
    print(f"\nTraining configuration {p.name} result is saving to {p.result_dir}")
    utils.utils.serialize_json(p) if p.verbose else None

    # Tensorboard
    if p.monitor_app == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter((Path(p.result_dir) / "tensorboard").as_posix())

    # TODO: Add wandb support
    # if gpu == 0 and p.monitor_app == "wandb":
    #     import wandb

    #     wandb.login()
    #     if not p.resume:
    #         wandb.init(dir=Path(p.work_root).as_posix(), config=p, project=p.name)

    # Get model
    model = get_model(p)
    model = model.cuda(gpu)

    # Get model summary
    get_model_summary(p, model)

    # Dataset -------------------------------->
    print("Loading dataset...")
    train_transforms, val_transforms = get_transformations(p)
    train_dataset, p.num_train_sample = get_dataset(p, train_transforms, "train")
    val_dataset, p.num_val_sample = get_dataset(p, val_transforms, "valid")
    train_loader = get_dataloader(p, train_dataset, "train")
    val_loader = get_dataloader(p, val_dataset, "valid")
    print("Train transformations:", train_transforms) if p.verbose else None
    print("Valid transformations:", val_transforms) if p.verbose else None
    # <-------------------------------- Dataset

    # Criterion -------------------------------->
    print("Creating loss...")
    criterion = get_criterion(p.loss)
    criterion.cuda(gpu)
    print(criterion) if p.verbose else None
    # <-------------------------------- Criterion

    # Optimizer -------------------------------->
    print("Creating optimizer...")
    optimizer = get_optimizer(p, model)
    print(optimizer) if p.verbose else None
    # <-------------------------------- Optimizer

    # Scheduler -------------------------------->
    print("Creating scheduler...")
    scheduler = get_scheduler(p, optimizer)
    # <-------------------------------- Scheduler

    start_epoch = 0
    # Load weights from checkpoint ------------->
    # Resume training from the epoch specified in checkpoint
    if p.model_kwargs.get("checkpoint") is not None:
        assert Path(
            p.model_kwargs.checkpoint
        ).exists(), f"{p.model_kwargs.checkpoint} not found"
        print(f"Load the model with weight {Path(p.model_kwargs.checkpoint).name}...")
        (
            model,
            start_epoch,
            best_eval_result,
            _,
            _,
        ) = load_resume_state_dict(
            model,
            optimizer,
            scheduler,
            p.model_kwargs.checkpoint,
            p.resume,
        )
    # <------------- Load weights from checkpoint

    # To be commented out if not needed ------------>
    # Evaluate pretrained model, remove if not needed
    if p.get("val_weight") and p.model_kwargs.get("checkpoint") is not None:
        print("Evaluate the pretrained model...")
        eval_model(
            p,
            val_loader,
            criterion,
            model,
            compair_input=True,
            save_prediction=True,
            summarise=True,
        )
        return
    elif p.get("val_weight") and p.model_kwargs.get("checkpoint") is None:
        raise ValueError("Please provide a checkpoint path for weight validation")
    # <------------ commented out if not needed

    # To be commented out if not needed ------------>
    # Evaluate initial model for comparison weight initialization method performance, remove if not needed
    print("Evaluate the initial model...")
    best_eval_result, val_loss = eval_model(
        p,
        val_loader,
        criterion,
        model,
        current_epoch=start_epoch,
        compair_input=True,
    )
    # <------------ commented out if not needed

    # path for saving the best eval result model weights so far
    temp_checkpoint = (
        Path(p.result_dir) / "checkpoints" / Path("_tmp_" + p.model_name + ".pt")
    )
    temp_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # early stopping setup ---------------------------->
    early_stopper = EarlyStopper(
        patience=p.early_stop.get("patience", None),
        min_delta=1e-4,
        monitor=p.early_stop.get("monitor", "val_loss"),
    )
    # <---------------------------- early stopping setup

    # Trainval loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print(f'Starting training at {p.start.strftime("%Y-%m-%d %H:%M:%S")}...')

    for epoch in range(start_epoch, p.epochs):

        # Train one epoch
        train_loss, lr = train_one_epoch(
            gpu,
            p,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            (epoch + 1, p.epochs),  # 1-indexed
        )

        # Evaluate
        # every p.val_interval epochs, or first 1 or last 3 epochs.
        # If p.val_interval is not defined, evaluate very epochs//10
        if do_eval(
            p.epochs,
            epoch,
            start_epoch,
            p.scheduler_kwargs.warmup_epoch,
            p.val_interval,
            p.val_start_epoch,
        ):
            # evaluate the model
            current_eval_result, val_loss = eval_model(
                p, val_loader, criterion, model, current_epoch=epoch + 1
            )

            # tensorboard ----------->
            if p.monitor_app == "tensorboard":
                writer.add_scalar("Learning Rate", lr, epoch + 1)

                tag_loss_dict = {"Train": train_loss, "Val": val_loss}
                writer.add_scalars("Loss", tag_loss_dict, epoch + 1)

                for k, _ in current_eval_result.items():
                    if k == "PSNR":
                        tag_psnr_dict = {"": current_eval_result[k]}
                    if k == "RMSE":
                        tag_rmse_dict = {"": current_eval_result[k]}
                writer.add_scalars("PSNR", tag_psnr_dict, epoch + 1)
                writer.add_scalars("RMSE", tag_rmse_dict, epoch + 1)
            # tensorboard <-----------

            # only save the best model to save time
            is_better, best_eval_result = validate_results(
                current_eval_result, best_eval_result, p.best_metric
            )
            if is_better:
                # Temporary checkpoint
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "state_dict": model.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "best_result": best_eval_result,
                    },
                    temp_checkpoint,
                )
            if epoch > 200 and early_stopper(val_loss, train_loss, current_eval_result):
                print("Early stopped!!!")
                break

    # Trainval loop <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # post-process after training
    end = datetime.now(ZoneInfo("Pacific/Auckland"))
    d, h, m, s = get_time_span(p.start, end)
    if d != 0:
        print(
            f'\n### Finished training at {end.strftime("%Y-%m-%d %H:%M:%S")}, total time: {d}d, {h}h,{m}m,{s}s.'
        )
    else:
        print(
            f'\n### Finished training at {end.strftime("%Y-%m-%d %H:%M:%S")}, total time: {h}h,{m}m,{s}s.'
        )

    # Evaluate the best model at the end
    input_data = p.input_data.copy()
    input_data.pop(
        "lr_dem"
    )  # "lr_dem" is always used, so we don't include it in the filename
    final_checkpoint = (
        Path(p.result_dir)
        / "checkpoints"
        / Path(
            p.model_name
            + "_r"
            + str(p.resolution)
            + "_"
            + "_".join(input_data.keys())
            + f"_{'_'.join([f'{k}{v:.2f}' for k, v in best_eval_result.items() if k in {'RMSE', 'PSNR'}])}.pt"
        )
    )
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    Path(temp_checkpoint).rename(final_checkpoint)

    print(f"\nEvaluating the best model of {p.model_name}")
    model, best_epoch, _, _, _ = load_resume_state_dict(
        model,
        optimizer,
        scheduler,
        final_checkpoint,
        resume=True,  # to print the best epoch, not really resuming
    )
    eval_model(
        p,
        val_loader,
        criterion,
        model,
        current_epoch=best_epoch,
        compair_input=False,
        save_prediction=True,
        summarise=True,
    )


if __name__ == "__main__":
    main(configs.get("gpu", 0), configs)
