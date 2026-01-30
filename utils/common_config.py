import torch
from torch.utils.data import DataLoader
from data import data_utils as tr
from torchvision import transforms as v1
from utils.utils import load_model_from_url, get_diff_params
from losses.loss_schemes import get_loss
from easydict import EasyDict as edict

# from torchvision.transforms import v2  # v2 is not stable yet, even though it has some nice features

"""
Model getters 
"""


def get_model(p):
    """Return the model"""
    model = None

    if p.model_name.lower() == "edsr":
        from models.EDSR import EDSR

        if "dfc" in p.dataset.lower():
            in_channels = (
                1  # lr_dem
                + p.input_data.get("image", 0)
                + p.input_data.get("mask", 0)
                + p.input_data.get("canopy", 0)
                + p.input_data.get("coord", 0)
            )
            model = EDSR(
                in_channels=in_channels,
                out_channels=1,
                n_resblocks=p.model_kwargs.num_block,
                n_features=p.model_kwargs.num_feature,
                scale=1,
                spn=p.model_kwargs.spn,  # experiment with adding spn
            )

    if p.model_name.lower() in {"completionformer"}:
        from models.CompletionFormer import Model

        if "dfc" in p.dataset.lower():
            args = edict()
            args.input_channels = p.input_data
            args.output_channels = 1
            args.prop_time = 6
            args.prop_kernel = 3
            args.preserve_input = False
            args.conf_prop = True
            args.affinity = "TGASS"
            args.affinity_gamma = 0.5
            args.legacy = False

            model = Model(args)

    if p.model_name.lower() in {"lrru"}:
        from models.LRRU import Model

        if "dfc" in p.dataset.lower():
            args = edict()
            args.input_channels = p.input_data
            args.output_channels = 1
            args.kernel_size = 3
            args.bc = 16
            args.prob = 1.0
            args.dkn_residual = True

            model = Model(args)

    if p.model_name.lower() == "jspsr":
        from models.JSPSR import Model

        if "dfc" in p.dataset.lower():
            model = Model(
                in_channels=p.input_data,
                out_channels=1,
                num_feature=p.model_kwargs.num_feature,
                layers=(
                    p.model_kwargs.num_block,
                    p.model_kwargs.num_block,
                    p.model_kwargs.num_block,
                    p.model_kwargs.num_block,
                ),
                spn=p.model_kwargs.spn,
                spn_scale=p.model_kwargs.get("spn_scale", 1.0),
            )

    else:
        raise NotImplementedError(f"Unsupported model name {p.model_name}")

    name_string = model.name if hasattr(model, "name") else p.model.__class__.__name__
    scale_string = (
        f"for X{p.scale} scale..."
        if p.scale is not None
        else f"for {p.resolution}m resolution..."
    )
    print(f"Creating model {name_string} {scale_string}")

    if p.model_kwargs.pretrained:
        model = load_model_from_url(model.url, model)
        print(f"Loaded pretrained model {model.url}") if p.verbose else None

    return model


"""
    Transformations, datasets and dataloaders
"""


def get_transformations(p):
    """Return transformations for training and evaluationg"""

    # Train and valid/test transformations, not for inference due to cropping
    transform_list_ts = []

    if not p.get("crop_mode") or p.crop_mode.lower() == "random":
        if "dfc" in p.dataset.lower():
            transform_list_ts.append(tr.RandomCrop(p.patch_size))
        else:
            transform_list_ts.append(tr.RandomCrop(p.patch_size, p.scale))
    elif p.crop_mode.lower() == "tile":
        if "dfc" in p.dataset.lower():
            transform_list_ts.append(
                tr.TileCrop(p.patch_size, n_tile=p.patches_per_image)
            )
        else:
            transform_list_ts.append(
                tr.TileCrop(p.patch_size, scale=p.scale, n_tile=p.patches_per_image)
            )

    transform_list_ts.append(
        tr.ToTensor(
            p.get("normalize", None),
            p.get("mask_channel", None),
            p.get("relative", False),
            **(p.tensor_kwargs if p.tensor_kwargs else {}),
        )
    )
    transform_ts = v1.Compose(transform_list_ts)

    transform_list_tr = transform_list_ts.copy()
    transform_list_tr.insert(1, tr.RandomFlipRotate90()) if p.augment else None

    (
        (
            transform_list_tr.insert(
                1,
                tr.Normalize(normalize_list=p.normalize, resolution=p.resolution),
            )
            if "dfc" in p.dataset.lower()
            else transform_list_tr.insert(1, tr.Normalize(normalize_list=p.normalize))
        )
        if p.normalize
        else None
    )

    transform_tr = v1.Compose(transform_list_tr)

    return transform_tr, transform_ts


def get_dataset(p, transforms, split="train"):
    """Return the train dataset"""
    dataset = None

    print(f"Preparing {split} loader for dataset: {p.dataset}")

    if "dfc30" in p.dataset.lower():
        from data.dfc30 import DFC30

        dataset = DFC30(split=split, transform=transforms, **p)
    else:
        raise NotImplementedError(f"Unsupported dataset: {p.dataset}")

    assert dataset is not None, f"Dataset {p.dataset} not found!"

    return dataset, len(dataset)


def get_dataloader(p, dataset, split="train", sampler=None):
    """Return the train/valid dataloader and total number of samples"""
    batch_size = p.train_batch_size if split == "train" else p.valid_batch_size
    shuffle = True if split == "train" and sampler is None else False
    workers = p.workers if split == "train" else 0
    drop_last = True if split == "train" else False
    pin_memory = True if split == "train" else False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn,
        sampler=sampler,
    )

    return dataloader


""" 
    Loss functions 
"""


def get_criterion(kwargs):
    """
    get list of losses
    """
    config = kwargs.copy()

    if len(config.keys()) == 1:
        from losses.loss_schemes import SingleLoss

        loss, _ = config.popitem()
        loss_dict = {loss: {"loss_fn": get_loss(loss), "weight": 1}}
        criterion = SingleLoss(**loss_dict)

    else:
        from losses.loss_schemes import MultiLoss

        loss_dict = {}
        for loss, weight in config.items():
            loss_dict[loss] = {}
            loss_dict[loss]["loss_fn"] = get_loss(loss)
            loss_dict[loss]["weight"] = weight

        criterion = MultiLoss(**loss_dict)

    return criterion


"""
    Optimizers and schedulers
"""


def get_optimizer(p, model):
    """Return optimizer for a given model and setup"""
    lr = p.optimizer_kwargs.lr
    momentum = p.optimizer_kwargs.momentum
    weight_decay = p.optimizer_kwargs.weight_decay

    # set different learning rate for different parts of the model
    params = []
    if p.optimizer_kwargs.diff_lr:  # experiment with different learning rates
        if "jspsr" in p.model_name.lower():
            base_params, diff_params = get_diff_params(model, "postprocessor")
            params.append({"params": [param[1] for param in base_params]})
            params.append({"params": [param[1] for param in diff_params], "lr": 0.0003})
        else:
            raise NotImplementedError(
                f"Undefined model parts for different learning rates: {p.model_name}"
            )

    # diff_lr is False, or compiled model
    if not params:
        params = model.parameters()

    if p.optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif p.optimizer.lower() == "adam":
        opt = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif p.optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif p.optimizer.lower() == "rmsprop":
        opt = torch.optim.RMSprop(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError(f"Undefined optimizer: {p.optimizer}")
    return opt


def get_scheduler(p, optimizer):
    """Return scheduler for a given optimizer and setup"""
    warmup_epoch = p.scheduler_kwargs.get("warmup_epoch", 0)
    max_lr = p.scheduler_kwargs.get("max_lr", 0.1)
    T_max = p.epochs
    step_size = (
        p.scheduler_kwargs.get("step_size")
        if p.scheduler_kwargs.get("step_size") is not None
        else p.epochs // 3
    )
    gamma = (
        p.scheduler_kwargs.get("gamma")
        if p.scheduler_kwargs.get("gamma") is not None
        else 0.1
    )

    if p.scheduler.lower() == "onecyclelr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=p.epochs,
            div_factor=90,
        )
        if p.verbose:
            print(
                f"Init scheduler OneCycleLR max_lr: {max_lr}, total_steps: {p.epochs}"
            )

    elif p.scheduler.lower() == "cosineannealinglr":
        # cosine learning rate if T_max equal to total epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=1e-6
        )
        if p.verbose:
            print(f"Init scheduler CosineAnnealingLR T_max: {T_max}")

    elif p.scheduler.lower() == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        if p.verbose:
            print(f"Init scheduler StepLR step_size: {step_size}, gamma: {gamma}")

    elif p.scheduler.lower() == "warmupsteplr":
        train_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

        def warmup(current_step: int):
            # 5 epochs warmup: 0.001, 0.01, 0.1
            return 1 / (10 ** float(warmup_epoch - current_step))

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, train_scheduler], [warmup_epoch]
        )
        if p.verbose:
            print(f"Init scheduler WarmpuStepLR step_size: {step_size}, gamma: {gamma}")

    elif p.scheduler.lower() == "constantlr":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)
        if p.verbose:
            print("Init scheduler ConstantLR")

    else:
        raise NotImplementedError(f"Undefined scheduler: {p.scheduler}")

    return scheduler
