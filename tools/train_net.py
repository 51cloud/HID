#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import json

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule
from slowfast.models import head_helper, resnet_helper, stem_helper
from slowfast.models.losses import MemoryMoCo, NCESoftmaxLoss, TripletLoss
import torchsnooper
from sklearn.metrics import precision_recall_fscore_support
logger = logging.get_logger(__name__)

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}

# @torchsnooper.snoop()
def train_epoch(
        train_loader, model, optimizer, train_meter, cur_epoch, cfg,
        writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.

    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    # data_preds = []
    precisions = []
    recalls = []
    f1s = []
    max_indices_array = []
    labels_array = []

    for cur_iter, (person_infos, videos, labels, _, meta) in enumerate(train_loader):
        
        device_id = 0
        device = torch.cuda.get_device_properties(device_id)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        tymode = 'train'
        videos = videos[0]
        outputs, person_ids = model(person_infos, videos, tymode)
        #normal:0, before:1, thefting:2
        before = False
        theft = False
        normal = True
        preds = torch.empty(1, 3)
        normal_pred = torch.empty(1, 3)
        before_pred = torch.empty(1, 3)
        theft_pred = torch.empty(1, 3)
        for i in range(len(outputs)):
            person_pred = outputs[i]
            normal_pred = person_pred
            max_index = torch.argmax(person_pred, dim=1)
            if (max_index == 1) and (theft==False):
                before = True
                before_pred = person_pred
                normal = False
            if max_index == 2:
                theft = True
                theft_pred = person_pred
                before = False
                normal = False
        if normal:
            preds = normal_pred
        if before:
            preds = before_pred
        if theft:
            preds = theft_pred
        inputs = outputs
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        labels = labels.cuda()
        all_loss = loss_fun(preds, labels)
        misc.check_nan_losses(all_loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        all_loss.backward()
        # Update the parameters.
        optimizer.step()

        # all_loss = all_loss.cpu().detach().numpy()

        top1_err, top3_err = None, None
        if cfg.DATA.MULTI_LABEL:
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                [all_loss] = du.all_reduce([all_loss])
            all_loss = all_loss.item()
        else:
            # Compute the errors.
            # print('train--preds', preds)
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 3))
            top1_err, top3_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                all_loss, top1_err, top3_err = du.all_reduce(
                    [all_loss, top1_err, top3_err]
                )

            # Copy the stats from GPU to CPU (sync point).
            all_loss, top1_err, top3_err = (
                all_loss.item(),
                top1_err.item(),
                top3_err.item(),
            )

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top3_err,
            all_loss,
            lr,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": all_loss,
                    "Train/lr": lr,
                    "Train/Top1": (1 - top1_err),
                    "Train/Top3": (1 - top3_err),
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

        max_indices = torch.argmax(preds, dim=1)
        max_indices_id = int(max_indices)
        labels_id = int(labels)

        max_indices_array.append(max_indices_id)
        labels_array.append(labels_id)
        
    # Log epoch stats.

    precision, recall, f1, _ = precision_recall_fscore_support(labels_array, max_indices_array, average='weighted')

    print('precision, recall, f1: ', precision, recall, f1)

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (person_infos, videos, labels, _, meta) in enumerate(val_loader):
        videos = videos[0]
        tymode = 'val'
        outputs, _ = model(person_infos, videos, tymode)
        before = False
        theft = False
        normal = True
        preds = torch.empty(1, 3)
        normal_pred = torch.empty(1, 3)
        before_pred = torch.empty(1, 3)
        theft_pred = torch.empty(1, 3)
        for i in range(len(outputs)):
            person_pred = outputs[i]
            normal_pred = person_pred
            max_index = torch.argmax(person_pred, dim=1)
            if (max_index == 1) and (theft==False):
                before = True
                before_pred = person_pred
                normal = False
            if max_index == 2:
                theft = True
                theft_pred = person_pred
                before = False
                normal = False
        if normal:
            preds = normal_pred
        if before:
            preds = before_pred
        if theft:
            preds = theft_pred
        inputs = outputs
        labels = labels.cuda()
        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
        else:
            # Compute the errors.
            # print('val--preds', preds)
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 3))

            # Combine the errors across the GPUs.
            top1_err, top3_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top3_err = du.all_reduce([top1_err, top3_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top3_err = top1_err.item(), top3_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top3_err,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1": (1 - top1_err), "Val/Top3": (1-top3_err)},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

        val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        tymode = ''
        for inputs, _, _, _ in loader:
            inputs1 = [inputs[0], ]
            if use_gpu:
                if isinstance(inputs1, (list,)):
                    for i in range(len(inputs1)):
                        inputs1[i] = inputs1[i].cuda(non_blocking=True)
                else:
                    inputs1 = inputs1.cuda(non_blocking=True)
            inputs = inputs1
        inputs1 = [inputs, tymode]
        yield inputs1

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)

    device = torch.device('cuda')
    model.to(device)

    # model = nn.DataParallel(model)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")

    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg,
            writer
        )

        # Compute precise BN stats.
        # if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
        #     calculate_and_update_precise_bn(
        #         precise_bn_loader,
        #         model,
        #         min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
        #         cfg.NUM_GPUS > 0,
        #     )
        # _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(
                cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        ):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(
                cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        ):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
