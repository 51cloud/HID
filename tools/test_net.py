#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
import torch.nn as nn

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
from sklearn.metrics import precision_recall_fscore_support

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    precisions = []
    recalls = []
    f1s = []
    labels_array = []
    max_indices_array = []
    for cur_iter, (person_infos, videos, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            
            labels = labels.cuda()
            

            # Perform the forward pass.
            videos = videos[0]
            tymode = 'test'           
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
                
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)

            max_indices = torch.argmax(preds, dim=1)
            # max_indices_array = max_indices.numpy()
            # labels_array = labels.numpy()
            max_indices_id = int(max_indices)
            labels_id = int(labels)

            labels_array.append(max_indices_id)
            max_indices_array.append(labels_id)

        test_meter.iter_tic()
    precision, recall, f1, _ = precision_recall_fscore_support(labels_array, max_indices_array, average='weighted')

    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            with PathManager.open(save_path, "wb") as f:
                pickle.dump([all_labels, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics()
    test_meter.reset()

def similar_score(feature1, feature2, weight):
    # print(feature1[0].shape,feature2[0].shape,)
    dist = torch.dist(weight*feature1[0], weight*feature2[0])
    similarity = 1 / (1 + dist)
    return similarity

class AttentionSimilarity(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionSimilarity, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention = nn.Linear(input_size, hidden_size, bias=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x1, x2, preds):
        # 计算注意力得分
        # print('x1.shape:', x1.shape) # (1,2048,8,7,7)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        score = torch.matmul(x1.cuda(), preds.transpose(1, 0))
        # 计算注意力权重
        attention_weights_x1 = torch.softmax(score, dim=0)
        attention_weights_x2 = torch.softmax(score, dim=1)
        # 计算加权后的特征
        attended_x1 = torch.sum(attention_weights_x2 * x2, dim=0)
        attended_x2 = torch.sum(attention_weights_x1 * x1, dim=0)
        # 计算相似度得分
        similarity = self.out(torch.cat((attended_x1, attended_x2), dim=0))
        return similarity

def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
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

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    # print('cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS:',cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
                len(test_loader.dataset)
                % 1
                == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            len(test_loader.dataset)
            // 1,
            1,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
