#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""
import os

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from PIL import Image
# from scipy.misc import imresize
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm
from slowfast.utils.ImageMISC import imresize

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY
from torch.nn import functional as F

from emotion.face_detector.face_detector import DnnDetector, HaarCascadeDetector
from emotion.model.model import Mini_Xception, Change_Size
from emotion.utils import get_label_emotion, normalization, histogram_equalization, standerlization
from emotion.face_alignment.face_alignment import FaceAlignment
import torchvision.transforms.transforms as transforms

from Gaze.model import ModelSpatial, Change_Gaze_Size, Change_Attention_Size, Change_Scene_Size
from Gaze.utils import imutils, evaluation
from Gaze.config import *
import xml.etree.ElementTree as ET

from Graph.model import Build_Graph

import h5py
import gc
import numpy as np
# 假设 additional_attributes 是一个默认的嵌套字典
from collections import defaultdict

# from retinaface import RetinaFace

import torchsnooper

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
            self,
            dim_in,
            fusion_conv_channel_ratio,
            fusion_kernel,
            alpha,
            eps=1e-5,
            bn_mmt=0.1,
            inplace_relu=True,
            norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
                cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                # dim_in=[
                #     width_per_group * 32,
                #     width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                # ],
                dim_in = [1,1],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, x2, tymodel, bboxes=None):
        if tymodel == 'train':
            # torch.Size([1, 3, 8, 224, 224])
            # torch.Size([1, 3, 32, 224, 224])
            x = self.s1(x)
            x2 = self.s1(x2)
            # torch.Size([1, 64, 8, 56, 56])
            # torch.Size([1, 8, 32, 56, 56])
            x = self.s1_fuse(x)
            x2 = self.s1_fuse(x2)
            # torch.Size([1, 80, 8, 56, 56])
            # torch.Size([1, 8, 32, 56, 56])
            x = self.s2(x)
            x2 = self.s2(x2)
            # torch.Size([1, 256, 8, 56, 56])
            # torch.Size([1, 32, 32, 56, 56])
            x = self.s2_fuse(x)
            x2 = self.s2_fuse(x2)
            # torch.Size([1, 320, 8, 56, 56])
            # torch.Size([1, 32, 32, 56, 56])
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x2[pathway] = pool(x2[pathway])
            # torch.Size([1, 320, 8, 56, 56])
            # torch.Size([1, 32, 32, 56, 56])
            x = self.s3(x)
            x2 = self.s3(x2)
            # torch.Size([1, 512, 8, 28, 28])
            # torch.Size([1, 64, 32, 28, 28])
            x = self.s3_fuse(x)
            x2 = self.s3_fuse(x2)
            # torch.Size([1, 640, 8, 28, 28])
            # torch.Size([1, 64, 32, 28, 28])
            x = self.s4(x)
            x2 = self.s4(x2)
            # torch.Size([1, 1024, 8, 14, 14])
            # torch.Size([1, 128, 32, 14, 14])
            x = self.s4_fuse(x)
            x2 = self.s4_fuse(x2)
            # torch.Size([1, 1280, 8, 14, 14])
            # torch.Size([1, 128, 32, 14, 14])
            x = self.s5(x)
            x2 = self.s5(x2)
            x1 = (x + x2) / 2
            # torch.Size([1, 2048, 8, 7, 7])
            # torch.Size([1, 256, 32, 7, 7])
            if self.enable_detection:
                x1 = self.head(x1, bboxes)
            else:
                x1 = self.head(x1)
            # torch.Size([1, 12])
        else:
            x = self.s1(x)
            x = self.s1_fuse(x)
            x = self.s2(x)
            x = self.s2_fuse(x)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            x = self.s3(x)
            x = self.s3_fuse(x)
            x = self.s4(x)
            x = self.s4_fuse(x)
            x = self.s5(x)
            if self.enable_detection:
                x = self.head(x, bboxes)
            else:
                x = self.head(x)
            x1 = x
        return x1


'''
# @torchsnooper.snoop()
class getEmotion():

    def __init__(self, videos):
        super(getEmotion, self).__init__()
        self.pretrained = '/public/home/zhouz/perl5/HAR/emotion/checkpoint/model_weights/weights_epoch_75.pth.tar'
        self.root = 'face_detector'
        device = torch.device('cuda:0')
        self.mini_xception = Mini_Xception().to(device)
        self.face_alignment = FaceAlignment()
        self.checkpoint = torch.load(self.pretrained, map_location=device)
        self.haar = True
        self.image = False
        self.videos = videos
        self.mini_xception.eval()
        self.mini_xception.load_state_dict(self.checkpoint['mini_xception'])
        self.change_size = Change_Size().to(device)

    def emotionMap(self, haar, image, video_path):
        root = self.root
        face_alignment = self.face_alignment
        mini_xception = self.mini_xception
        change_size = self.change_size
        # Face detection
        face_detector = None
        if haar:
            face_detector = HaarCascadeDetector(root)
        else:
            face_detector = DnnDetector(root)

        video = None
        isOpened = False
        if not image:
            if video_path:
                video = cv2.VideoCapture(video_path)
            else:
                video = cv2.VideoCapture(0)  # 480, 640
            isOpened = video.isOpened()
            # print('video.isOpened:', isOpened)

        emotion_map = []

        while isOpened:
            isOpened, frame = video.read()
            # if loaded video or image (not live camera) .. resize it
            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                # print('frame:', frame)  # frame: [[[101 140 152]
                # faces
                faces = face_detector.detect_faces(frame)
                # print("faces:", faces)

                for face in faces:
                    (x, y, w, h) = face
                    # print('face:', face)
                    # print('frame', frame)
                    # preprocessing
                    input_face = face_alignment.frontalize_face(face, frame)
                    # print('input_face:', input_face, input_face.size)  # 10800
                    input_face = cv2.resize(input_face, (48, 48))
                    # print('input_face1:', input_face, input_face.size)  # 2304
                    input_face = histogram_equalization(input_face)  # 提升像素亮度
                    # print('input_face2:', input_face, input_face.size)  # 2304
                    # cv2.imshow('input face', cv2.resize(input_face, (120, 120)))
                    device = torch.device('cuda:0')
                    input_face = transforms.ToTensor()(input_face).to(device)
                    # print('input_face3:', input_face, input_face.shape)  # [1, 48, 48])
                    input_face = torch.unsqueeze(input_face, 0)
                    # print('input_face4:', input_face, input_face.shape)  # [1, 1, 48, 48]

                    # input_face = input_face.to(device)
                    emotion = mini_xception(input_face)
                    # print('emotion.shape:', emotion.shape)  # [1, 7, 44, 44]
                    emotion = change_size(emotion)
                    # print('emotion.shape1:', emotion.shape)  # [1, 7, 1, 1]
                    # print('emotion:', emotion)  # tensor([[[[-0.4496]],[[-4.9016]],[[-1.3870]],[[ 5.3225]],[[-0.7316]],[[-5.5555]],[[ 0.9640]]]], device='cuda:0', grad_fn=<MeanBackward1>)
                    emotion = torch.squeeze(emotion, 0)
                    emotion = emotion.tolist()
                    if len(emotion_map) < 8:
                        # print('emotion_map.size:', len(emotion_map))
                        emotion_map.append(emotion)

                    break

        while (len(emotion_map) < 8) & (len(emotion_map) != 0):
            # print('emotion_map.size1:', len(emotion_map))
            emotion_map.append(emotion_map[-1])
        return emotion_map

    def __next__(self):
        emotion_maps = []
        haar = self.haar
        image = self.image
        videos = self.videos
        # print('videos:', videos)
        for video in videos:
            # print('video:', video)
            emotion_map = self.emotionMap(haar, image, video)
            # print('emotion_map_row:', emotion_map)
            if len(emotion_map) != 0:
                # emotion_map = torch.stack(emotion_map)
                # print('emotion_map:', emotion_map)
                emotion_maps.append(emotion_map)
        # emotion_maps = np.array(emotion_maps)
        # emotion_maps = torch.from_numpy(emotion_maps)
        while len(emotion_maps) < len(videos):
            emotion_maps.append(emotion_maps[-1])
        return emotion_maps
'''


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


class getGaze():

    def __init__(self, videos):
        # file_path11 = './'
        # file_names = os.listdir(file_path11)
        # print('file_names:', file_names)
        self.pretrained = './emotion/checkpoint/model_weights/weights_epoch_75.pth.tar'
        self.root = 'face_detector'
        device = torch.device('cuda')
        self.mini_xception = Mini_Xception().to(device)
        self.checkpoint = torch.load(self.pretrained, map_location=device)
        self.haar = True
        self.image = False
        self.videos = videos
        self.mini_xception.eval()
        self.mini_xception.load_state_dict(self.checkpoint['mini_xception'])
        self.change_size = Change_Size().to(device)

        super(getGaze, self).__init__()
        self.model_weights = './Gaze/weight/model_demo.pt'
        self.vis_mode = 'heatmap'
        self.out_threshold = 100

        self.root = 'face_detector'
        self.face_alignment = FaceAlignment()
        self.videos = videos
        device = torch.device('cuda')
        self.fc_inout = nn.Linear(49, 1).to(device)
        self.Change_Gaze_Size = Change_Gaze_Size().to(device)
        self.Change_Attention_Size = Change_Attention_Size().to(device)
        self.Change_Scene_Size = Change_Scene_Size().to(device)

    def img_cut(self, src_copy, points, h, w):
        target_points = [(0, 0), (w, 0), (w, h), (0, h)]
        points, target_points = np.array(points, dtype=np.float32), np.array(target_points, dtype=np.float32)
        m = cv2.getPerspectiveTransform(points, target_points)
        # print('透视变换矩阵:', m)

        result = cv2.warpPerspective(src_copy, m, (0, 0))
        result = result[:h, :w]
        # win_name = 'result'
        # cv.namedwindow(win_name, cv.window_normal)
        # cv.resizewindow(win_name, width=w, height=h)
        # cv.imshow(win_name, result)
        # cv.waitkey(0)
        # cv.destroyallwindows()
        return result

    def imresize(arr, size, interp='bilinear', mode=None):
        im = toimage(arr, mode=mode)
        ts = type(size)
        if issubdtype(ts, numpy.signedinteger):
            percent = size / 100.0
            size = tuple((array(im.size) * percent).astype(int))
        elif issubdtype(type(size), numpy.floating):
            size = tuple((array(im.size) * size).astype(int))
        else:
            size = (size[1], size[0])
        func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
        imnew = im.resize(size, resample=func[interp])
        return fromimage(imnew)

    def gazeMap(self, video_path):

        face_alignment = self.face_alignment
        mini_xception = self.mini_xception
        change_size = self.change_size
        Change_Gaze_Size = self.Change_Gaze_Size
        Change_Attention_Size = self.Change_Attention_Size
        Change_Scene_Size = self.Change_Scene_Size
        root = self.root

        # print('video_path:', video_path)  # /home/zhouzhuo/scratch/HID/Normal/normal082.mp4 /home/zhouzhuo/scratch/HID/Stealing/stealing129.mp4 /home/zhouzhuo/scratch/HID/Before_Stealing/Before_Stealing028.mp4

        annotation_path_root = video_path.split('HID')[0] + 'HID/Annotation/'

        annotation_path = annotation_path_root + video_path.split('HID/')[1].split('.mp4')[0]

        print('annotation_path:',
              annotation_path)  # video_path: /home/zhouzhuo/scratch/HID/Stealing/stealing099.mp4  annotation_path: /home/zhouzhuo/scratch/HID/Annotation/Stealing/stealing099
        if os.path.exists(annotation_path):

            cap = cv2.VideoCapture(video_path)
            frame_num = cap.get(7)
            isOpened = cap.isOpened()
            # print('video.isOpened:', isOpened)
            # clip = VideoFileClip(video_path)

            emotion_map = []
            gaze_map = []
            attention_map = []
            scene_map = []

            model = ModelSpatial()
            model_dict = model.state_dict()
            pretrained_dict = torch.load(self.model_weights)
            pretrained_dict = pretrained_dict['model']
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            model.cuda()
            model.train(False)
            i = 0
            while isOpened:
                isOpened, frame = cap.read()

                # if loaded video or image (not live camera) .. resize it
                if frame is not None:
                    formatted_str = '{:08d}'.format(i)
                    annotation = os.path.join(annotation_path, formatted_str + '.xml')
                    print('annotation',
                          annotation)  # annotation /home/zhouzhuo/scratch/HID/Annotation/Stealing/stealing033/00000000.xml
                    tree = ET.parse(annotation)
                    root = tree.getroot()
                    # 遍历XML文档
                    # 遍历所有object元素
                    for obj in root.findall('object'):
                        # 获取对象名称
                        name = obj.find('name').text
                        if name == 'face':
                            # 获取边界框坐标
                            xmin = int(obj.find('bndbox/xmin').text)
                            ymin = int(obj.find('bndbox/ymin').text)
                            xmax = int(obj.find('bndbox/xmax').text)
                            ymax = int(obj.find('bndbox/ymax').text)

                            # 打印坐标信息
                            # print(f'{name}: ({xmin}, {ymin}), ({xmax}, {ymax})')
                    i = i + 1
                    fh, fw = len(frame), len(frame[0])
                    # print(fh, fw)
                    # print('frame:', frame)
                    # faces

                    left = xmin
                    top = ymin
                    right = xmax
                    bottom = ymax
                    width = xmax - xmin
                    height = ymax - ymin

                    # points = [(left, top), (right, top), (right, bottom), (left, bottom)]

                    # faces: [[89 55 66 66]]
                    faces = (left, top, width, height)
                    faces = [[*faces]]

                    # faces = face_detector.detect_faces(frame)
                    # print("faces1:", faces)

                    src_frame = frame.copy()

                    frame2 = frame.copy()

                    for face in faces:

                        input_face = face_alignment.frontalize_face(face, frame2)
                        # print('input_face:', input_face, input_face.size)  # 10800
                        input_face = cv2.resize(input_face, (48, 48))
                        # print('input_face1:', input_face, input_face.size)  # 2304
                        input_face = histogram_equalization(input_face)  # 提升像素亮度
                        # print('input_face2:', input_face, input_face.size)  # 2304
                        # cv2.imshow('input face', cv2.resize(input_face, (120, 120)))
                        device = torch.device('cuda')
                        input_face = transforms.ToTensor()(input_face).to(device)
                        # print('input_face3:', input_face, input_face.shape)  # [1, 48, 48])
                        input_face = torch.unsqueeze(input_face, 0)
                        # print('input_face4:', input_face, input_face.shape)  # [1, 1, 48, 48]

                        # input_face = input_face.to(device)
                        emotion = mini_xception(input_face)
                        # print('emotion.shape:', emotion.shape)  # [1, 7, 44, 44]
                        emotion = change_size(emotion)
                        # print('emotion.shape1:', emotion.shape)  # [1, 2048, 7, 7]
                        # print('emotion:', emotion)  # tensor([[[[-0.4496]],[[-4.9016]],[[-1.3870]],[[ 5.3225]],[[-0.7316]],[[-5.5555]],[[ 0.9640]]]], device='cuda:0', grad_fn=<MeanBackward1>)
                        emotion = torch.squeeze(emotion, 0)
                        emotion = emotion.tolist()

                        (x, y, w, h) = face
                        left = x
                        top = y
                        right = x + w
                        bottom = y - h
                        width = w
                        height = h
                        test_transforms = _get_transform()

                        points = [(left, top), (right, top), (right, bottom), (left, bottom)]

                        head = self.img_cut(src_frame, points, fh, fw)

                        # frame = frame.convert('RGB')

                        head_box = [left, top, right, bottom]

                        # head = raw_frame.crop((head_box))  # head crop
                        # print('head:', head)
                        head_image = Image.fromarray(head)
                        head = test_transforms(head_image)  # transform inputs
                        # print('head--:', head)
                        # print('frame:', src_frame)
                        frame_image = Image.fromarray(
                            src_frame)  # 'Tensor' object has no attribute '__array_interface__'
                        frame = test_transforms(frame_image)
                        head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3],
                                                                    width,
                                                                    height,
                                                                    resolution=input_resolution).unsqueeze(0)

                        head = head.unsqueeze(0).cuda()
                        frame = frame.unsqueeze(0).cuda()
                        head_channel = head_channel.unsqueeze(0).cuda()

                        # forward pass
                        raw_hm, attention, inout = model(frame, head_channel, head)

                        # heatmap modulation
                        raw_hm = raw_hm.cpu().detach().numpy() * 255
                        raw_hm = raw_hm.squeeze()
                        scene = inout.clone()
                        inout = self.fc_inout(inout)
                        inout = inout.cpu().detach().numpy()
                        inout = 1 / (1 + np.exp(-inout))
                        inout = (1 - inout) * 255
                        # gaze = imresize(raw_hm, (height, width)) - inout
                        gaze = imresize(raw_hm, (48, 48)) - inout
                        gaze = torch.from_numpy(gaze)
                        gaze = Change_Gaze_Size(gaze)
                        gaze = gaze.squeeze()
                        # print('gaze:', gaze, gaze.shape)  # (50, 50)
                        attention = Change_Attention_Size(attention)
                        attention = attention.squeeze()
                        # print('attention:', attention, attention.shape)  # torch.Size([1, 1, 7, 7])
                        scene = Change_Scene_Size(scene)
                        scene = scene.squeeze()
                        # print('scene:', scene, scene.shape)  # torch.Size([1, 1])
                        gaze = gaze.tolist()
                        attention = attention.tolist()
                        scene = scene.tolist()

                        if len(emotion_map) < 8:
                            emotion_map.append(emotion)

                        if len(gaze_map) < 8:
                            # print('gaze_map.size::', len(gaze_map))
                            # gaze1 = np.array.tolist(gaze)
                            # print('gaze1:', gaze1)
                            # print('attention1:', attention1)
                            # print('scene1:', scene1)
                            gaze_map.append(gaze)
                            attention_map.append(attention)
                            scene_map.append(scene)
                        break
                if (len(gaze_map) == 8) & (len(emotion_map) == 8):
                    isOpened = False
            # print('gaze_map:', gaze_map)
            while (len(emotion_map) < 8) & (len(emotion_map) != 0):
                # print('gaze_map.size::', len(gaze_map))
                emotion_map.append(emotion_map[-1])

            while (len(gaze_map) < 8) & (len(gaze_map) != 0):
                # print('gaze_map.size::', len(gaze_map))
                gaze_map.append(gaze_map[-1])
                attention_map.append(attention_map[-1])
                scene_map.append(scene_map[-1])


        else:
            # Face detection
            face_detector = HaarCascadeDetector(root)

            video = None
            isOpened = False

            cap = cv2.VideoCapture(video_path)
            frame_num = cap.get(7)
            isOpened = cap.isOpened()
            # print('video.isOpened:', isOpened)
            # clip = VideoFileClip(video_path)
            j = 0

            emotion_map = []
            gaze_map = []
            attention_map = []
            scene_map = []

            model = ModelSpatial()
            model_dict = model.state_dict()
            pretrained_dict = torch.load(self.model_weights)
            pretrained_dict = pretrained_dict['model']
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            model.cuda()
            model.train(False)

            i = 1

            # print("frame_num:", frame_num)

            while isOpened:
                isOpened, frame = cap.read()

                # if loaded video or image (not live camera) .. resize it
                if frame is not None:

                    fh, fw = len(frame), len(frame[0])
                    # print(fh, fw)
                    # print('frame:', frame)
                    faces = face_detector.detect_faces(frame)
                    print("faces:", faces)  # faces: [[88 54 65 65]]
                    # obj = RetinaFace.detect_faces(frame)
                    # faces = obj['face_1']['facial_area']

                    src_frame = frame.copy()

                    frame2 = frame.copy()

                    for face in faces:
                        output_root = '/home/zhouzhuo/project/HAR/output_video/' + \
                                      video_path.split('/')[-1].split('.mp4')[0]
                        if not os.path.exists(output_root):
                            os.makedirs(output_root)
                        # 设置保存视频的路径和文件名
                        output_path = output_root + '/output_image' + str(
                            j) + '.jpg'  # /home/zhouzhuo/scratch/HID/Normal/normal082.mp4
                        print('output_path:', output_path)
                        j = j + 1

                        # 定义矩形框的坐标和尺寸
                        x, y, w, h = face  # 从faces中获取矩形框的坐标和尺寸

                        # 在帧上绘制矩形框
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # 将带有矩形框的帧写入输出视频文件
                        cv2.imwrite(output_path, frame)

                        print('frame.size:', frame.size)

                        input_face = face_alignment.frontalize_face(face, frame2)
                        # print('input_face:', input_face, input_face.size)  # 10800
                        input_face = cv2.resize(input_face, (48, 48))
                        # print('input_face1:', input_face, input_face.size)  # 2304
                        input_face = histogram_equalization(input_face)  # 提升像素亮度
                        # print('input_face2:', input_face, input_face.size)  # 2304
                        # cv2.imshow('input face', cv2.resize(input_face, (120, 120)))
                        device = torch.device('cuda')
                        input_face = transforms.ToTensor()(input_face).to(device)
                        # print('input_face3:', input_face, input_face.shape)  # [1, 48, 48])
                        input_face = torch.unsqueeze(input_face, 0)
                        # print('input_face4:', input_face, input_face.shape)  # [1, 1, 48, 48]

                        # input_face = input_face.to(device)
                        emotion = mini_xception(input_face)
                        # print('emotion.shape:', emotion.shape)  # [1, 7, 44, 44]
                        emotion = change_size(emotion)
                        # print('emotion.shape1:', emotion.shape)  # [1, 2048, 7, 7]
                        # print('emotion:', emotion)  # tensor([[[[-0.4496]],[[-4.9016]],[[-1.3870]],[[ 5.3225]],[[-0.7316]],[[-5.5555]],[[ 0.9640]]]], device='cuda:0', grad_fn=<MeanBackward1>)
                        emotion = torch.squeeze(emotion, 0)
                        emotion = emotion.tolist()

                        (x, y, w, h) = face
                        left = x
                        top = y
                        right = x + w
                        bottom = y - h
                        width = w
                        height = h
                        test_transforms = _get_transform()

                        points = [(left, top), (right, top), (right, bottom), (left, bottom)]

                        head = self.img_cut(src_frame, points, fh, fw)

                        # frame = frame.convert('RGB')

                        head_box = [left, top, right, bottom]

                        # head = raw_frame.crop((head_box))  # head crop
                        # print('head:', head)
                        head_image = Image.fromarray(head)
                        head = test_transforms(head_image)  # transform inputs
                        # print('head--:', head)
                        # print('frame:', src_frame)
                        frame_image = Image.fromarray(
                            src_frame)  # 'Tensor' object has no attribute '__array_interface__'
                        frame = test_transforms(frame_image)
                        head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3],
                                                                    width,
                                                                    height,
                                                                    resolution=input_resolution).unsqueeze(0)

                        head = head.unsqueeze(0).cuda()
                        frame = frame.unsqueeze(0).cuda()
                        head_channel = head_channel.unsqueeze(0).cuda()

                        # forward pass
                        raw_hm, attention, inout = model(frame, head_channel, head)

                        # heatmap modulation
                        raw_hm = raw_hm.cpu().detach().numpy() * 255
                        raw_hm = raw_hm.squeeze()
                        scene = inout.clone()
                        inout = self.fc_inout(inout)
                        inout = inout.cpu().detach().numpy()
                        inout = 1 / (1 + np.exp(-inout))
                        inout = (1 - inout) * 255
                        # gaze = imresize(raw_hm, (height, width)) - inout
                        gaze = imresize(raw_hm, (48, 48)) - inout
                        gaze = torch.from_numpy(gaze)
                        gaze = Change_Gaze_Size(gaze)
                        gaze = gaze.squeeze()
                        # print('gaze:', gaze, gaze.shape)  # (50, 50)
                        attention = Change_Attention_Size(attention)
                        attention = attention.squeeze()
                        # print('attention:', attention, attention.shape)  # torch.Size([1, 1, 7, 7])
                        scene = Change_Scene_Size(scene)
                        scene = scene.squeeze()
                        # print('scene:', scene, scene.shape)  # torch.Size([1, 1])
                        gaze = gaze.tolist()
                        attention = attention.tolist()
                        scene = scene.tolist()

                        if len(emotion_map) < 8:
                            emotion_map.append(emotion)

                        if len(gaze_map) < 8:
                            # print('gaze_map.size::', len(gaze_map))
                            # gaze1 = np.array.tolist(gaze)
                            # print('gaze1:', gaze1)
                            # print('attention1:', attention1)
                            # print('scene1:', scene1)
                            gaze_map.append(gaze)
                            attention_map.append(attention)
                            scene_map.append(scene)
                        break
                if (len(gaze_map) == 8) & (len(emotion_map) == 8):
                    isOpened = False
            # print('gaze_map:', gaze_map)
            while (len(emotion_map) < 8) & (len(emotion_map) != 0):
                # print('gaze_map.size::', len(gaze_map))
                emotion_map.append(emotion_map[-1])

            while (len(gaze_map) < 8) & (len(gaze_map) != 0):
                # print('gaze_map.size::', len(gaze_map))
                gaze_map.append(gaze_map[-1])
                attention_map.append(attention_map[-1])
                scene_map.append(scene_map[-1])

        return emotion_map, gaze_map, attention_map, scene_map

    def __next__(self):
        emotion_maps = []
        gaze_maps = []
        attention_maps = []
        scene_maps = []
        videos = self.videos
        for video in videos:
            emotion_map, gaze_map, attention_map, scene_map = self.gazeMap(video)
            # print('emotion_map:', len(emotion_map))
            if len(emotion_map) != 0:
                emotion_maps.append(emotion_map)
            if len(gaze_map) != 0:
                gaze_maps.append(gaze_map)
                attention_maps.append(attention_map)
                scene_maps.append(scene_map)
        # print('len(emotion_maps):', len(emotion_maps))
        # print('len(gaze_maps):', len(gaze_maps))
        while (len(emotion_maps) < len(videos)) & (len(emotion_maps) != 0):
            emotion_maps.append(emotion_maps[-1])
        while (len(gaze_maps) < len(videos)) & (len(gaze_maps) != 0):
            # print('len(gaze_maps):', len(gaze_maps))
            gaze_maps.append(gaze_maps[-1])
            attention_maps.append(attention_maps[-1])
            scene_maps.append(scene_maps[-1])
        return emotion_maps, gaze_maps, attention_maps, scene_maps


def variability_func(y):
    for i in range(len(y)):
        y[i] = torch.std(y[i])
    return y


def read_h5_file(filepath):
    data_dict = defaultdict(lambda: defaultdict(dict))

    with h5py.File(filepath, 'r') as f:
        print('f.key():', f.keys())
        for frame_number in f.keys():
            print('frame_number', frame_number)
            frame_group = f[frame_number]
            print('frame_group', frame_group.keys())
            frame_data = defaultdict(lambda: defaultdict(dict))

            for person_id in frame_group.keys():
                print('person_id', person_id)
                person_group = frame_group[person_id]
                print('person_group', person_group)
                person_data = {}
                if person_id == 'face_bboxes':
                    print('person_group:', person_group)
                    if isinstance(person_group, h5py.Dataset):
                        scene_feat = np.array(person_group)

                else:
                    for attr_name, attr_value in person_group.items():
                        print('attr_name', attr_name)
                        print('attr_value', attr_value)
                        if isinstance(attr_value, h5py.Dataset):
                            person_data[attr_name] = np.array(attr_value)

                frame_data[person_id] = person_data

            data_dict[frame_number] = frame_data
            data_dict[frame_number]['scene_feat'] = scene_feat

    return data_dict


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        # self.get_emotion = getEmotion

        self.Build_Graph = Build_Graph()
        self.get_gaze = getGaze

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.data_bn = nn.BatchNorm3d(2048)

        self.data_rule = nn.ReLU()

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.conv = nn.Conv3d(in_channels=2048, out_channels=1, kernel_size=1)

        self.projection = nn.Linear(2048, 3, bias=True)

        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)

    def forward(self, person_infos, videos, tymode, bboxes=None):
        i = 0
        person_feats = []
        person_ids = []
        # print('person_infos.keys():', len(person_infos.keys()), person_infos.keys())
        if (len(person_infos.keys())==1) and ('scene_feat' in person_infos.keys()):
            person_feats1 = []
            person_ids1 = []
            scene_feat = person_infos['scene_feat']
            scene_feat = scene_feat.permute(0, 2, 1, 3, 4).float().cuda()
            scene_feat = [scene_feat,] 
            scene_feat = self.head(scene_feat)
            person_feats1.append(scene_feat)
            person_ids1.append(int('-1'))
            return person_feats1, person_ids1
        else:
            i = 0
            for person_id in person_infos.keys():
                body_image = person_infos[person_id][0].cuda()
                gaze_feat = person_infos[person_id][1].cuda()
                emotion_feat = person_infos[person_id][2].cuda()
                attention_feat = person_infos[person_id][3].cuda()
                scene_feat = person_infos[person_id][4].cuda()

                # x1 = x[0]
                # x1 = x1.type(torch.cuda.FloatTensor)
                body_image = body_image.float()
                x = [body_image, ]
                x = self.s1(x)  # x1.size torch.Size([8, 64, 8, 56, 56])
                x = self.s2(x)  # x2.size torch.Size([8, 256, 8, 56, 56])
                for pathway in range(self.num_pathways):
                    pool = getattr(self, "pathway{}_pool".format(pathway))
                    x[pathway] = pool(x[pathway])

                x = self.s3(x)  # x3.size torch.Size([8, 512, 8, 28, 28])
                x = self.s4(x)  # [8,1024,8,14,14]
                x = self.s5(x)  # [8,2048,8,7,7]

                gaze_feat = gaze_feat.permute(0, 2, 1, 3, 4).float()
                gaze_feat = F.avg_pool3d(gaze_feat, kernel_size=(1, 9, 9), stride=(1, 9, 9))

                emotion_feat = emotion_feat.permute(0, 2, 1, 3, 4).float()
                emotion_feat = F.avg_pool3d(emotion_feat, kernel_size=(1, 9, 9), stride=(1, 9, 9))

                attention_feat = attention_feat.permute(0, 2, 1, 3, 4).float()
                attention_feat = F.avg_pool3d(attention_feat, kernel_size=(1, 9, 9), stride=(1, 9, 9))

                scene_feat = scene_feat.permute(0, 2, 1, 3, 4).float()
                scene_feat = F.avg_pool3d(scene_feat, kernel_size=(1, 9, 9), stride=(1, 9, 9))

                x = x[0]

                build_graph = self.Build_Graph
                cl, cl_edge = build_graph(x, gaze_feat, emotion_feat, attention_feat, scene_feat)

                feature_tensors = [x, gaze_feat, emotion_feat, attention_feat, scene_feat]

                feature_fusion = torch.sum(cl.unsqueeze(-2).unsqueeze(-1).unsqueeze(-1) * torch.stack(feature_tensors, dim=1), dim=1)

                person_feat = [feature_fusion, ]

                person_feat = self.head(person_feat)
                person_feats.append(person_feat)
                person_ids.append(person_id)
                i += 1
                
                # Clear intermediate variables and GPU cache
                del body_image, gaze_feat, emotion_feat, attention_feat, scene_feat, x#, cl, cl_edge, feature_tensors, weighted_sum, person_feat
                gc.collect()
                torch.cuda.empty_cache()

            return person_feats, person_ids
