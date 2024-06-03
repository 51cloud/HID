import os
import random
import torch
import torch.utils.data
import cv2
import numpy as np
from iopath.common.file_io import g_pathmgr
from torchvision import transforms
# import cupy as cp

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)
# 假设 additional_attributes 是一个默认的嵌套字典
from collections import defaultdict
import h5py


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            # self._num_clips = (
            #         cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            # )
            self._num_clips = 1

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        self.SMOOTHING_RADIUS = None
        self.alpha_sr = 0.1
        self.alpha_ri = 0.1
        self.alpha_rs = 0.1
        self.num_aug = 4

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []

        self._labels = []

        self._spatial_temporal_idx = []

        self._texts = []

        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):

                assert (len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)) == 2)
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )

                file_name = path.split('.')[0].split('/')[-1]
                root_path = path.split('HID')[0]
                text_root_path = root_path + 'HID/track_results/HID/'
                text_path = text_root_path + file_name + '/' + file_name + '.h5'

                if self.mode in ["train"]:
                    for idx in range(self._num_clips):
                        self._path_to_videos.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                        )
                        self._labels.append(int(label))
                        self._spatial_temporal_idx.append(idx)
                        self._video_meta[clip_idx * self._num_clips + idx] = {}
                        self._texts.append(text_path)
                else:
                    for idx in range(self._num_clips):
                        self._path_to_videos.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                        )
                        self._labels.append(int(label))
                        self._spatial_temporal_idx.append(idx)
                        self._video_meta[clip_idx * self._num_clips + idx] = {}
                        self._texts.append(text_path)

        assert (
                len(self._path_to_videos) > 0
        ), "Failed to load Kinetics  from {}".format(
            path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """

        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["val", "test"]:
            temporal_sample_index = (
                    self._spatial_temporal_idx[index]
                    // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                        self._spatial_temporal_idx[index]
                        % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                     + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        frames = []
        label = []
        videos = []
        threshold_lower = 1.0e-5
        threshold_upper = 1.0e+5

        for i_try in range(self._num_retries):
            # print('self.mode:', self.mode)
            # if self.mode in ["train"]:
            label = self._labels[index]
            video1 = self._path_to_videos[index]

            cap = cv2.VideoCapture(video1)
            video_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            videos.append(video1)

            try:
                video1_container = container.get_video_container(
                    # self._path_to_videos[index],
                    video1,
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video1 from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )

            # Select a random video if the current video was not able to access.
            if video1_container is None:
                logger.warning(
                    "Failed to meta load video1 idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.

            h5_path = self._texts[index]

            data_dict = self.read_h5_file(h5_path)

            frame_len = len(data_dict.keys()) 

            if frame_len < 8:
                frame_len = 8
            # print('frame_len:', frame_len)
            if frame_len > 20:
                frame_len = 20

            frame_rate = 8

            frames1 = decoder.decode(
                video1_container,
                frame_rate,
                # sampling_rate,
                # self.cfg.DATA.NUM_FRAMES,
                frame_len,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,

            )

            frame_count = 0
            gaze_feat = []
            emotion_feat = []
            attention_feat = []
            scene_feat = np.empty((frame_len, 2048, 64, 64), dtype=np.uint8)
            person_infos = {}
            body_images = {}
            gaze_feat = {}
            emotion_feat = {}
            attention_feat = {}
            person_infoss = {}
            
            for i in range(frame_len):
                frame_group = data_dict[str(frame_count)]
                for person_id, attributes in frame_group.items():
                    if person_id == 'face_bboxes':
                        continue
                    elif person_id == 'scene_feat':
                        attributes = np.array(attributes)
                        scene_feat[i, :, :, :] = attributes

                    elif isinstance(attributes, dict):
                        if person_id not in body_images.keys():
                            body_images[person_id] = np.empty((3, frame_len, 224, 224), dtype=np.uint8)
                            gaze_feat[person_id] = np.empty((frame_len, 2048, 64, 64), dtype=np.uint8)
                            emotion_feat[person_id] = np.empty((frame_len, 2048, 64, 64), dtype=np.uint8) 
                            attention_feat[person_id] = np.empty((frame_len, 2048, 64, 64), dtype=np.uint8)
                            if person_id not in person_infos.keys():
                                person_infos[person_id] = []
                        person_group = frame_group[person_id]
                        for attr_name, attr_value in attributes.items():
                            if attr_name == 'body_bbox':
                                # print('attr_value1:', attr_value)
                                x1, y1, w, h = attr_value
                                if x1 < 0:
                                    x1 = 0
                                if y1 < 0:
                                    y1 = 0
                                if w <= 0:
                                    w = 3
                                if h <= 0:
                                    h = 3
                                frame = frames1[i].numpy()
                                cropped_frame = frame[int(y1):int(y1 + h), int(x1):int(x1 + w)]

                                if cropped_frame is None:
                                    logger.warning(
                                        "Failed to decode video1 idx {} from {}; trial {}".format(
                                            index, self._path_to_videos[index], i_try
                                        )
                                    )
                                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                                        # let's try another one
                                        index = random.randint(0, len(self._path_to_videos) - 1)
                                    continue

                                mean = self.cfg.DATA.MEAN

                                std = self.cfg.DATA.STD

                                cropped_frame = torch.tensor(cropped_frame)

                                cropped_frame = utils.tensor_normalize(
                                    cropped_frame, mean, std
                                )
                                cropped_frame = cropped_frame.unsqueeze(0)
                                cropped_frame = cropped_frame.permute(3, 0, 1, 2)
                                cropped_frame = utils.spatial_sampling(
                                    cropped_frame,
                                    spatial_idx=spatial_sample_index,
                                    min_scale=min_scale,
                                    max_scale=max_scale,
                                    crop_size=crop_size,
                                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                                )
                                cropped_frame = cropped_frame.squeeze(1)
                                body_images[person_id][:, i, :, :] = cropped_frame

                            if attr_name == 'gaze_feat':
                                gaze_feat[person_id][i, :, :, :] = attr_value
                            if attr_name == 'emotion_feat':
                                emotion_feat[person_id][i, :, :, :] = attr_value
                            if attr_name == 'attention_feat':
                                attention_feat[person_id][i, :, :, :] = attr_value
                if (frame_count + frame_rate) < video_frame:
                    frame_count += frame_rate
            if len(person_infos.keys()) == 0:
                person_infos['scene_feat'] = scene_feat
            else:
                for person_id in person_infos.keys():
                    person_infos[person_id].append(body_images[person_id])
                    person_infos[person_id].append(gaze_feat[person_id])
                    person_infos[person_id].append(emotion_feat[person_id])
                    person_infos[person_id].append(attention_feat[person_id])
                    person_infos[person_id].append(scene_feat)

            if len(person_infos) > 3:
                keys_to_keep = list(person_infos.keys())[:3]
                person_infos = {key: person_infos[key] for key in keys_to_keep}

            return person_infos, videos, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video3 after {} retries.".format(
                    self._num_retries
                )
            )

    def _aug_frame(
            self,
            frames,
            spatial_sample_index,
            min_scale,
            max_scale,
            crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def _crop_frames_with_bboxes(self, frames, body_bboxes):
        """
        根据bbox裁剪每一帧
        """
        body_video = {}
        for person_id, bboxes in body_bboxes.items():
            body_frame = []
            for i in range(8):
                x1, y1, w, h = bboxes[i]
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if w <= 0:
                    w = 3
                if h <= 0:
                    h = 3
                frame = frames[i].numpy()
                cropped_frame = frame[int(y1):int(y1 + h), int(x1):int(x1 + w)]
                cropped_frame = cv2.resize(cropped_frame, (64, 64))
                body_frame[i] = cropped_frame
            body_video[person_id] = body_frame
        # print('body_video:', body_video)
        return cropped_videos

    def read_h5_file(self, filepath):
        data_dict = defaultdict(lambda: defaultdict(dict))
        threshold_lower = 1.0e-5
        threshold_upper = 1.0e+5
        frame_rate = 8
        with h5py.File(filepath, 'r') as f:
            frame_len = len(f.keys())
            for frame_number in f.keys():
                if (int(frame_number) + frame_rate) % frame_rate == 0:
                    scene_feat = np.random.randint(0.5, 4, size=(1, 2048, 64, 64), dtype=np.uint8)
                    frame_group = f[frame_number]
                    frame_data = defaultdict(lambda: defaultdict(dict))
                    person_data = {}
                    person_data['scene_feat'] =  np.random.randint(0.5, 4, size=(1, 2048, 64, 64), dtype=np.uint8)
                    person_data['gaze_feat'] =  np.random.randint(0.5, 4, size=(1, 2048, 64, 64), dtype=np.uint8)
                    person_data['emotion_feat'] =  np.random.randint(0.5, 4, size=(1, 2048, 64, 64), dtype=np.uint8)
                    person_data['attention_feat'] =  np.random.randint(0.5, 4, size=(1, 2048, 64, 64), dtype=np.uint8)
                    for person_id in frame_group.keys():
                        person_group = frame_group[person_id]
                        
                        if person_id == 'scene_feat':
                            if isinstance(person_group, h5py.Dataset):
                                scene_feat = np.array(person_group)
                                nan_indices = np.where(np.isnan(scene_feat))
                                lower_indices = np.where(abs(scene_feat) <= threshold_lower)
                                upper_indices = np.where(abs(scene_feat) >= threshold_upper)
                                
                                # 替换异常值为随机数
                                if len(nan_indices[0]) > 0:
                                    scene_feat[nan_indices] = np.random.uniform(0.5, 4, size=len(nan_indices[0]))
                                
                                # 替换异常值为随机数
                                if len(lower_indices[0]) > 0:
                                    scene_feat[lower_indices] = np.random.uniform(0.5, 4, size=len(lower_indices[0]))
                                
                                if len(upper_indices[0]) > 0:
                                    scene_feat[upper_indices] = np.random.uniform(0.5, 4, size=len(upper_indices[0]))

                                scene_feat = scene_feat
                                
                        elif person_id == 'face_bboxes':
                            if isinstance(person_group, h5py.Dataset):
                                scene_feat = np.array(person_group)
                                nan_indices = np.where(np.isnan(scene_feat))
                                lower_indices = np.where(abs(scene_feat) <= threshold_lower)
                                upper_indices = np.where(abs(scene_feat) >= threshold_upper)
                                
                                if len(nan_indices[0]) > 0:
                                    scene_feat[nan_indices] = np.random.uniform(0.5, 4, size=len(nan_indices[0]))
                                
                                if len(lower_indices[0]) > 0:
                                    scene_feat[lower_indices] = np.random.uniform(0.5, 4, size=len(lower_indices[0]))
                                
                                if len(upper_indices[0]) > 0:
                                    scene_feat[upper_indices] = np.random.uniform(0.5, 4, size=len(upper_indices[0]))
                                scene_feat = scene_feat
                        else:
                            for attr_name, attr_value in person_group.items():
                                if attr_name == 'gaze_feature':
                                    attr_name = 'gaze_feat'
                                    attr_value = np.array(attr_value)
                                    nan_indices = np.where(np.isnan(attr_value))
                                    lower_indices = np.where(abs(attr_value) <= threshold_lower)
                                    upper_indices = np.where(abs(attr_value) >= threshold_upper)
                                    
                                    if len(nan_indices[0]) > 0:
                                        attr_value[nan_indices] = np.random.uniform(0.5, 4, size=len(nan_indices[0]))
                                    
                                    if len(lower_indices[0]) > 0:
                                        attr_value[lower_indices] = np.random.uniform(0.5, 4, size=len(lower_indices[0]))
                                    
                                    if len(upper_indices[0]) > 0:
                                        attr_value[upper_indices] = np.random.uniform(0.5, 4, size=len(upper_indices[0]))
        
                                    if attr_value.shape == (1, 1, 14, 14):
                                        attr_value = np.pad(attr_value, ((0, 0), (0, 0), (25, 25), (25, 25)), mode='constant', constant_values=0)
                                    elif attr_value.shape == (1, 1, 64, 64):
                                        attr_value = attr_value
                                    else:
                                        attr_value = np.random.randint(100, 256, size=(1, 1, 64, 64), dtype=np.uint8)
                                    
                                    person_data[attr_name][:,0,:,:] = attr_value
                                    
                                if attr_name == 'emotion_feat':
                                    attr_value = np.array(attr_value)
                                    nan_indices = np.where(np.isnan(attr_value))
                                    lower_indices = np.where(abs(attr_value) <= threshold_lower)
                                    upper_indices = np.where(abs(attr_value) >= threshold_upper)
                                    
                                    if len(nan_indices[0]) > 0:
                                        attr_value[nan_indices] = np.random.uniform(0.5, 4, size=len(nan_indices[0]))
                                    
                                    if len(lower_indices[0]) > 0:
                                        attr_value[lower_indices] = np.random.uniform(0.5, 4, size=len(lower_indices[0]))
                                    
                                    if len(upper_indices[0]) > 0:
                                        attr_value[upper_indices] = np.random.uniform(0.5, 4, size=len(upper_indices[0]))
                                    
                                    person_data[attr_name][:,0,:,:] = attr_value
                                if attr_name == 'attention_feat':
                                    attr_value = np.array(attr_value)
                                    nan_indices = np.where(np.isnan(attr_value))
                                    lower_indices = np.where(abs(attr_value) <= threshold_lower)
                                    upper_indices = np.where(abs(attr_value) >= threshold_upper)
                                    
                                    if len(nan_indices[0]) > 0:
                                        attr_value[nan_indices] = np.random.uniform(0.5, 4, size=len(nan_indices[0]))

                                    if len(lower_indices[0]) > 0:
                                        attr_value[lower_indices] = np.random.uniform(0.5, 4, size=len(lower_indices[0]))
                                    
                                    if len(upper_indices[0]) > 0:
                                        attr_value[upper_indices] = np.random.uniform(0.5, 4, size=len(upper_indices[0]))
                                    
                                    person_data[attr_name][:,0,:,:] = attr_value
                                
                                if isinstance(attr_value, h5py.Dataset):
                                    person_data[attr_name] = np.array(attr_value)

                        frame_data[person_id] = person_data

                    data_dict[frame_number] = frame_data
                    data_dict[frame_number]['scene_feat'] = scene_feat
        return data_dict
