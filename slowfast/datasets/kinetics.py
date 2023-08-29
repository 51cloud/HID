import os
import random
import torch
import torch.utils.data
import cv2
import numpy as np
from iopath.common.file_io import g_pathmgr
from torchvision import transforms

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


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

        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)) == 2)
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                print('path:', path) # /public/home/jiaxm/perl5/datasets/N-UCLA/view1/v01_s10_e05_a11.avi
                # print("self._num_clips:",self._num_clips) # 30
                if self.mode in ["train"]:
                    for idx in range(self._num_clips):
                        self._path_to_videos.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                        )
                        self._labels.append(int(label))
                        self._spatial_temporal_idx.append(idx)
                        self._video_meta[clip_idx * self._num_clips + idx] = {}
                else:
                    for idx in range(self._num_clips):
                        self._path_to_videos.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                        )
                        self._labels.append(int(label))
                        self._spatial_temporal_idx.append(idx)
                        self._video_meta[clip_idx * self._num_clips + idx] = {}

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

        for i_try in range(self._num_retries):
            # print('self.mode:', self.mode)
            if self.mode in ["train"]:

                video1 = self._path_to_videos[index]
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
                # print('self._video_meta[]:', self._video_meta)
                # print('self._video_meta[].len:', len(self._video_meta))
                frames1 = decoder.decode(
                    video1_container,
                    sampling_rate,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,

                )

                # If decoding failed (wrong format, video is too short, and etc),
                # select another video.
                if frames1 is None:
                    logger.warning(
                        "Failed to decode video1 idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                if self.aug:
                    if self.cfg.AUG.NUM_SAMPLE > 1:

                        frame_list = []
                        label_list = []
                        index_list = []
                        for _ in range(self.cfg.AUG.NUM_SAMPLE):
                            new_frames1 = self._aug_frame(
                                frames1,
                                spatial_sample_index,
                                min_scale,
                                max_scale,
                                crop_size,
                            )
                            label = self._labels[index]
                            new_frames1 = utils.pack_pathway_output(
                                self.cfg, new_frames1
                            )
                            frame_list.append(new_frames1)
                            label_list.append(label)
                            index_list.append(index)
                        return frame_list, label_list, index_list, {}

                    else:
                        frames1 = self._aug_frame(
                            frames1,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                else:
                    # print('frames1:', frames1)

                    # frames2 = self.stable(frames1)
                    frames2 = frames1
                    # frames3 = self.GenNegative(frames1)
                    # print('new_frames1.shape:', frames1.shape)
                    # print('MEAN, STD:', self.cfg.DATA.MEAN, self.cfg.DATA.STD)

                    # frames1 = utils.tensor_normalize(
                    #     frames1, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    # )
                    # device = torch.device('cuda:0')
                    # frames2 = frames2.to(device)
                    mean = self.cfg.DATA.MEAN
                    # mean = np.array(mean)
                    # mean = torch.from_numpy(mean)
                    # mean = mean.to(device)
                    # print('mean:', mean)
                    std = self.cfg.DATA.STD
                    # std = np.array(std)
                    # std = torch.from_numpy(std)
                    # std = std.to(device)

                    frames2 = utils.tensor_normalize(
                        frames2, mean, std
                    )

                    # frames3 = utils.tensor_normalize(
                    #     frames3, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    # )

                    # print('new_frames1:', frames1)
                    # T H W C -> C T H W.
                    # frames1 = frames1.permute(3, 0, 1, 2)

                    frames2 = frames2.permute(3, 0, 1, 2)

                    # frames3 = frames3.permute(3, 0, 1, 2)
                    # Perform data augmentation.


                    frames2 = utils.spatial_sampling(
                        frames2,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )



                label = self._labels[index]
                frames = utils.pack_pathway_output(self.cfg, frames2) # original video , stable video , other_video
                # print('new_new_frames1:', frames1)
            elif self.mode in ["val", "test"]:
                # print('-----------------val-----------------')
                video_container = None
                video1 = self._path_to_videos[index]
                videos.append(video1)
                try:
                    video_container = container.get_video_container(
                        self._path_to_videos[index],
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video3 from {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                # Select a random video if the current video was not able to access.
                if video_container is None:
                    logger.warning(
                        "Failed to meta load video3 idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                # Decode video. Meta info is used to perform selective decoding.
                frames = decoder.decode(
                    video_container,
                    sampling_rate,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,
                )

                # If decoding failed (wrong format, video is too short, and etc),
                # select another video.
                if frames is None:
                    logger.warning(
                        "Failed to decode video3 idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                if self.aug:
                    if self.cfg.AUG.NUM_SAMPLE > 1:

                        frame_list = []
                        label_list = []
                        index_list = []
                        for _ in range(self.cfg.AUG.NUM_SAMPLE):
                            new_frames = self._aug_frame(
                                frames,
                                spatial_sample_index,
                                min_scale,
                                max_scale,
                                crop_size,
                            )
                            label = self._labels[index]
                            new_frames = utils.pack_pathway_output(
                                self.cfg, new_frames
                            )
                            frame_list.append(new_frames)
                            label_list.append(label)
                            index_list.append(index)
                        return frame_list, label_list, index_list, {}

                    else:
                        frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )

                else:
                    # frames = self.stable(frames)
                    frames = utils.tensor_normalize(
                        frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )
                    # T H W C -> C T H W.
                    frames = frames.permute(3, 0, 1, 2)
                    # Perform data augmentation.
                    frames = utils.spatial_sampling(
                        frames,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )

                label = self._labels[index]
                frames = utils.pack_pathway_output(self.cfg, frames)
            # print('videos:', videos)
            #     frames_clips = []
            #     label_clips = []
            #     path1 = video1.split('Before_Stealing/')[0] # /public/home/zhouz/perl5/dataset/crime/Before_Stealing/Before_Stealing172.mp4
            #     path2 = video1.split('Before_Stealing/')[1]
            #     video_clips_path = path1 + 'test_ori_clip/' + path2.split('.')[0]
            #     video_clips_lists = os.listdir(video_clips_path)
            #     for video_clip_list in video_clips_lists:
            #         video_clip_path = video_clips_path + '/' + video_clip_list
            #         print("self._path_to_videos[index]:", self._path_to_videos[index])
            #         print("video_clip_path:", video_clip_path)
            #         label_clip = video_clip_list.split('label')[1].split('.')[0]
            #         try:
            #             video_container = container.get_video_container(
            #                 video_clip_path,
            #                 self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
            #                 self.cfg.DATA.DECODING_BACKEND,
            #             )
            #         except Exception as e:
            #             logger.info(
            #                 "Failed to load video3 from {} with error {}".format(
            #                     video_clip_path, e
            #                 )
            #             )
            #         # Select a random video if the current video was not able to access.
            #         if video_container is None:
            #             logger.warning(
            #                 "Failed to meta load video3 idx {} from {}; trial {}".format(
            #                     index, video_clip_path, i_try
            #                 )
            #             )
            #             if self.mode not in ["test"] and i_try > self._num_retries // 2:
            #                 # let's try another one
            #                 index = random.randint(0, len(self._path_to_videos) - 1)
            #             continue
            #
            #         # Decode video. Meta info is used to perform selective decoding.
            #         frames_clip = decoder.decode(
            #             video_container,
            #             sampling_rate,
            #             self.cfg.DATA.NUM_FRAMES,
            #             temporal_sample_index,
            #             self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
            #             video_meta=self._video_meta[index],
            #             target_fps=self.cfg.DATA.TARGET_FPS,
            #             backend=self.cfg.DATA.DECODING_BACKEND,
            #             max_spatial_scale=min_scale,
            #         )
            #
            #         # If decoding failed (wrong format, video is too short, and etc),
            #         # select another video.
            #         if frames_clip is None:
            #             logger.warning(
            #                 "Failed to decode video3 idx {} from {}; trial {}".format(
            #                     index, video_clip_path, i_try
            #                 )
            #             )
            #             if self.mode not in ["test"] and i_try > self._num_retries // 2:
            #                 # let's try another one
            #                 index = random.randint(0, len(self._path_to_videos) - 1)
            #             continue
            #
            #         frames_clip = utils.tensor_normalize(
            #             frames_clip, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            #         )
            #         # T H W C -> C T H W.
            #         frames_clip = frames_clip.permute(3, 0, 1, 2)
            #         # Perform data augmentation.
            #         frames_clip = utils.spatial_sampling(
            #             frames_clip,
            #             spatial_idx=spatial_sample_index,
            #             min_scale=min_scale,
            #             max_scale=max_scale,
            #             crop_size=crop_size,
            #             random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            #             inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            #         )
            #         frames_clip = utils.pack_pathway_output(self.cfg, frames_clip)
            #         frames_clips.append(frames_clip)
            #         label_clips.append(label_clip)
            return frames, videos, label, index, {}
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

    def movingAverage(self, curve, radius):
        window_size = 2 * radius + 1
        # Define the filter
        f = np.ones(window_size) / window_size
        # Add padding to the boundaries
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        # Apply convolution
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        # Remove padding
        curve_smoothed = curve_smoothed[radius:-radius]
        # return smoothed curve
        return curve_smoothed

    def smooth(self, trajectory):

        smoothed_trajectory = np.copy(trajectory)
        # Filter the x, y and angle curves
        for i in range(3):
            smoothed_trajectory[:, i] = self.movingAverage(trajectory[:, i], radius=self.SMOOTHING_RADIUS)

        return smoothed_trajectory

    # Compute trajectory using cumulative sum of transformations

    # Step 5.1 : Fix border artifacts
    def fixBorder(self, frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

    def stable(self, frames):
        # Step 1 : Set Input and Output Videos
        # Read input video
        # print("frames.shape", frames.shape) # (8,1080,1920,3)
        frames = frames.cpu().detach().numpy()
        # print('frames:', frames)
        # Get frame count
        n_frames = int(len(frames))
        prev = frames[0]
        # print("prev:", prev)
        # Get width and height of video stream
        w = len(frames[0][0])
        h = len(frames[0])
        # print("w,h", w, h)
        # Define the codec for output video

        self.SMOOTHING_RADIUS = 3

        # Read the first frame and convert it to grayscale

        # Step 2: Read the first frame and convert it to grayscale
        # Read first frame

        # Convert frame to grayscale
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        # Step 3: Find motion between frames

        # Pre-define transformation-store array
        transforms = np.zeros((n_frames, 3), np.float32)

        for i in range(n_frames - 1):
            # Detect feature points in previous frame
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                               qualityLevel=0.01, minDistance=30, blockSize=3)
            # Read next frame
            if i < n_frames:
                curr = frames[i + 1]
            else:
                curr = frames[i]
            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

            # Sanity check
            assert prev_pts.shape == curr_pts.shape

            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Find transformation matrix
            # m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)  # will only work with OpenCV-3 or less

            m, inlier = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            # Extract traslation
            dx = m[0, 2]
            dy = m[1, 2]
            # Extract rotation angle
            da = np.arctan2(m[1, 0], m[0, 0])

            # Store transformation
            transforms[i] = [dx, dy, da]

            # Move to next frame
            prev_gray = curr_gray

            # print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

        # Step 4: Calculate smooth motion between frames
        # Compute trajectory using cumulative sum of transformations
        trajectory = np.cumsum(transforms, axis=0)

        # curve_smoothed = self.movingAverage()

        smoothed_trajectory = self.smooth(trajectory)

        trajectory = np.cumsum(transforms, axis=0)
        # Step 4.3 : Calculate smooth transforms

        # Calculate difference in smoothed_trajectory and trajectory
        difference = smoothed_trajectory - trajectory

        # Calculate newer transformation array
        transforms_smooth = transforms + difference

        # Step 5: Apply smoothed camera motion to frames
        # Reset stream to first frame
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        new_frame = np.zeros((n_frames, h, w, 3), np.float32)
        # Write n_frames-1 transformed frames
        for i in range(n_frames):
            # Read next frame
            frame = frames[i]

            # Extract transformations from the new transformation array
            dx = transforms_smooth[i, 0]
            dy = transforms_smooth[i, 1]
            da = transforms_smooth[i, 2]

            # Reconstruct transformation matrix accordingly to new values
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy

            # Apply affine wrapping to the given frame
            frame_stabilized = cv2.warpAffine(frame, m, (w, h))

            # Fix border artifacts
            frame_stabilized = self.fixBorder(frame_stabilized)
            # Write the frame to the file
            new_frame[i] = frame_stabilized
            # If the image is too big, resize it.
            # if frame_out.shape[1] > 1920:
            #     frame_out = cv2.resize(frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2))
            #     print('frame_out', frame_out)
            #     # cv2.imshow("Before and After", frame_out)
            # cv2.waitKey(10)
        new_frame = torch.from_numpy(new_frame).cuda()
        new_frame = torch.tensor(new_frame, dtype=torch.uint8)
        # print('new_frame.shape:', new_frame.shape)
        # print('new_frame:', new_frame)
        return new_frame

    def GenNegative(self, inp):

        c, t, h, w = inp.size()

        augmented_sentences = []
        num_new_per_technique = int(self.num_aug / 4) + 1
        n_sr = max(1, int(self.alpha_sr * t))
        n_ri = max(1, int(self.alpha_ri * t))
        n_rs = max(1, int(self.alpha_rs * t))

        # # sr synonym replacement
        # for _ in range(num_new_per_technique):
        #     inp = synonym_replacement(inp, n_sr)

        # ri random insertion
        for _ in range(num_new_per_technique):
            inp = self.random_insertion(inp, n_ri)

        # rs random swap
        for _ in range(num_new_per_technique):
            inp = self.random_swap(inp, n_rs)

        # # rd random delte
        # for _ in range(num_new_per_technique):
        #     inp = random_deletion(inp, self.p_rd)

        return inp

    ########################################################################
    # Random swap
    # Randomly swap two words in the sentence n times
    ########################################################################

    def random_swap(self, videos, n):
        new_videos = videos
        for _ in range(n):
            new_videos = self.swap_word(new_videos)
        return new_videos

    def swap_word(self, new_videos):
        c, t, h, w = new_videos.size()
        random_idx_1 = random.randint(0, t - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, t - 1)
            counter += 1
            if counter > 3:
                return new_videos
        new_videos[:, random_idx_1, :, :], new_videos[:, random_idx_2, :, :] \
            = new_videos[:, random_idx_2, :, :], new_videos[:, random_idx_1, :, :]
        return new_videos

    ########################################################################
    # Random insertion
    # Randomly insert n words into the sentence
    ########################################################################

    def random_insertion(self, videos, n):
        new_videos = videos
        for _ in range(n):
            new_videos = self.add_picture(new_videos)
        return new_videos

    def add_picture(self, videos):
        c, t, h, w = videos.size()
        new_videos = videos
        random_idx = random.randint(0, t - 1)
        random_idx2 = random.randint(0, t - 1)
        # this is from the same sample, may be need modify
        new_videos[:, random_idx + 1:, :, :] = videos[:, random_idx:t - 1, :, :]
        new_videos[:, random_idx, :, :] = videos[:, random_idx2, :, :]
        return new_videos

    ########################################################################
    # main data augmentation function
    ########################################################################