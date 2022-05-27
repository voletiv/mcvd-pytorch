# https://github.com/edenton/svg/blob/master/data/kth.py
import numpy as np
import os
import pickle
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .h5 import HDF5Dataset


class UCF101Dataset(Dataset):

    def __init__(self, data_path, frames_per_sample=5, image_size=64, train=True, random_time=True, random_horizontal_flip=True,
                 total_videos=-1, skip_videos=0, with_target=True):

        self.data_path = data_path                    # '/path/to/Datasets/UCF101_64_h5' (with .hdf5 file in it), or to the hdf5 file itself
        self.train = train
        self.frames_per_sample = frames_per_sample
        self.image_size = image_size
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.with_target = with_target

        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

        # Train
        # self.num_train_vids = 9624
        # self.num_test_vids = 3696   # -> 369 : https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
        with self.videos_ds.opener(self.videos_ds.shard_paths[0]) as f:
            self.num_train_vids = f['num_train'][()]
            self.num_test_vids = f['num_test'][()]//10  # https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video

        print(f"Dataset length: {self.__len__()}")

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else self.num_train_vids if self.train else self.num_test_vids

    def max_index(self):
        return self.num_train_vids if self.train else self.num_test_vids

    def __getitem__(self, index, time_idx=0):

        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        if not self.train:
            video_index = video_index * 10 + self.num_train_vids    # https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

        # random crop
        crop_c = np.random.randint(int(self.image_size/240*320) - self.image_size) if self.train else int((self.image_size/240*320 - self.image_size)/2)

        # random horizontal flip
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0

        # read data
        prefinals = []
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            target = int(f['target'][str(idx_in_shard)][()])
            # slice data
            video_len = f['len'][str(idx_in_shard)][()]
            if self.random_time and video_len > self.frames_per_sample:
                time_idx = np.random.choice(video_len - self.frames_per_sample)
            for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
                img = f[str(idx_in_shard)][str(i)][()]
                arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img[:, crop_c:crop_c + self.image_size]))
                prefinals.append(arr)

        video = torch.stack(prefinals)

        if self.with_target:
            return video, torch.tensor(target)
        else:
            return video
