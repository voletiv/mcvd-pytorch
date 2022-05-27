# https://github.com/edenton/svg/blob/master/data/bair.py
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .h5 import HDF5Dataset


class BAIRDataset(Dataset):

    def __init__(self, data_path, frames_per_sample=5, random_time=True, random_horizontal_flip=True, color_jitter=0,
                 total_videos=-1, with_target=True):

        self.data_path = data_path                    # '/path/to/Datasets/BAIR_h5/train' (with shard_0001.hdf5 in it), or /path/to/BAIR_h5/train/shard_0001.hdf5
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.color_jitter = color_jitter
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.with_target = with_target

        self.jitter = transforms.ColorJitter(hue=color_jitter)

        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

        print(f"Dataset length: {self.__len__()}")

    def window_stack(self, a, width=3, step=1):
        return torch.stack([a[i:1+i-width or None:step] for i in range(width)]).transpose(0, 1)

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.videos_ds)

    def max_index(self):
        return len(self.videos_ds)

    def __getitem__(self, index, time_idx=0):

        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

        prefinals = []
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
            if self.random_time and video_len > self.frames_per_sample:
                time_idx = np.random.choice(video_len - self.frames_per_sample)
            for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
                # byte_str = f[str(idx_in_shard)][str(i)][()]
                # img = Image.frombytes('RGB', (64, 64), byte_str)
                # arr = np.expand_dims(np.array(img.getdata()).reshape(img.size[1], img.size[0], 3), 0)
                img = f[str(idx_in_shard)][str(i)][()]
                arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img))
                prefinals.append(arr)

        data = torch.stack(prefinals)
        data = self.jitter(data)

        if self.with_target:
            return data, torch.tensor(1)
        else:
            return data
