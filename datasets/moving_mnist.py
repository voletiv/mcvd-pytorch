import gzip
import math
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

import progressbar

from collections import OrderedDict

pbar = None


def mmnist_data_loader(collate_fn=None, n_frames=10, num_digits=1, with_target=False,
                       batch_size=100, n_workers=8, is_train=True, drop_last=True,
                       dset_path=os.path.dirname(os.path.realpath(__file__))):
    dset = MovingMNIST(dset_path, is_train, n_frames, num_digits, with_target=with_target)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=is_train, collate_fn=collate_fn,
                              num_workers=n_workers, drop_last=drop_last, pin_memory=True)
    # Returns images of size [1, 64, 64] in [-1, 1]
    return dloader


class ToTensor(object):
    """Converts a numpy.ndarray (... x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (... x C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, scale=True):
        self.scale = scale
    def __call__(self, arr):
        if isinstance(arr, np.ndarray):
            video = torch.from_numpy(np.rollaxis(arr, axis=-1, start=-3))
            if self.scale:
                return video.float().div(255)
            else:
                return video.float()
        else:
            raise NotImplementedError


# def load_mnist(root):
#     # Load MNIST dataset for generating training data.
#     path = os.path.join(root, 'train-images-idx3-ubyte.gz')
#     with gzip.open(path, 'rb') as f:
#         mnist = np.frombuffer(f.read(), np.uint8, offset=16)
#         mnist = mnist.reshape(-1, 28, 28)
#     return mnist


# def load_fixed_set(root, is_train):
#     # Load the fixed dataset
#     filename = 'mnist_test_seq.npy'
#     path = os.path.join(root, filename)
#     dataset = np.load(path)
#     dataset = dataset[..., np.newaxis]
#     return dataset


# loads mnist from web on demand
def load_mnist(root, is_train=True):

    def load_mnist_images(filename):
        if not os.path.exists(os.path.join(root, filename)):
            download(root, filename)
        with gzip.open(os.path.join(root, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
        return data

    if is_train:
        return load_mnist_images('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    return load_mnist_images('http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy')


def download(root, filename):
    def show_progress(block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None
    print("Downloading %s" % os.path.basename(filename))
    os.makedirs(root, exist_ok=True)
    urlretrieve(filename, os.path.join(root, os.path.basename(filename)), show_progress)


def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    if not os.path.exists(path):
        download(root, 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root):
    # Load the fixed dataset
    path = os.path.join(root, 'mnist_test_seq.npy')
    if not os.path.exists(path):
        download(root, 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy')
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train, n_frames, num_digits, transform=transforms.Compose([ToTensor()]), step_length=0.1, with_target=False):
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_digits != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_digits = num_digits
        self.n_frames = n_frames
        self.transform = transform
        self.with_target = with_target

        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = step_length

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames):
                top    = start_y[i]
                left   = start_x[i]
                bottom = top + self.digit_size_
                right  = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]

        return data

    def __getitem__(self, idx):
        if self.is_train or self.num_digits != 2:
            # Generate data on the fly
            images = self.generate_moving_mnist(self.num_digits)
        else:
            images = self.dataset[:, idx, ...]

        if self.with_target:
            targets = np.array(images > 127, dtype=float) * 255.0

        if self.transform is not None:
            images = self.transform(images)
            if self.with_target:
                targets = self.transform(targets)

        if self.with_target:
            return images, targets
        else:
            return images

    def __len__(self):
        return self.length
