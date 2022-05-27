import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN
from torch.utils.data import DataLoader

from datasets.celeba import CelebA
# from datasets.ffhq import FFHQ
from datasets.imagenet import ImageNetDataset
from datasets.moving_mnist import MovingMNIST
from datasets.stochastic_moving_mnist import StochasticMovingMNIST
from datasets.bair import BAIRDataset
from datasets.kth import KTHDataset
from datasets.cityscapes import CityscapesDataset
from datasets.ucf101 import UCF101Dataset
from torch.utils.data import Subset


DATASETS = ['CIFAR10', 'CELEBA', 'LSUN', 'FFHQ', 'IMAGENET', 'MOVINGMNIST', 'STOCHASTICMOVINGMNIST', 'BAIR', 'KTH', 'CITYSCAPES', 'UCF101']


def get_dataloaders(data_path, config):
    dataset, test_dataset = get_dataset(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                            num_workers=config.data.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=True,
                             num_workers=config.data.num_workers, drop_last=True)
    return dataloader, test_loader


def get_dataset(data_path, config, video_frames_pred=0, start_at=0):

    assert config.data.dataset.upper() in DATASETS, \
        f"datasets/__init__.py: dataset can only be in {DATASETS}! Given {config.data.dataset.upper()}"

    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset.upper() == 'CIFAR10':
        dataset = CIFAR10(data_path, train=True, download=True,
                          transform=tran_transform)
        test_dataset = CIFAR10(data_path, train=False, download=True,
                               transform=test_transform)

    elif config.data.dataset.upper() == 'CELEBA':
        if config.data.random_flip:
            dataset = CelebA(
                             root=data_path, split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(
                             root=data_path, split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(
                              root=data_path, split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)


    elif config.data.dataset.upper() == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(
                            root=data_path, classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(
                            root=data_path, classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(
                            root=data_path, classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset.upper() == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=data_path,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(
                # path=os.path.join(args.exp, 'datasets', 'FFHQ'),
                path=data_path,
                transform=transforms.ToTensor(),
                resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    elif config.data.dataset.upper() == "IMAGENET":
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                                            transforms.RandomResizedCrop(224 if config.data.image_size < 256 else 256),
                                            transforms.Resize(config.data.image_size),
                                            transforms.RandomHorizontalFlip(p=(0.5 if config.data.random_flip else 0.0)),
                                            transforms.ToTensor(),
                                            # normalize,
                                            ])
        val_transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.Resize(config.data.image_size),
                                            transforms.ToTensor(),
                                            # normalize,
                                            ])
        dataset = ImageNetDataset(data_path, "train", train_transform, config.data.classes)
        test_dataset = ImageNetDataset(data_path, "val", val_transform, config.data.classes)

    elif config.data.dataset.upper() == "MOVINGMNIST":
        n_frames = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
        dataset = MovingMNIST(data_path, is_train=True, n_frames=n_frames, num_digits=getattr(config.data, "num_digits", 2),
                              step_length=config.data.step_length, with_target=True)
        test_dataset = MovingMNIST(data_path, is_train=False, n_frames=n_frames, num_digits=getattr(config.data, "num_digits", 2),
                                   step_length=config.data.step_length, with_target=True)

    elif config.data.dataset.upper() == "STOCHASTICMOVINGMNIST":
        seq_len = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
        dataset = StochasticMovingMNIST(data_path, train=True, seq_len=seq_len, num_digits=getattr(config.data, "num_digits", 2),
                                        step_length=config.data.step_length, with_target=True)
        test_dataset = StochasticMovingMNIST(data_path, train=False, seq_len=seq_len, num_digits=getattr(config.data, "num_digits", 2),
                                             step_length=config.data.step_length, with_target=True, total_videos=256)

    elif config.data.dataset.upper() == "BAIR":
        # BAIR_h5 (data_path)
        # |-- train
        # |   |-- shard_0001.hdf5
        # |-- test
        # |   |-- shard_0001.hdf5
        frames_per_sample = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
        dataset = BAIRDataset(os.path.join(data_path, "train"), frames_per_sample=frames_per_sample, random_time=True,
                              random_horizontal_flip=config.data.random_flip, color_jitter=getattr(config.data, 'color_jitter', 0.0))
        test_dataset = BAIRDataset(os.path.join(data_path, "test"), frames_per_sample=frames_per_sample, random_time=True,
                                   random_horizontal_flip=False, color_jitter=0.0)

    elif config.data.dataset.upper() == "KTH":
        # KTH64_h5 (data_path)
        # |-- shard_0001.hdf5
        frames_per_sample = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
        dataset = KTHDataset(data_path, frames_per_sample=frames_per_sample, train=True,
                             random_time=True, random_horizontal_flip=config.data.random_flip)
        test_dataset = KTHDataset(data_path, frames_per_sample=frames_per_sample, train=False,
                                  random_time=True, random_horizontal_flip=False, total_videos=256, start_at=start_at)

    elif config.data.dataset.upper() == "CITYSCAPES":
        # Cityscapes_h5 (data_path)
        # |-- train
        # |   |-- shard_0001.hdf5
        # |-- test
        # |   |-- shard_0001.hdf5
        frames_per_sample = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
        dataset = CityscapesDataset(os.path.join(data_path, "train"), frames_per_sample=frames_per_sample, random_time=True,
                                    random_horizontal_flip=config.data.random_flip, color_jitter=getattr(config.data, 'color_jitter', 0.0))
        test_dataset = CityscapesDataset(os.path.join(data_path, "test"), frames_per_sample=frames_per_sample, random_time=True,
                                         random_horizontal_flip=False, color_jitter=0.0, total_videos=256)

    elif config.data.dataset.upper() == "UCF101":
        # UCF101_h5 (data_path)
        # |-- shard_0001.hdf5
        frames_per_sample = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
        dataset = UCF101Dataset(data_path, frames_per_sample=frames_per_sample, image_size=config.data.image_size, train=True, random_time=True,
                                random_horizontal_flip=config.data.random_flip)
        test_dataset = UCF101Dataset(data_path, frames_per_sample=frames_per_sample, image_size=config.data.image_size, train=False, random_time=True,
                                     random_horizontal_flip=False, total_videos=256)

    subset_num = getattr(config.data, "subset", -1)
    if subset_num > 0:
        subset_indices = list(range(subset_num))
        dataset = Subset(dataset, subset_indices)

    test_subset_num = getattr(config.data, "test_subset", -1)
    if test_subset_num > 0:
        subset_indices = list(range(test_subset_num))
        test_dataset = Subset(test_dataset, subset_indices)

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
