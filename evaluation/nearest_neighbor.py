"""
prdc
Copyright (c) 2020-present NAVER Corp.
Modified by Yang Song (yangsong@cs.stanford.edu)
MIT license
"""
import sklearn.metrics
import pathlib

import numpy as np
import torch
from torchvision.datasets import LSUN, CelebA, CIFAR10
from datasets.ffhq import FFHQ
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, RandomHorizontalFlip, ToPILImage, ToTensor
from torchvision.utils import save_image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--k', type=int, default=9)
parser.add_argument('--n_samples', type=int, default=10)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('-i', type=str, required=True)

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from evaluation.inception import InceptionV3

def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def get_activations(model, images, dims=2048):
        # Reshape to (n_images, 3, height, width)
    with torch.no_grad():
        pred = model(images)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    return pred.reshape(pred.size(0), -1)


def _compute_features_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        act = f['features'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        act = get_activations(files, model, batch_size, dims, cuda, verbose=False)
    return act


def get_nearest_neighbors(dataset, path, name, n_samples, k=10, cuda=True):
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
        model.eval()

    flipper = RandomHorizontalFlip(p=1.)
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    data_features = []
    data = []
    for x, _ in tqdm(dataset, desc="sweeping the whole dataset"):
        if cuda: x = x.cuda()
        data_features.append(get_activations(model, x).cpu())
        data.append(x.cpu())

    data_features = torch.cat(data_features, dim=0)
    data = torch.cat(data, dim=0)

    samples = torch.load(path)[:n_samples]
    flipped_samples = torch.stack([to_tensor(flipper(to_pil(img))) for img in samples], dim=0)
    if cuda:
        samples = samples.cuda()
        flipped_samples = flipped_samples.cuda()

    sample_features = get_activations(model, samples).cpu()
    flip_sample_feature = get_activations(model, flipped_samples).cpu()
    sample_cdist = torch.cdist(sample_features, data_features)
    flip_sample_cdist = torch.cdist(flip_sample_feature, data_features)

    plot_data = []
    for i in tqdm(range(len(samples)), desc='find nns and save images'):
        plot_data.append(samples[i].cpu())
        all_dists = torch.min(sample_cdist[i], flip_sample_cdist[i])
        indices = torch.topk(-all_dists, k=k)[1]
        for ind in indices:
            plot_data.append(data[ind])

    plot_data = torch.stack(plot_data, dim=0)
    save_image(plot_data, '{}.png'.format(name), nrow=k+1)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == 'church':
        transforms = Compose([
            Resize(96),
            CenterCrop(96),
            ToTensor()
        ])
        dataset = LSUN('exp/datasets/lsun', ['church_outdoor_train'], transform=transforms)

    elif args.dataset == 'tower' or args.dataset == 'bedroom':
        transforms = Compose([
            Resize(128),
            CenterCrop(128),
            ToTensor()
        ])
        dataset = LSUN('exp/datasets/lsun', ['{}_train'.format(args.dataset)], transform=transforms)

    elif args.dataset == 'celeba':
        transforms = Compose([
            CenterCrop(140),
            Resize(64),
            ToTensor(),
        ])
        dataset = CelebA('exp/datasets/celeba', split='train', transform=transforms)

    elif args.dataset == 'cifar10':
        dataset = CIFAR10('exp/datasets/cifar10', train=True, transform=ToTensor())
    elif args.dataset == 'ffhq':
        dataset = FFHQ(path='exp/datasets/FFHQ', transform=ToTensor(), resolution=256)

    dataloader = DataLoader(dataset, batch_size=128, drop_last=False)
    get_nearest_neighbors(dataloader, args.path, args.i, args.n_samples, args.k, torch.cuda.is_available())
