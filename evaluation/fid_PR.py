#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import os
import pathlib
import torch
import urllib

from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from .inception import InceptionV3


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('FID: fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activations(images, model, batch_size=50, dims=2048,
                          device=torch.device('cpu'), verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Tensor of images (n, 3, H, W), float values in [0, 1]
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(images):
        print(('FID: Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    pred_arr = torch.empty((len(images), dims))

    for i in tqdm(range(0, len(images), batch_size), leave=False, desc='InceptionV3'):
        if verbose:
            print('\rFID: Propagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i
        end = i + batch_size

        # images = np.array([imread(str(f)).astype(np.float32)
        #                    for f in files[start:end]])
        # # Reshape to (n_images, 3, height, width)
        # images = images.transpose((0, 3, 1, 2))
        # images /= 255
        # batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = images[start:end].to(device)

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().reshape(pred.size(0), -1)

    if verbose:
        print('FID: activations done')

    return pred_arr


def calculate_activation_statistics(images, model, batch_size=50,
                                    dims=2048, device=torch.device('cpu'), verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Tensor of images (n, 3, H, W), float values in [0, 1]
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = calculate_activations(images, model, batch_size, dims, device, verbose).data.numpy()
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path_or_samples(path_or_samples, model, batch_size, dims, device):
    if isinstance(path_or_samples, str):
        assert path_or_samples.endswith('.npz'), "path is not .npz!"
        f = np.load(path_or_samples)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        assert isinstance(path_or_samples, torch.Tensor), "sample is not tensor!"
        # path = pathlib.Path(path)
        # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(path_or_samples, model, batch_size, dims, device)
    return m, s


def calc_cdist(feat1, feat2, batch_size=10000):
    dists = []
    for feat2_batch in feat2.split(batch_size):
        dists.append(torch.cdist(feat1, feat2_batch).cpu())
    return torch.cat(dists, dim=1)


def calculate_precision_recall_part(feat_r, feat_g, k=3, batch_size=10000):
    # Precision
    NNk_r = []
    for feat_r_batch in feat_r.split(batch_size):
        NNk_r.append(calc_cdist(feat_r_batch, feat_r, batch_size).kthvalue(k+1).values)
    NNk_r = torch.cat(NNk_r)
    precision = []
    for feat_g_batch in feat_g.split(batch_size):
        dist_g_r_batch = calc_cdist(feat_g_batch, feat_r, batch_size)
        precision.append((dist_g_r_batch <= NNk_r).any(dim=1).float())
    precision = torch.cat(precision).mean().item()
    # Recall
    NNk_g = []
    for feat_g_batch in feat_g.split(batch_size):
        NNk_g.append(calc_cdist(feat_g_batch, feat_g, batch_size).kthvalue(k+1).values)
    NNk_g = torch.cat(NNk_g)
    recall = []
    for feat_r_batch in feat_r.split(batch_size):
        dist_r_g_batch = calc_cdist(feat_r_batch, feat_g, batch_size)
        recall.append((dist_r_g_batch <= NNk_g).any(dim=1).float())
    recall = torch.cat(recall).mean().item()
    return precision, recall


def calc_cdist_full(feat1, feat2, batch_size=10000):
    dists = []
    for feat1_batch in feat1.split(batch_size):
        dists_batch = []
        for feat2_batch in feat2.split(batch_size):
            dists_batch.append(torch.cdist(feat1_batch, feat2_batch).cpu())
        dists.append(torch.cat(dists_batch, dim=1))
    return torch.cat(dists, dim=0)


def calculate_precision_recall_full(feat_r, feat_g, k=3, batch_size=10000):
    NNk_r = calc_cdist_full(feat_r, feat_r, batch_size).kthvalue(k+1).values
    NNk_g = calc_cdist_full(feat_g, feat_g, batch_size).kthvalue(k+1).values
    dist_g_r = calc_cdist_full(feat_g, feat_r, batch_size)
    dist_r_g = dist_g_r.T
    # Precision
    precision = (dist_g_r <= NNk_r).any(dim=1).float().mean().item()
    # Recall
    recall = (dist_r_g <= NNk_g).any(dim=1).float().mean().item()
    return precision, recall


def calculate_precision_recall(feat_r, feat_g, device=torch.device('cuda'), k=3,
                               batch_size=10000, save_cpu_ram=False, **kwargs):
    feat_r = feat_r.to(device)
    feat_g = feat_g.to(device)
    if save_cpu_ram:
        return calculate_precision_recall_part(feat_r, feat_g, k, batch_size)
    else:
        return calculate_precision_recall_full(feat_r, feat_g, k, batch_size)


def get_activations(path_or_samples, model, batch_size, dims, device):
    if isinstance(path_or_samples, str):
        assert path_or_samples.endswith('.pt') or path_or_samples.endswith('.pth'), "path is not .pt or .pth!"
        act = torch.load(path_or_samples)
    else:
        assert isinstance(path_or_samples, torch.Tensor), "sample is not tensor!"
        act = calculate_activations(path_or_samples, model, batch_size, dims, device)
    return act


def get_fid_PR(real_path_or_samples, fake_path_or_samples, device=torch.device('cuda'),
               batch_size=50, dims=2048, k=3, save_feats_path=None):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    # Real
    feat_r = get_activations(real_path_or_samples, model, batch_size, dims, device)
    # Fake
    feat_g = get_activations(fake_path_or_samples, model, batch_size, dims, device)
    if save_feats_path is not None:
        torch.save(feat_g, save_feats_path)
    # PR
    precision, recall = calculate_precision_recall(feat_r, feat_g, device, k)
    # FID
    feat_r, feat_g = feat_r.data.numpy(), feat_g.data.numpy()
    mu_r, sigma_r = np.mean(feat_r, axis=0), np.cov(feat_r, rowvar=False)
    mu_g, sigma_g = np.mean(feat_g, axis=0), np.cov(feat_g, rowvar=False)
    fid_value = calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    return fid_value, precision, recall


def get_PR(real_path_or_samples, fake_path_or_samples, device=torch.device('cuda'), batch_size=50, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    # Real
    feat_r = get_activations(real_path_or_samples, model, batch_size, dims, device)
    # Fake
    feat_g = get_activations(fake_path_or_samples, model, batch_size, dims, device)
    # PR
    precision, recall = calculate_precision_recall(feat_r, feat_g)
    return precision, recall



def get_fid(path_or_samples1, path_or_samples2, device=torch.device('cuda'), batch_size=50, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = _compute_statistics_of_path_or_samples(path_or_samples1, model, batch_size, dims, device)
    m2, s2 = _compute_statistics_of_path_or_samples(path_or_samples2, model, batch_size, dims, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


STATS_LINKS = {
    'CIFAR10': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz',
    'LSUN': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz',
    'CELEBA': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz', # cropped CelebA 64x64
    'SVHN': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_svhn_train.npz',
    'IMAGENET_TRAIN': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_train.npz',
    'IMAGENET_VALID': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_valid.npz',
}

FEATS_PATHS = {
    'CIFAR10': 'cifar10-inception-v3-compat-features-2048.pt',
    'LSUN': 'lsun-inception-v3-compat-features-2048.pt',
    'CELEBA': 'celeba-inception-v3-compat-features-2048.pt', # cropped CelebA 64x64
    'SVHN': 'svhn-inception-v3-compat-features-2048.pt',
    'IMAGENET64': 'imagenet64-inception-v3-compat-features-2048.pt',
}


def get_stats_path(dataset, stats_dir, download=True):
    # dataset = getattr(config.fast_fid, 'dataset', config.data.dataset).upper()
    stats_npz_path = os.path.join(stats_dir, os.path.basename(STATS_LINKS[dataset]))
    if not os.path.exists(stats_npz_path):
        if not download:
            raise FileNotFoundError(f"Stats file not found! Required: {stats_npz_path}. Please download by setting '--stats_download'.")
        else:
            urllib.request.urlretrieve(STATS_LINKS[dataset], stats_npz_path)
    return stats_npz_path


def get_feats_path(dataset, feats_dir, download=False):
    feats_path = os.path.join(feats_dir, os.path.basename(FEATS_PATHS[dataset]))
    if not os.path.exists(feats_path):
        if not download:
            raise FileNotFoundError(f"Feats file not found! Required: {feats_path}")
        # else:
        #     urllib.request.urlretrieve(FEATS_LINKS[dataset], feats_path)
    return feats_path
