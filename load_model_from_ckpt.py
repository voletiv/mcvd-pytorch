import argparse
import numpy as np
import os
import torch
import yaml

from collections import OrderedDict
from functools import partial
from imageio import mimwrite
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

try:
    from torchvision.transforms.functional import resize, InterpolationMode
    interp = InterpolationMode.NEAREST
except:
    from torchvision.transforms.functional import resize
    interp = 0

from datasets import get_dataset, data_transform, inverse_data_transform
from main import dict2namespace
from models import get_sigmas, anneal_Langevin_dynamics, anneal_Langevin_dynamics_consistent, ddpm_sampler, ddim_sampler, FPNDM_sampler
from models.ema import EMAHelper
from runners.ncsn_runner import get_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint.pt')
    parser.add_argument('--data_path', type=str, default='/mnt/data/scratch/data/CIFAR10', help='Path to the dataset')
    args = parser.parse_args()
    return args.ckpt_path, args.data_path


# Make and load model
def load_model(ckpt_path, device=device):
    # Parse config file
    with open(os.path.join(os.path.dirname(ckpt_path), 'config.yml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Load config file
    config = dict2namespace(config)
    config.device = device
    # Load model
    scorenet = get_model(config)
    if config.device != torch.device('cpu'):
        scorenet = torch.nn.DataParallel(scorenet)
        states = torch.load(ckpt_path, map_location=config.device)
    else:
        states = torch.load(ckpt_path, map_location='cpu')
        states[0] = OrderedDict([(k.replace('module.', ''), v) for k, v in states[0].items()])
    scorenet.load_state_dict(states[0], strict=False)
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(scorenet)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(scorenet)
    scorenet.eval()
    return scorenet, config


def get_sampler_from_config(config):
        version = getattr(config.model, 'version', "DDPM")
        # Sampler
        if version == "SMLD":
            consistent = getattr(config.sampling, 'consistent', False)
            sampler = anneal_Langevin_dynamics_consistent if consistent else anneal_Langevin_dynamics
        elif version == "DDPM":
            sampler = partial(ddpm_sampler, config=config)
        elif version == "DDIM":
            sampler = partial(ddim_sampler, config=config)
        elif version == "FPNDM":
            sampler = partial(FPNDM_sampler, config=config)
        return sampler


def get_sampler(config):
    sampler = get_sampler_from_config(config)
    sampler_partial = partial(sampler, n_steps_each=config.sampling.n_steps_each,
                    step_lr=config.sampling.step_lr, just_beta=False,
                    final_only=True, denoise=config.sampling.denoise,
                    subsample_steps=getattr(config.sampling, 'subsample', None),
                    clip_before=getattr(config.sampling, 'clip_before', True),
                    verbose=False, log=False, gamma=getattr(config.model, 'gamma', False))
    def sampler_fn(init, scorenet, cond, cond_mask, subsample=getattr(config.sampling, 'subsample', None), verbose=False):
        init = init.to(config.device)
        cond = cond.to(config.device)
        if cond_mask is not None:
            cond_mask = cond_mask.to(config.device)
        return inverse_data_transform(config, sampler_partial(init, scorenet, cond=cond, cond_mask=cond_mask,
                                                              subsample_steps=subsample, verbose=verbose)[-1].to('cpu'))
    return sampler_fn


def init_samples(n_init_samples, config):
    # Initial samples
    # n_init_samples = min(36, config.training.batch_size)
    version = getattr(config.model, 'version', "DDPM")
    init_samples_shape = (n_init_samples, config.data.channels*config.data.num_frames, config.data.image_size, config.data.image_size)
    if version == "SMLD":
        init_samples = torch.rand(init_samples_shape)
        init_samples = data_transform(self.config, init_samples)
    elif version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
        if getattr(config.model, 'gamma', False):
            used_k, used_theta = net.k_cum[0], net.theta_t[0]
            z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(config.device)
            init_samples = z - used_k*used_theta # we don't scale here
        else:
            init_samples = torch.randn(init_samples_shape)
    return init_samples


if __name__ == '__main__':
    # data_path = '/path/to/data/CIFAR10'
    ckpt_path, data_path = parse_args()

    scorenet, config = load_model(ckpt_path, device)

    # Initial samples
    dataset, test_dataset = get_dataset(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                            num_workers=config.data.num_workers)
    train_iter = iter(dataloader)
    x, y = next(train_iter)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False,
                             num_workers=config.data.num_workers, drop_last=True)
    test_iter = iter(test_loader)
    test_x, test_y = next(test_iter)

    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    version = getattr(net, 'version', 'SMLD').upper()
    net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

    if version == "SMLD":
        sigmas = net.sigmas
        labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
        used_sigmas = sigmas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        device = sigmas.device

    elif version == "DDPM" or version == "DDIM":
        alphas = net.alphas
        labels = torch.randint(0, len(alphas), (x.shape[0],), device=x.device)
        used_alphas = alphas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        device = alphas.device


# CUDA_VISIBLE_DEVICES=3 python -i load_model_from_ckpt.py --ckpt_path /path/to/ncsnv2/cifar10/BASELINE_DDPM_800k/logs/checkpoint.pt
