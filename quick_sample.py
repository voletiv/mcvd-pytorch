import argparse
import numpy as np
import os
import torch
import yaml

from collections import OrderedDict
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
from models import get_sigmas, anneal_Langevin_dynamics
from models.ema import EMAHelper
from runners.ncsn_runner import get_model, conditioning_fn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

from models import ddpm_sampler


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint.pt')
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    parser.add_argument('--save_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    return args.ckpt_path, args.data_path, args.save_path


# Make and load model
def load_model(ckpt_path, device):
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


if __name__ == '__main__':
    # data_path = '/path/to/data/CIFAR10'
    ckpt_path, data_path, save_path = parse_args()

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

    for batch, (X, y) in enumerate(dataloader):
        break

    X = X.to(config.device)
    X = data_transform(config, X)

    conditional = config.data.num_frames_cond > 0
    cond = None
    if conditional:
        X, cond = conditioning_fn(config, X)

    init_samples = torch.randn(len(X), config.data.channels*config.data.num_frames,
                               config.data.image_size, config.data.image_size,
                               device=config.device)

    all_samples = ddpm_sampler(init_samples, scorenet, cond=cond[:len(init_samples)],
                               n_steps_each=config.sampling.n_steps_each,
                               step_lr=config.sampling.step_lr, just_beta=False,
                               final_only=True, denoise=config.sampling.denoise,
                               subsample_steps=getattr(config.sampling, 'subsample', None),
                               verbose=True)

    sample = all_samples[-1].reshape(all_samples[-1].shape[0], config.data.channels,
                                     config.data.image_size, config.data.image_size)

    sample = inverse_data_transform(config, sample)

    image_grid = make_grid(sample, np.sqrt(config.training.batch_size))
    step = 0
    save_image(image_grid,
               os.path.join(save_path, 'image_grid_{}.png'.format(step)))
    torch.save(sample, os.path.join(save_path, 'samples_{}.pt'.format(step)))

    # CUDA_VISIBLE_DEVICES=3 python -i load_model_from_ckpt.py --ckpt_path /path/to/ncsnv2/cifar10/BASELINE_DDPM_800k/logs/checkpoint.pt
