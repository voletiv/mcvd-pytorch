
# CUDA_VISIBLE_DEVICES=3 python -i load_model_from_ckpt.py --ckpt_path /path/to/logs/checkpoint.pt

# Load CIFAR10
import torch
from torchvision.datasets import CIFAR10
ds = CIFAR10('/path/to/data/cifar10', train=True)
data = torch.from_numpy(ds.data)

# Transform data
from datasets import get_dataset, data_transform, inverse_data_transform

# Sampler
from models import ddpm_sampler


all_samples = ddpm_sampler(init_samples, scorenet,
                           n_steps_each=config.sampling.n_steps_each,
                           step_lr=config.sampling.step_lr, verbose=True,
                           final_only=config.sampling.final_only,
                           denoise=config.sampling.denoise,
                           subsample_steps=getattr(config.sampling, 'subsample', None))
