import datetime
import logging
import imageio
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import psutil
import scipy.stats as st
import sys
import time
import yaml

import torch
import torch.nn.functional as F
import torchvision.transforms as Transforms

from cv2 import rectangle, putText
from functools import partial
from math import ceil, log10
from multiprocessing import Process
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from torch.distributions.gamma import Gamma
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resized_crop
from torchvision.utils import make_grid, save_image

import models.eval_models as eval_models

from datasets import get_dataset, data_transform, inverse_data_transform
from datasets.ffhq_tfrecords import FFHQ_TFRecordsDataLoader
from evaluation.fid_PR import get_fid, get_fid_PR, get_stats_path, get_feats_path
from losses import get_optimizer, warmup_lr
from losses.dsm import anneal_dsm_score_estimation
from models import (ddpm_sampler,
                    ddim_sampler,
                    FPNDM_sampler,
                    anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_consistent,
                    anneal_Langevin_dynamics_inpainting,
                    anneal_Langevin_dynamics_interpolation)
from models.ema import EMAHelper
from models.fvd.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
from models.unet import UNet_SMLD, UNet_DDPM
#import pdb; pdb.set_trace()

__all__ = ['NCSNRunner']


def count_training_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_proc_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    try:
        num = torch.cuda.device_count()
        mem = 0
        for i in range(num):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            mem += (mem_total - mem_free)/1024**3
        return mem
    except:
        return 0


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99, save_seq=True):
        self.momentum = momentum
        self.save_seq = save_seq
        if self.save_seq:
            self.vals, self.steps = [], []
        self.reset()

    def reset(self):
        self.val, self.avg = None, 0

    def update(self, val, step=None):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if step is not None:
                self.steps.append(step)


def conditioning_fn(config, X, num_frames_pred=0, prob_mask_cond=0.0, prob_mask_future=0.0, conditional=True):
    imsize = config.data.image_size
    if not conditional:
        return X.reshape(len(X), -1, imsize, imsize), None, None

    cond = config.data.num_frames_cond
    train = config.data.num_frames
    pred = num_frames_pred
    future = getattr(config.data, "num_frames_future", 0)

    # Frames to train on / sample
    pred_frames = X[:, cond:cond+pred].reshape(len(X), -1, imsize, imsize)

    # Condition (Past)
    cond_frames = X[:, :cond].reshape(len(X), -1, imsize, imsize)

    if prob_mask_cond > 0.0:
        cond_mask = (torch.rand(X.shape[0], device=X.device) > prob_mask_cond)
        cond_frames = cond_mask.reshape(-1, 1, 1, 1) * cond_frames
        cond_mask = cond_mask.to(torch.int32) # make 0,1
    else:
        cond_mask = None

    # Future
    if future > 0:

        if prob_mask_future == 1.0:
            future_frames = torch.zeros(len(X), config.data.channels*future, imsize, imsize)
            # future_mask = torch.zeros(len(X), 1, 1, 1).to(torch.int32) # make 0,1
        else:
            future_frames = X[:, cond+train:cond+train+future].reshape(len(X), -1, imsize, imsize)
            if prob_mask_future > 0.0:
                if getattr(config.data, "prob_mask_sync", False):
                    future_mask = cond_mask
                else:
                    future_mask = (torch.rand(X.shape[0], device=X.device) > prob_mask_future)
                future_frames = future_mask.reshape(-1, 1, 1, 1) * future_frames
            #     future_mask = future_mask.to(torch.int32) # make 0,1
            # else:
            #     future_mask = None

        cond_frames = torch.cat([cond_frames, future_frames], dim=1)

    return pred_frames, cond_frames, cond_mask   # , future_mask


def stretch_image(X, ch, imsize):
    return X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 2, 1, 4, 3).reshape(len(X), ch, -1, imsize).permute(0, 1, 3, 2)


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


def get_model(config):

    version = getattr(config.model, 'version', 'SMLD').upper()
    arch = getattr(config.model, 'arch', 'ncsn')
    depth = getattr(config.model, 'depth', 'deep')

    if arch == 'unetmore':
        from models.better.ncsnpp_more import UNetMore_DDPM # This lets the code run on CPU when 'unetmore' is not used
        return UNetMore_DDPM(config).to(config.device)#.to(memory_format=torch.channels_last).to(config.device)
    elif arch in ['unetmore3d', 'unetmorepseudo3d']:
        from models.better.ncsnpp_more import UNetMore_DDPM # This lets the code run on CPU when 'unetmore' is not used
        # return UNetMore_DDPM(config).to(memory_format=torch.channels_last_3d).to(config.device) # channels_last_3d doesn't work!
        return UNetMore_DDPM(config).to(config.device)#.to(memory_format=torch.channels_last).to(config.device)

    else:
        Exception("arch is not valid [ncsn, unet, unetmore, unetmore3d]")

class NCSNRunner():
    def __init__(self, args, config, config_uncond):
        self.args = args
        self.config = config
        self.config_uncond = config_uncond
        self.version = getattr(self.config.model, 'version', "SMLD")
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)
        self.get_mode()

    def get_mode(self):
        self.condf, self.condp = self.config.data.num_frames_cond, getattr(self.config.data, "prob_mask_cond", 0.0)
        self.futrf, self.futrp = getattr(self.config.data, "num_frames_future", 0), getattr(self.config.data, "prob_mask_future", 0.0)
        self.prob_mask_sync = getattr(self.config.data, "prob_mask_sync", False)
        if not getattr(self.config.sampling, "ssim", False):
            if getattr(self.config.sampling, "fvd", False):
                self.mode_pred, self.mode_interp, self.mode_gen = None, None, "three"
            else:
                self.mode_pred, self.mode_interp, self.mode_gen = None, None, None
        elif self.condp == 0.0 and self.futrf == 0:                                                   # (1) Prediction
            self.mode_pred, self.mode_interp, self.mode_gen = "one", None, None
        elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:                            # (1) Interpolation
            self.mode_pred, self.mode_interp, self.mode_gen = None, "one", None
        elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:                             # (1) Interp + (2) Pred
            self.mode_pred, self.mode_interp, self.mode_gen = "two", "one", None
        elif self.condp > 0.0 and self.futrf == 0:                                                  # (1) Pred + (3) Gen
            self.mode_pred, self.mode_interp, self.mode_gen = "one", None, "three"
        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:  # (1) Interp + (2) Pred + (3) Gen
            self.mode_pred, self.mode_interp, self.mode_gen = "two", "one", "three"
        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:      # (1) Interp + (3) Gen
            self.mode_pred, self.mode_interp, self.mode_gen = None, "one", "three"

    def get_time(self):
        curr_time = time.time()
        curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
        elapsed = str(datetime.timedelta(seconds=(curr_time - self.start_time)))
        return curr_time_str, elapsed

    def convert_time_stamp_to_hrs(self, time_day_hr):
        time_day_hr = time_day_hr.split(",")
        if len(time_day_hr) > 1:
            days = time_day_hr[0].split(" ")[0]
            time_hr = time_day_hr[1]
        else:
            days = 0
            time_hr = time_day_hr[0]
        # Hr
        hrs = time_hr.split(":")
        return float(days)*24 + float(hrs[0]) + float(hrs[1])/60 + float(hrs[2])/3600

    def train(self):
        # If FFHQ tfrecord, reset dataloader
        if self.config.data.dataset.upper() == 'FFHQ':
            dataloader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.training.batch_size, self.config.data.image_size)
            test_loader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.training.batch_size, self.config.data.image_size)
            test_iter = iter(test_loader)
        else:
            dataset, test_dataset = get_dataset(self.args.data_path, self.config, video_frames_pred=self.config.data.num_frames, start_at=self.args.start_at)
            dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                    num_workers=self.config.data.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, drop_last=True)
            test_iter = iter(test_loader)

        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        # tb_logger = self.config.tb_logger

        scorenet = get_model(self.config)
        scorenet = torch.nn.DataParallel(scorenet)

        logging.info(f"Number of parameters: {count_parameters(scorenet)}")
        logging.info(f"Number of trainable parameters: {count_training_parameters(scorenet)}")

        optimizer = get_optimizer(self.config, scorenet.parameters())

        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            logging.info(f"Number of GPUs : {num_devices}")
            for i in range(num_devices):
                logging.info(torch.cuda.get_device_properties(i))
        else:
            logging.info(f"Running on CPU!")

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pt'))
            scorenet.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
            logging.info(f"Resuming training from checkpoint.pt in {self.args.log_path} at epoch {start_epoch}, step {step}.")

        if self.config.training.log_all_sigmas:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(getattr(self.config.model, 'num_classes'))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(getattr(self.config.model, 'num_classes')):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(getattr(self.config.model, 'num_classes')):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        print(scorenet)
        net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Conditional
        conditional = self.config.data.num_frames_cond > 0
        cond, test_cond = None, None

        # Future
        future = getattr(self.config.data, "num_frames_future", 0)

        # Initialize meters
        self.init_meters()

        # Initial samples
        n_init_samples = min(36, self.config.training.batch_size)
        init_samples_shape = (n_init_samples, self.config.data.channels*self.config.data.num_frames, self.config.data.image_size, self.config.data.image_size)
        if self.version == "SMLD":
            init_samples = torch.rand(init_samples_shape, device=self.config.device)
            init_samples = data_transform(self.config, init_samples)
        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
            if getattr(self.config.model, 'gamma', False):
                used_k, used_theta = net.k_cum[0], net.theta_t[0]
                z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                init_samples = z - used_k*used_theta # we don't scale here
            else:
                init_samples = torch.randn(init_samples_shape, device=self.config.device)

        # Sampler
        sampler = self.get_sampler()

        self.total_train_time = 0
        self.start_time = time.time()

        early_end = False
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for batch, (X, y) in enumerate(dataloader):

                optimizer.zero_grad()
                lr = warmup_lr(optimizer, step, getattr(self.config.optim, 'warmup', 0), self.config.optim.lr)
                scorenet.train()
                step += 1

                # Data
                X = X.to(self.config.device)
                X = data_transform(self.config, X)
                X, cond, cond_mask = conditioning_fn(self.config, X, num_frames_pred=self.config.data.num_frames,
                                                     prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                     prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0),
                                                     conditional=conditional)

                # Loss
                itr_start = time.time()
                loss = anneal_dsm_score_estimation(scorenet, X, labels=None, cond=cond, cond_mask=cond_mask,
                                                   loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                   gamma=getattr(self.config.model, 'gamma', False),
                                                   L1=getattr(self.config.training, 'L1', False), hook=hook,
                                                   all_frames=getattr(self.config.model, 'output_all_frames', False))
                # tb_logger.add_scalar('loss', loss, global_step=step)
                # tb_hook()

                # Optimize
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(scorenet.parameters(), getattr(self.config.optim, 'grad_clip', np.inf))
                optimizer.step()

                # Training time
                itr_time = time.time() - itr_start
                self.total_train_time += itr_time
                self.time_train.update(self.convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=self.total_train_time))) + self.time_train_prev)

                # Record
                self.losses_train.update(loss.item(), step)
                self.epochs.update(epoch + (batch + 1)/len(dataloader))
                self.lr_meter.update(lr)
                self.grad_norm.update(grad_norm.item())
                if step == 1 or step % getattr(self.config.training, "log_freq", 1) == 0:
                    logging.info("elapsed: {}, train time: {:.04f}, mem: {:.03f}GB, GPUmem: {:.03f}GB, step: {}, lr: {:.06f}, grad: {:.04f}, loss: {:.04f}".format(
                        str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3],
                        self.time_train.val, get_proc_mem(), get_GPU_mem(), step, lr, grad_norm, loss.item()))

                if self.config.model.ema:
                    ema_helper.update(scorenet)

                if step >= self.config.training.n_iters:
                    early_end = True
                    break

                # Save model
                if (step % 1000 == 0 and step != 0) or step % self.config.training.snapshot_freq == 0:
                    states = [
                        scorenet.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    logging.info(f"Saving checkpoint.pt in {self.args.log_path}")
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pt'))
                    if step % self.config.training.snapshot_freq == 0:
                        ckpt_path = os.path.join(self.args.log_path, 'checkpoint_{}.pt'.format(step))
                        logging.info(f"Saving {ckpt_path}")
                        torch.save(states, ckpt_path)

                test_scorenet = None
                # Get test_scorenet
                if step == 1 or step % self.config.training.val_freq == 0 or (step % self.config.training.snapshot_freq == 0 or step % self.config.training.sample_freq == 0) and self.config.training.snapshot_sampling:

                    if self.config.model.ema:
                        test_scorenet = ema_helper.ema_copy(scorenet)
                    else:
                        test_scorenet = scorenet

                    test_scorenet.eval()

                # Validation
                if step == 1 or step % self.config.training.val_freq == 0:
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    test_X, test_cond, test_cond_mask = conditioning_fn(self.config, test_X, num_frames_pred=self.config.data.num_frames,
                                                                        prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                                        prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0),
                                                                        conditional=conditional)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(test_scorenet, test_X, labels=None, cond=test_cond, cond_mask=test_cond_mask,
                                                                    loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                                    gamma=getattr(self.config.model, 'gamma', False),
                                                                    L1=getattr(self.config.training, 'L1', False), hook=test_hook,
                                                                    all_frames=getattr(self.config.model, 'output_all_frames', False))
                    # tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                    # test_tb_hook()
                    self.losses_test.update(test_dsm_loss.item(), step)
                    logging.info("elapsed: {}, step: {}, mem: {:.03f}GB, GPUmem: {:.03f}GB, test_loss: {:.04f}".format(
                        str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3],
                        step, get_proc_mem(), get_GPU_mem(), test_dsm_loss.item()))

                    # Plot graphs
                    try:
                        plot_graphs_process.join()
                    except:
                        pass
                    plot_graphs_process = Process(target=self.plot_graphs)
                    plot_graphs_process.start()

                # Sample from model
                if (step % self.config.training.snapshot_freq == 0 or step % self.config.training.sample_freq == 0) and self.config.training.snapshot_sampling:

                    logging.info(f"Saving images in {self.args.log_sample_path}")

                    # Calc video metrics with max_data_iter=1
                    if conditional and step % self.config.training.snapshot_freq == 0 and self.config.training.snapshot_sampling: # only at snapshot_freq, not at sample_freq

                        vid_metrics = self.video_gen(scorenet=test_scorenet, ckpt=step, train=True)

                        if 'mse' in vid_metrics.keys():
                            self.mses.update(vid_metrics['mse'], step)
                            self.psnrs.update(vid_metrics['psnr'])
                            self.ssims.update(vid_metrics['ssim'])
                            self.lpipss.update(vid_metrics['lpips'])
                            if vid_metrics['mse'] < self.best_mse['mse']:
                                self.best_mse = vid_metrics
                            if vid_metrics['psnr'] > self.best_psnr['psnr']:
                                self.best_psnr = vid_metrics
                            if vid_metrics['ssim'] > self.best_ssim['ssim']:
                                self.best_ssim = vid_metrics
                            if vid_metrics['lpips'] < self.best_lpips['lpips']:
                                self.best_lpips = vid_metrics
                            if self.calc_fvd1:
                                self.fvds.update(vid_metrics['fvd'])
                                if vid_metrics['fvd'] < self.best_fvd['fvd']:
                                    self.best_fvd = vid_metrics

                        if 'mse2' in vid_metrics.keys():
                            self.mses2.update(vid_metrics['mse2'], step)
                            self.psnrs2.update(vid_metrics['psnr2'])
                            self.ssims2.update(vid_metrics['ssim2'])
                            self.lpipss2.update(vid_metrics['lpips2'])
                            if vid_metrics['mse2'] < self.best_mse2['mse2']:
                                self.best_mse2 = vid_metrics
                            if vid_metrics['psnr2'] > self.best_psnr2['psnr2']:
                                self.best_psnr2 = vid_metrics
                            if vid_metrics['ssim2'] > self.best_ssim2['ssim2']:
                                self.best_ssim2 = vid_metrics
                            if vid_metrics['lpips2'] < self.best_lpips2['lpips2']:
                                self.best_lpips2 = vid_metrics
                            if self.calc_fvd2:
                                self.fvds2.update(vid_metrics['fvd2'])
                                if vid_metrics['fvd2'] < self.best_fvd2['fvd2']:
                                    self.best_fvd2 = vid_metrics

                        if self.calc_fvd3:
                            self.fvds3.update(vid_metrics['fvd3'], step)
                            if vid_metrics['fvd3'] < self.best_fvd3['fvd3']:
                                self.best_fvd3 = vid_metrics

                        # Show best results for every metric

                        if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
                            self.mses_pred, self.psnrs_pred, self.ssims_pred, self.lpipss_pred, self.fvds_pred = self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds
                            self.best_mse_pred, self.best_psnr_pred, self.best_ssim_pred, self.best_lpips_pred, self.best_fvd_pred, self.calc_fvd_pred = self.best_mse, self.best_psnr, self.best_ssim, self.best_lpips, self.best_fvd, self.calc_fvd1
                        elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation
                            self.mses_interp, self.psnrs_interp, self.ssims_interp, self.lpipss_interp, self.fvds_interp = self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds
                            self.best_mse_interp, self.best_psnr_interp, self.best_ssim_interp, self.best_lpips_interp, self.best_fvd_interp, self.calc_fvd_interp = self.best_mse, self.best_psnr, self.best_ssim, self.best_lpips, self.best_fvd, self.calc_fvd1
                        elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred
                            self.mses_interp, self.psnrs_interp, self.ssims_interp, self.lpipss_interp, self.fvds_interp = self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds
                            self.mses_pred, self.psnrs_pred, self.ssims_pred, self.lpipss_pred, self.fvds_pred = self.mses2, self.psnrs2, self.ssims2, self.lpipss2, self.fvds2
                            self.best_mse_interp, self.best_psnr_interp, self.best_ssim_interp, self.best_lpips_interp, self.best_fvd_interp, self.calc_fvd_interp = self.best_mse, self.best_psnr, self.best_ssim, self.best_lpips, self.best_fvd, self.calc_fvd1
                            self.best_mse_pred, self.best_psnr_pred, self.best_ssim_pred, self.best_lpips_pred, self.best_fvd_pred, self.calc_fvd_pred = self.best_mse2, self.best_psnr2, self.best_ssim2, self.best_lpips2, self.best_fvd2, self.calc_fvd2
                        elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen
                            self.mses_pred, self.psnrs_pred, self.ssims_pred, self.lpipss_pred, self.fvds_pred = self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds
                            self.best_mse_pred, self.best_psnr_pred, self.best_ssim_pred, self.best_lpips_pred, self.best_fvd_pred, self.calc_fvd_pred = self.best_mse, self.best_psnr, self.best_ssim, self.best_lpips, self.best_fvd, self.calc_fvd1
                        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:     # (1) Interp + (2) Pred + (3) Gen
                            self.mses_interp, self.psnrs_interp, self.ssims_interp, self.lpipss_interp, self.fvds_interp = self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds
                            self.mses_pred, self.psnrs_pred, self.ssims_pred, self.lpipss_pred, self.fvds_pred = self.mses2, self.psnrs2, self.ssims2, self.lpipss2, self.fvds2
                            self.best_mse_interp, self.best_psnr_interp, self.best_ssim_interp, self.best_lpips_interp, self.best_fvd_interp, self.calc_fvd_interp = self.best_mse, self.best_psnr, self.best_ssim, self.best_lpips, self.best_fvd, self.calc_fvd1
                            self.best_mse_pred, self.best_psnr_pred, self.best_ssim_pred, self.best_lpips_pred, self.best_fvd_pred, self.calc_fvd_pred = self.best_mse2, self.best_psnr2, self.best_ssim2, self.best_lpips2, self.best_fvd2, self.calc_fvd2
                        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:  # (1) Interp + (3) Gen
                            self.mses_interp, self.psnrs_interp, self.ssims_interp, self.lpipss_interp, self.fvds_interp = self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds
                            self.best_mse_interp, self.best_psnr_interp, self.best_ssim_interp, self.best_lpips_interp, self.best_fvd_interp, self.calc_fvd_interp = self.best_mse, self.best_psnr, self.best_ssim, self.best_lpips, self.best_fvd, self.calc_fvd1

                        format_p = lambda dd : ", ".join([f"{k}:{v:.4f}" if k != 'ckpt' and k != 'preds_per_test' else f"{k}:{v:7d}" if k == 'ckpt' else f"{k}:{v:3d}" for k, v in dd.items()])
                        if self.mode_pred is not None:
                            logging.info(f"PRED: {self.mode_pred}")
                            logging.info(f"Best-MSE   pred - {format_p(self.best_mse_pred)}")
                            logging.info(f"Best-PSNR  pred - {format_p(self.best_psnr_pred)}")
                            logging.info(f"Best-SSIM  pred - {format_p(self.best_ssim_pred)}")
                            logging.info(f"Best-LPIPS pred - {format_p(self.best_lpips_pred)}")
                            if self.calc_fvd_pred:
                                logging.info(f"Best-FVD   pred - {format_p(self.best_fvd_pred)}")
                        if self.mode_interp is not None:
                            logging.info(f"INTERPOLATION: {self.mode_interp}")
                            logging.info(f"Best-MSE   interp - {format_p(self.best_mse_interp)}")
                            logging.info(f"Best-PSNR  interp - {format_p(self.best_psnr_interp)}")
                            logging.info(f"Best-SSIM  interp - {format_p(self.best_ssim_interp)}")
                            logging.info(f"Best-LPIPS interp - {format_p(self.best_lpips_interp)}")
                            if self.calc_fvd_interp:
                                logging.info(f"Best-FVD   interp - {format_p(self.best_fvd_interp)}")
                        if self.mode_gen is not None and self.calc_fvd3:
                            logging.info(f"GENERATION: {self.mode_gen}")
                            logging.info(f"Best-FVD gen  - {format_p(self.best_fvd3)}")

                        # Plot video graphs
                        try:
                            plot_video_graphs_process.join()
                        except:
                            pass
                        plot_video_graphs_process = Process(target=self.plot_video_graphs)
                        plot_video_graphs_process.start()

                    # Samples
                    if conditional:
                        try:
                            test_X, test_y = next(test_iter)
                        except StopIteration:
                            test_iter = iter(test_loader)
                            test_X, test_y = next(test_iter)
                        test_X = test_X[:len(init_samples)].to(self.config.device)
                        test_X = data_transform(self.config, test_X)
                        test_X, test_cond, test_cond_mask = conditioning_fn(self.config, test_X, num_frames_pred=self.config.data.num_frames,
                                                                            prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                                            prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0),
                                                                            conditional=conditional)

                    all_samples = sampler(init_samples, test_scorenet, cond=test_cond, cond_mask=test_cond_mask,
                                          n_steps_each=self.config.sampling.n_steps_each,
                                          step_lr=self.config.sampling.step_lr, just_beta=False,
                                          final_only=True, denoise=self.config.sampling.denoise,
                                          subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                          clip_before=getattr(self.config.sampling, 'clip_before', True),
                                          verbose=False, log=False, gamma=getattr(self.config.model, 'gamma', False)).to('cpu')
                    pred = all_samples[-1].reshape(all_samples[-1].shape[0], self.config.data.channels*self.config.data.num_frames,
                                                   self.config.data.image_size, self.config.data.image_size)
                    pred = inverse_data_transform(self.config, pred)

                    if conditional:

                        reali = inverse_data_transform(self.config, test_X.to('cpu'))
                        condi = inverse_data_transform(self.config, test_cond.to('cpu'))
                        if future > 0:
                            condi, futri = condi[:, :self.config.data.num_frames_cond*self.config.data.channels], condi[:, self.config.data.num_frames_cond*self.config.data.channels:]

                        # Save gif
                        gif_frames = []
                        for t in range(condi.shape[1]//self.config.data.channels):
                            cond_t = condi[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                            frame = torch.cat([cond_t, 0.5*torch.ones(*cond_t.shape[:-1], 2), cond_t], dim=-1)
                            frame = frame.permute(0, 2, 3, 1).numpy()
                            frame = np.stack([putText(f.copy(), f"{t+1:2d}p", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                            nrow = ceil(np.sqrt(2*condi.shape[0])/2)
                            gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                            gif_frames.append((gif_frame*255).astype('uint8'))
                            if t == 0:
                                gif_frames.append((gif_frame*255).astype('uint8'))
                            del frame, gif_frame
                        for t in range(pred.shape[1]//self.config.data.channels):
                            real_t = reali[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                            pred_t = pred[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                            frame = torch.cat([real_t, 0.5*torch.ones(*pred_t.shape[:-1], 2), pred_t], dim=-1)
                            frame = frame.permute(0, 2, 3, 1).numpy()   # BHWC
                            frame = np.stack([putText(f.copy(), f"{t+1:02d}", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                            nrow = ceil(np.sqrt(2*pred.shape[0])/2)
                            gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                            gif_frames.append((gif_frame*255).astype('uint8'))
                            if t == pred.shape[1]//self.config.data.channels - 1 and future == 0:
                                gif_frames.append((gif_frame*255).astype('uint8'))
                            del frame, gif_frame
                        if future > 0:
                            for t in range(futri.shape[1]//self.config.data.channels):
                                futr_t = futri[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                                frame = torch.cat([futr_t, 0.5*torch.ones(*futr_t.shape[:-1], 2), futr_t], dim=-1)
                                frame = frame.permute(0, 2, 3, 1).numpy()
                                frame = np.stack([putText(f.copy(), f"{t+1:2d}f", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                                nrow = ceil(np.sqrt(2*condi.shape[0])/2)
                                gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                                gif_frames.append((gif_frame*255).astype('uint8'))
                                if t == futri.shape[1]//self.config.data.channels - 1:
                                    gif_frames.append((gif_frame*255).astype('uint8'))
                                del frame, gif_frame

                        # Save gif
                        imageio.mimwrite(os.path.join(self.args.log_sample_path, f"video_grid_{step}.gif"), gif_frames, fps=4)
                        del gif_frames

                        # Stretch out multiple frames horizontally
                        pred = stretch_image(pred, self.config.data.channels, self.config.data.image_size)
                        reali = stretch_image(reali, self.config.data.channels, self.config.data.image_size)
                        condi = stretch_image(condi, self.config.data.channels, self.config.data.image_size)
                        if future > 0:
                            futri = stretch_image(futri, self.config.data.channels, self.config.data.image_size)

                        padding = 0.5 * torch.ones(len(reali), self.config.data.channels, self.config.data.image_size, 4)
                        if self.config.data.channels == 1:
                            data = torch.cat([condi, padding, reali, padding, pred], dim=-1)
                            if future > 0:
                                data = torch.cat([data, padding, futri], dim=-1)
                        else:
                            padding_red, padding_green = torch.ones(len(reali), self.config.data.channels, self.config.data.image_size, 4), torch.ones(len(reali), self.config.data.channels, self.config.data.image_size, 4)
                            padding_red[:, [1, 2]], padding_green[:, [0, 2]] = 0, 0
                            data = torch.cat([condi, padding_green, reali, padding_green, padding_red, pred, padding_red], dim=-1)
                            if future > 0:
                                data = torch.cat([data, futri], dim=-1)

                        nrow = ceil(np.sqrt((self.config.data.num_frames_cond+self.config.data.num_frames*2+future)*n_init_samples)/(self.config.data.num_frames_cond+self.config.data.num_frames*2+future))
                        image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)

                    else:
                        # Stretch out multiple frames horizontally
                        pred = stretch_image(pred, self.config.data.channels, self.config.data.image_size)
                        nrow = ceil(np.sqrt(n_init_samples))
                        image_grid = make_grid(pred, nrow)

                    save_image(image_grid, os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                    torch.save(pred, os.path.join(self.args.log_sample_path, 'samples_{}.pt'.format(step)))

                    del all_samples

                del test_scorenet

                self.time_elapsed.update(self.convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=(time.time() - self.start_time)))) + self.time_elapsed_prev)

                # Save meters
                if step == 1 or step % self.config.training.val_freq == 0 or step % 1000 == 0 or step % self.config.training.snapshot_freq == 0:
                    self.save_meters()

            if early_end:
                break

            # If FFHQ tfrecord, reset dataloader
            if self.config.data.dataset.upper() == 'FFHQ':
                dataloader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.training.batch_size, self.config.data.image_size)
                test_loader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.training.batch_size, self.config.data.image_size)
                test_iter = iter(test_loader)

        # Save model at the very end
        states = [
            scorenet.state_dict(),
            optimizer.state_dict(),
            epoch,
            step,
        ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        logging.info(f"Saving checkpoints in {self.args.log_path}")
        torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pt'.format(step)))
        torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pt'))

        # Show best results for every metric
        logging.info("Best-MSE - {}".format(", ".join([f"{k}:{self.best_mse[k]}" for k in self.best_mse])))
        logging.info("Best-PSNR - {}".format(", ".join([f"{k}:{self.best_psnr[k]}" for k in self.best_psnr])))
        logging.info("Best-SSIM - {}".format(", ".join([f"{k}:{self.best_ssim[k]}" for k in self.best_ssim])))
        logging.info("Best-LPIPS - {}".format(", ".join([f"{k}:{self.best_lpips[k]}" for k in self.best_lpips])))
        if getattr(self.config.sampling, "fvd", False):
            logging.info("Best-FVD - {}".format(", ".join([f"{k}:{self.best_fvd[k]}" for k in self.best_fvd])))

    def plot_graphs(self):
        # Losses
        plt.plot(self.losses_train.steps, self.losses_train.vals, label='Train')
        plt.plot(self.losses_test.steps, self.losses_test.vals, label='Test')
        plt.xlabel("Steps")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, 'loss.png'))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, 'loss_log.png'))
        plt.clf()
        plt.close()
        # Epochs
        plt.plot(self.losses_train.steps, self.epochs.vals)
        plt.xlabel("Steps")
        plt.ylabel("Epochs")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'epochs.png'))
        plt.clf()
        plt.close()
        # LR
        plt.plot(self.losses_train.steps, self.lr_meter.vals)
        plt.xlabel("Steps")
        plt.ylabel("LR")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'lr.png'))
        plt.clf()
        plt.close()
        # Grad Norm
        plt.plot(self.losses_train.steps, self.grad_norm.vals)
        plt.xlabel("Steps")
        plt.ylabel("Grad Norm")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'grad.png'))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, 'grad_log.png'))
        plt.clf()
        plt.close()
        # Time train
        plt.plot(self.losses_train.steps, self.time_train.vals)
        plt.xlabel("Steps")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'time_train.png'))
        plt.clf()
        plt.close()
        # Time elapsed
        plt.plot(self.losses_train.steps[:len(self.time_elapsed.vals)], self.time_elapsed.vals)
        plt.xlabel("Steps")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'time_elapsed.png'))
        plt.clf()
        plt.close()

    def plot_video_graphs_single(self, name, mses, psnrs, ssims, lpipss, fvds, calc_fvd,
                                 best_mse, best_psnr, best_ssim, best_lpips, best_fvd):
        # MSE
        plt.plot(mses.steps, mses.vals)
        if best_mse['ckpt'] > -1:
            plt.scatter(best_mse['ckpt'], mses.vals[mses.steps.index(best_mse['ckpt'])], color='k')
            plt.text(best_mse['ckpt'], mses.vals[mses.steps.index(best_mse['ckpt'])], f"{mses.vals[mses.steps.index(best_mse['ckpt'])]:.04f}\n{best_mse['ckpt']}", c='r')
        plt.xlabel("Steps")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        # plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, f"mse_{name}.png"))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, f"mse_{name}_log.png"))
        plt.clf()
        plt.close()
        # PSNR
        plt.plot(mses.steps, psnrs.vals)
        if best_psnr['ckpt'] > -1:
            plt.scatter(best_psnr['ckpt'], psnrs.vals[mses.steps.index(best_psnr['ckpt'])], color='k')
            plt.text(best_psnr['ckpt'], psnrs.vals[mses.steps.index(best_psnr['ckpt'])], f"{psnrs.vals[mses.steps.index(best_psnr['ckpt'])]:.04f}\n{best_psnr['ckpt']}", c='r')
        plt.xlabel("Steps")
        plt.ylabel("PSNR")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        # plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, f"psnr_{name}.png"))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, f"psnr_{name}_log.png"))
        plt.clf()
        plt.close()
        # SSIM
        plt.plot(mses.steps, ssims.vals)
        if best_ssim['ckpt'] > -1:
            plt.scatter(best_ssim['ckpt'], ssims.vals[mses.steps.index(best_ssim['ckpt'])], color='k')
            plt.text(best_ssim['ckpt'], ssims.vals[mses.steps.index(best_ssim['ckpt'])], f"{ssims.vals[mses.steps.index(best_ssim['ckpt'])]:.04f}\n{best_ssim['ckpt']}", c='r')
        plt.xlabel("Steps")
        plt.ylabel("SSIM")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        # plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, f"ssim_{name}.png"))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, f"ssim_{name}_log.png"))
        plt.clf()
        plt.close()
        # LPIPS
        plt.plot(mses.steps, lpipss.vals)
        if best_lpips['ckpt'] > -1:
            plt.scatter(best_lpips['ckpt'], lpipss.vals[mses.steps.index(best_lpips['ckpt'])], color='k')
            plt.text(best_lpips['ckpt'], lpipss.vals[mses.steps.index(best_lpips['ckpt'])], f"{lpipss.vals[mses.steps.index(best_lpips['ckpt'])]:.04f}\n{best_lpips['ckpt']}", c='r')
        plt.xlabel("Steps")
        plt.ylabel("LPIPS")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        # plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, f"lpips_{name}.png"))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, f"lpips_{name}_log.png"))
        plt.clf()
        plt.close()
        # FVD
        if calc_fvd:
            plt.plot(mses.steps, fvds.vals)
            if best_fvd['ckpt'] > -1:
                plt.scatter(best_fvd['ckpt'], fvds.vals[mses.steps.index(best_fvd['ckpt'])], color='k')
                plt.text(best_fvd['ckpt'], fvds.vals[mses.steps.index(best_fvd['ckpt'])], f"{fvds.vals[mses.steps.index(best_fvd['ckpt'])]:.04f}\n{best_fvd['ckpt']}", c='r')
            plt.xlabel("Steps")
            plt.ylabel("FVD")
            plt.grid(True)
            plt.grid(visible=True, which='minor', axis='y', linestyle='--')
            # plt.legend(loc='upper right')
            self.savefig(os.path.join(self.args.log_path, f"fvd_{name}.png"))
            plt.yscale("log")
            self.savefig(os.path.join(self.args.log_path, f"fvd_{name}_log.png"))
            plt.clf()
            plt.close()

    def plot_video_graphs(self):
        # Pred
        if self.mode_pred is not None:
            self.plot_video_graphs_single("pred",
                                          self.mses_pred, self.psnrs_pred, self.ssims_pred, self.lpipss_pred, self.fvds_pred, self.calc_fvd_pred,
                                          self.best_mse_pred, self.best_psnr_pred, self.best_ssim_pred, self.best_lpips_pred, self.best_fvd_pred)
        # Interp
        if self.mode_interp is not None:
            self.plot_video_graphs_single("interp",
                                          self.mses_interp, self.psnrs_interp, self.ssims_interp, self.lpipss_interp, self.fvds_interp, self.calc_fvd_interp,
                                          self.best_mse_interp, self.best_psnr_interp, self.best_ssim_interp, self.best_lpips_interp, self.best_fvd_interp)
        # Gen
        if self.mode_gen is not None and self.calc_fvd3:
            plt.plot(self.fvds3.steps, self.fvds3.vals)
            if self.best_fvd3['ckpt'] > -1:
                plt.scatter(self.best_fvd3['ckpt'], self.fvds3.vals[self.fvds3.steps.index(self.best_fvd3['ckpt'])], color='k')
                plt.text(self.best_fvd3['ckpt'], self.fvds3.vals[self.fvds3.steps.index(self.best_fvd3['ckpt'])], f"{self.fvds3.vals[self.fvds3.steps.index(self.best_fvd3['ckpt'])]:.04f}\n{self.best_fvd3['ckpt']}", c='r')
            plt.xlabel("Steps")
            plt.ylabel("FVD")
            plt.grid(True)
            plt.grid(visible=True, which='minor', axis='y', linestyle='--')
            # plt.legend(loc='upper right')
            self.savefig(os.path.join(self.args.log_path, 'fvd_gen.png'))
            plt.yscale("log")
            self.savefig(os.path.join(self.args.log_path, 'fvd_gen_log.png'))
            plt.clf()
            plt.close()

    def savefig(self, path, bbox_inches='tight', pad_inches=0.1):
        try:
            plt.savefig(path, bbox_inches=bbox_inches, pad_inches=pad_inches)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print(sys.exc_info()[0])

    def sample(self):
        if self.config.sampling.ckpt_id is None:
            ckpt = "latest"
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pt'), map_location=self.config.device)
        else:
            ckpt = self.config.sampling.ckpt_id
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pt'),
                                map_location=self.config.device)

        scorenet = get_model(self.config)
        scorenet = torch.nn.DataParallel(scorenet)

        scorenet.load_state_dict(states[0], strict=False)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(scorenet)

        # sigmas = get_sigmas(self.config)
        # sigmas = sigmas_th.cpu().numpy()

        # If FFHQ tfrecord, reset dataloader
        if self.config.data.dataset.upper() == 'FFHQ':
            dataloader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
        else:
            dataset_train, dataset_test = get_dataset(self.args.data_path, self.config, video_frames_pred=self.config.data.num_frames)
            dataset = dataset_train if getattr(self.config.sampling, "train", False) else dataset_test
            dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True, num_workers=self.config.data.num_workers)

        scorenet.eval()

        net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Conditional
        conditional = self.config.data.num_frames_cond > 0
        cond = None

        # Future
        future = getattr(self.config.data, "num_frames_future", 0)

        if not self.config.sampling.fid:
            if self.config.sampling.inpainting:
                data_iter = iter(dataloader)
                refer_images, _ = next(data_iter)
                refer_images = refer_images.to(self.config.device)
                refer_images = data_transform(self.config, refer_images)
                refer_images, cond, _ = conditioning_fn(self.config, refer_images, num_frames_pred=self.config.data.num_frames, conditional=conditional)
                width = ceil(np.sqrt(self.config.sampling.batch_size))

                # init_samples
                init_samples_shape = (width, width, self.config.data.channels*self.config.data.num_frames,
                                      self.config.data.image_size, self.config.data.image_size)
                if self.version == "SMLD":
                    init_samples = torch.rand(init_samples_shape, device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)
                elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                    if getattr(self.config.model, 'gamma', False):
                        used_k, used_theta = net.k_cum[0], net.theta_t[0]
                        z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                        init_samples = z - used_k*used_theta # we don't scale here
                    else:
                        init_samples = torch.randn(init_samples_shape, device=self.config.device)

                all_samples = anneal_Langevin_dynamics_inpainting(init_samples, refer_images[:width, ...], scorenet,
                                                                  self.config.data.image_size,
                                                                  self.config.sampling.n_steps_each,
                                                                  self.config.sampling.step_lr,
                                                                  cond=cond)


                torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pt'))

                # Stretch out multiple frames horizontally
                refer_images = stretch_image(refer_images, self.config.data.channels, self.config.data.image_size)


                refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                                                                                                     *refer_images.shape[
                                                                                                      1:])
                save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)

                if not self.config.sampling.final_only:
                    for i, sample in enumerate(tqdm(all_samples)):
                        sample = sample.reshape(self.config.sampling.batch_size, self.config.data.channels*self.config.data.num_frames,
                                                self.config.data.image_size, self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        # Stretch out multiple frames horizontally
                        sample = stretch_image(sample, self.config.data.channels, self.config.data.image_size)

                        image_grid = make_grid(sample, ceil(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'completion_{}.pt'.format(i)))
                else:
                    sample = all_samples[-1].reshape(self.config.sampling.batch_size, self.config.data.channels*self.config.data.num_frames,
                                                     self.config.data.image_size, self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    # Stretch out multiple frames horizontally
                    sample = stretch_image(sample, self.config.data.channels, self.config.data.image_size)

                    image_grid = make_grid(sample, ceil(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(ckpt)))
                    torch.save(sample, os.path.join(self.args.image_folder, 'completion_{}.pt'.format(ckpt)))

            elif self.config.sampling.interpolation:

                if self.config.sampling.data_init or conditional:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    samples, cond, _ = conditioning_fn(self.config, samples, num_frames_pred=self.config.data.num_frames, conditional=conditional)

                # z
                init_samples_shape = (self.config.sampling.batch_size, self.config.data.channels*self.config.data.num_frames,
                                      self.config.data.image_size, self.config.data.image_size)
                if self.version == "SMLD":
                    z = torch.rand(init_samples_shape, device=self.config.device)
                    z = data_transform(self.config, z)
                elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                    if getattr(self.config.model, 'gamma', False):
                        used_k, used_theta = net.k_cum[0], net.theta_t[0]
                        z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                        z = z - used_k*used_theta
                    else:
                        z = torch.randn(init_samples_shape, device=self.config.device)
                        # z = data_transform(self.config, z)

                # init_samples
                if self.config.sampling.data_init:
                    if self.version == "SMLD":
                        z = torch.randn_like(real)
                        init_samples = samples + float(self.config.model.sigma_begin) * z
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        alpha = net.alphas[0]
                        z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                        init_samples = alpha.sqrt() * samples + (1 - alpha).sqrt() * z
                else:
                    init_samples = z


                all_samples = anneal_Langevin_dynamics_interpolation(init_samples, scorenet,
                                                                     self.config.sampling.n_interpolations,
                                                                     self.config.sampling.n_steps_each,
                                                                     self.config.sampling.step_lr, verbose=True,
                                                                     final_only=self.config.sampling.final_only,
                                                                     cond=cond)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.reshape(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        # Stretch out multiple frames horizontally
                        sample = stretch_image(sample, self.config.data.channels, self.config.data.image_size)  

                        image_grid = make_grid(sample, nrow=self.config.sampling.n_interpolations)
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pt'.format(i)))
                else:
                    sample = all_samples[-1].reshape(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    # Stretch out multiple frames horizontally
                    sample = stretch_image(sample, self.config.data.channels, self.config.data.image_size)

                    image_grid = make_grid(sample, self.config.sampling.n_interpolations)
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(ckpt)))
                    torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pt'.format(ckpt)))

            else:

                if self.config.sampling.data_init or conditional:
                    data_iter = iter(dataloader)
                    real, _ = next(data_iter)
                    real = real.to(self.config.device)
                    real = data_transform(self.config, real)
                    real, cond, cond_mask = conditioning_fn(self.config, real, num_frames_pred=self.config.data.num_frames, conditional=conditional)

                # z
                init_samples_shape = (self.config.sampling.batch_size, self.config.data.channels*self.config.data.num_frames,
                                      self.config.data.image_size, self.config.data.image_size)
                if self.version == "SMLD":
                    z = torch.rand(init_samples_shape, device=self.config.device)
                    z = data_transform(self.config, z)
                elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                    if getattr(self.config.model, 'gamma', False):
                        used_k, used_theta = net.k_cum[0], net.theta_t[0]
                        z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                        z = z - used_k*used_theta
                    else:
                        z = torch.randn(init_samples_shape, device=self.config.device)
                        # z = data_transform(self.config, z)

                # init_samples
                if self.config.sampling.data_init:
                    if self.version == "SMLD":
                        z = torch.randn_like(real)
                        init_samples = real + float(self.config.model.sigma_begin) * z
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        alpha = net.alphas[0]
                        z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                        init_samples = alpha.sqrt() * real + (1 - alpha).sqrt() * z
                else:
                    init_samples = z

                # Sampler
                sampler = self.get_sampler()

                all_samples = sampler(init_samples, scorenet, cond=cond, cond_mask=cond_mask,
                                      n_steps_each=self.config.sampling.n_steps_each,
                                      step_lr=self.config.sampling.step_lr, verbose=True,
                                      final_only=self.config.sampling.final_only,
                                      denoise=self.config.sampling.denoise,
                                      subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                      clip_before=getattr(self.config.sampling, 'clip_before', True),
                                      log=True, gamma=getattr(self.config.model, 'gamma', False)).to('cpu')

                if not self.config.sampling.final_only:
                    for i, sample in tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.reshape(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        # Stretch out multiple frames horizontally
                        sample = stretch_image(sample, self.config.data.channels, self.config.data.image_size)

                        nrow = ceil(np.sqrt(self.config.data.num_frames*self.config.sampling.batch_size)/self.config.data.num_frames)
                        image_grid = make_grid(sample, nrow, pad_value=0.5)
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{:04d}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{:04d}.pt'.format(i)))

                else:
                    sample = all_samples[-1].reshape(all_samples[-1].shape[0], self.config.data.channels*self.config.data.num_frames,
                                                  self.config.data.image_size, self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    # Stretch out multiple frames horizontally
                    sample = stretch_image(sample, self.config.data.channels, self.config.data.image_size)

                    nrow = ceil(np.sqrt(self.config.data.num_frames*self.config.sampling.batch_size)/self.config.data.num_frames)
                    image_grid = make_grid(sample, nrow, pad_value=0.5)
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(ckpt)))
                    torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pt'.format(ckpt)))

                if conditional:
                    real, cond = real.to('cpu'), cond.to('cpu')
                    real = stretch_image(inverse_data_transform(self.config, real), self.config.data.channels, self.config.data.image_size)
                    if future > 0:
                        cond, futr = torch.tensor_split(cond, (self.config.data.num_frames_cond*self.config.data.channels,), dim=1)
                        futr = stretch_image(inverse_data_transform(self.config, futr), self.config.data.channels, self.config.data.image_size)
                    cond = stretch_image(inverse_data_transform(self.config, cond), self.config.data.channels, self.config.data.image_size)
                    padding = 0.5*torch.ones(len(real), self.config.data.channels, self.config.data.image_size, 2)
                    nrow = ceil(np.sqrt((self.config.data.num_frames_cond+self.config.data.num_frames*2+future)*self.config.sampling.batch_size)/(self.config.data.num_frames_cond+self.config.data.num_frames*2+future))
                    image_grid = make_grid(torch.cat(
                        [cond, padding, real, padding, sample] if future == 0 else [cond, padding, real, padding, sample, futr],
                            dim=-1), nrow=nrow, padding=6, pad_value=0.5)
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_full_grid_{}.png'.format(ckpt)))
                    torch.save(sample, os.path.join(self.args.image_folder, 'samples_full_{}.pt'.format(ckpt)))

        else:
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // self.config.sampling.batch_size
            if self.config.sampling.data_init:

                # If FFHQ tfrecord, reset dataloader
                if self.config.data.dataset.upper() == 'FFHQ':
                    dataloader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                else:
                    dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                            num_workers=self.config.data.num_workers)
                data_iter = iter(dataloader)

            conditional = self.config.data.num_frames_cond > 0
            cond = None
            if conditional:
                if self.config.data.dataset.upper() == 'FFHQ':
                    dataloader_cond = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                else:
                    dataloader_cond = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                            num_workers=self.config.data.num_workers)
                data_iter_cond = iter(dataloader_cond)

            # Sampler
            sampler = self.get_sampler()
            fids = {}
            for i in tqdm(range(n_rounds), desc='Generating samples for FID'):

                init_samples_shape = (self.config.sampling.batch_size, self.config.data.channels*self.config.data.num_frames,
                                      self.config.data.image_size, self.config.data.image_size)
                # z
                if self.version == "SMLD":
                    z = torch.rand(init_samples_shape, device=self.config.device)
                    z = data_transform(self.config, z)
                elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                    if getattr(self.config.model, 'gamma', False):
                        used_k, used_theta = net.k_cum[0], net.theta_t[0]
                        z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                        z = z - used_k*used_theta
                    else:
                        z = torch.randn(init_samples_shape, device=self.config.device)
                        # z = data_transform(self.config, z)

                # init_samples
                if self.config.sampling.data_init:
                    try:
                        real, _ = next(data_iter)
                    except StopIteration:
                        if self.config.data.dataset.upper() == 'FFHQ':
                            dataloader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                        data_iter = iter(dataloader)
                        real, _ = next(data_iter)
                    real = real.to(self.config.device)
                    real = data_transform(self.config, real)
                    real, cond, _ = conditioning_fn(self.config, real, num_frames_pred=self.config.data.num_frames, conditional=conditional)
                    if self.version == "SMLD":
                        z = torch.randn_like(real)
                        init_samples = real + float(self.config.model.sigma_begin) * z
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        alpha = net.alphas[0]
                        z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                        init_samples = alpha.sqrt() * real + (1 - alpha).sqrt() * z
                else:
                    init_samples = z

                if conditional and not self.config.sampling.data_init:
                    try:
                        real, _ = next(data_iter_cond)
                    except StopIteration:
                        if self.config.data.dataset.upper() == 'FFHQ':
                            dataloader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                        data_iter_cond = iter(dataloader)
                        real, _ = next(data_iter_cond)
                    real = real.to(self.config.device)
                    real = data_transform(self.config, real)
                    _, cond, _ = conditioning_fn(self.config, real, num_frames_pred=self.config.data.num_frames, conditional=conditional)

                all_samples = sampler(init_samples, scorenet, cond=cond,
                                      n_steps_each=self.config.sampling.n_steps_each,
                                      step_lr=self.config.sampling.step_lr, verbose=False,
                                      denoise=self.config.sampling.denoise,
                                      subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                      clip_before=getattr(self.config.sampling, 'clip_before', True),
                                      log=True, gamma=getattr(self.config.model, 'gamma', False)).to('cpu')

                final_samples = all_samples[-1].reshape(all_samples[-1].shape[0], self.config.data.channels*self.config.data.num_frames,
                                                        self.config.data.image_size, self.config.data.image_size)
                final_samples = inverse_data_transform(self.config, final_samples)
                gen_samples = final_samples if i == 0 else torch.cat([gen_samples, final_samples], dim=0)

            # FID
            feats_path = get_feats_path(getattr(self.config.fast_fid, 'dataset', self.config.data.dataset).upper(),
                                        self.args.feats_dir)
            k = self.config.fast_fid.pr_nn_k
            fid, precision, recall = get_fid_PR(feats_path, gen_samples, self.config.device, k=k)
            fids[ckpt], precisions[ckpt], recalls[ckpt] = fid, precision, recall
            print("ckpt: {}, fid: {}, precision: {}, recall: {}".format(ckpt, fid, precision, recall))

            self.write_to_pickle(os.path.join(self.args.image_folder, 'fids.pickle'), fids)
            self.write_to_yaml(os.path.join(self.args.image_folder, 'fids.yml'), fids)
            self.write_to_pickle(os.path.join(self.args.image_folder, f'precisions_k{k}.pickle'), precisions)
            self.write_to_yaml(os.path.join(self.args.image_folder, f'precisions_k{k}.yml'), precisions)
            self.write_to_pickle(os.path.join(self.args.image_folder, f'recalls_k{k}.pickle'), recalls)
            self.write_to_yaml(os.path.join(self.args.image_folder, f'recalls_k{k}.yml'), recalls)

            # Save samples
            # Stretch out multiple frames horizontally
            gen_samples_to_save = stretch_image(gen_samples[:self.config.sampling.batch_size], self.config.data.channels, self.config.data.image_size)
            nrow = ceil(np.sqrt(self.config.data.num_frames*self.config.sampling.batch_size)/self.config.data.num_frames)
            image_grid = make_grid(gen_samples_to_save, nrow, pad_value=0.5)
            save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(ckpt)))
            torch.save(gen_samples, os.path.join(self.args.image_folder, 'samples_{}.pt'.format(ckpt)))

    @torch.no_grad()
    def video_gen(self, scorenet=None, ckpt=None, train=False):
        # Sample n predictions per test data, choose the best among them for each metric

        calc_ssim = getattr(self.config.sampling, "ssim", False)
        calc_fvd = getattr(self.config.sampling, "fvd", False)

        # FVD
        if calc_fvd:

            if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
                calc_fvd1 = self.condf + self.config.sampling.num_frames_pred >= 10
                calc_fvd2 = calc_fvd3 = False
            elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation
                calc_fvd1 = self.condf + self.config.data.num_frames + self.futrf >= 10
                calc_fvd2 = calc_fvd3 = False
            elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred
                calc_fvd1 = self.condf + self.config.data.num_frames + self.futrf >= 10
                calc_fvd2 = self.condf + self.config.sampling.num_frames_pred >= 10
                calc_fvd3 = False
            elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen
                calc_fvd1 = calc_fvd3 = self.condf + self.config.sampling.num_frames_pred >= 10
                calc_fvd2 = False
            elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:      # (1) Interp + (2) Pred + (3) Gen
                calc_fvd1 = self.condf + self.config.data.num_frames + self.futrf >= 10
                calc_fvd2 = calc_fvd3 = self.condf + self.config.sampling.num_frames_pred >= 10
            elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:          # (1) Interp + (3) Gen
                calc_fvd1 = self.condf + self.config.data.num_frames + self.futrf >= 10
                calc_fvd2 = False
                calc_fvd3 = self.condf + self.config.sampling.num_frames_pred >= 10

            if calc_fvd1 or calc_fvd2 or calc_fvd3:
                i3d = load_i3d_pretrained(self.config.device)

            self.calc_fvd1, self.calc_fvd2, self.calc_fvd3 = calc_fvd1, calc_fvd2, calc_fvd3

        else:
            self.calc_fvd1, self.calc_fvd2, self.calc_fvd3 = calc_fvd1, calc_fvd2, calc_fvd3 = False, False, False

            if calc_ssim is False:
                return {}

        if train:
            assert(scorenet is not None and ckpt is not None)
            max_data_iter = 1   # self.config.sampling.max_data_iter
            preds_per_test = 1  # self.config.sampling.preds_per_test
        else:
            self.start_time = time.time()
            max_data_iter = self.config.sampling.max_data_iter
            preds_per_test = getattr(self.config.sampling, 'preds_per_test', 1)

        # Conditional
        conditional = self.config.data.num_frames_cond > 0
        assert conditional, f"Video generating model has to be conditional! num_frames_cond has to be > 0! Given {self.config.data.num_frames_cond}"
        cond = None
        prob_mask_cond = getattr(self.config.data, 'prob_mask_cond', 0.0)

        # Future
        future = getattr(self.config.data, "num_frames_future", 0)
        prob_mask_future = getattr(self.config.data, 'prob_mask_future', 0.0)

        if scorenet is None:

            if self.config.sampling.ckpt_id is None:
                ckpt = "latest"
                logging.info(f"Loading ckpt {os.path.join(self.args.log_path, 'checkpoint.pt')}")
                states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pt'), map_location=self.config.device)
            else:
                ckpt = self.config.sampling.ckpt_id
                ckpt_file = os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pt')
                logging.info(f"Loading ckpt {ckpt_file}")
                states = torch.load(ckpt_file, map_location=self.config.device)

            scorenet = get_model(self.config)
            scorenet = torch.nn.DataParallel(scorenet)

            scorenet.load_state_dict(states[0], strict=False)
            scorenet.eval()

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(scorenet)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(scorenet)


        net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Collate fn for n repeats
        def my_collate(batch):
            data, _ = zip(*batch)
            data = torch.stack(data).repeat_interleave(preds_per_test, dim=0)
            return data, torch.zeros(len(data))

        # Data
        if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
            num_frames_pred = self.config.sampling.num_frames_pred
        elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation
            num_frames_pred = self.config.data.num_frames
        elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred
            num_frames_pred = max(self.config.data.num_frames, self.config.sampling.num_frames_pred)
        elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen
            num_frames_pred = self.config.sampling.num_frames_pred
        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred + (3) Gen
            num_frames_pred = self.config.sampling.num_frames_pred
        elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:     # (1) Interp + (3) Gen
            num_frames_pred = max(self.config.data.num_frames, self.config.sampling.num_frames_pred)

        dataset_train, dataset_test = get_dataset(self.args.data_path, self.config, video_frames_pred=num_frames_pred, start_at=self.args.start_at)
        dataset = dataset_train if getattr(self.config.sampling, "train", False) else dataset_test
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size//preds_per_test, shuffle=True,
                                num_workers=self.config.data.num_workers, drop_last=False, collate_fn=my_collate)
        data_iter = iter(dataloader)

        if self.config.sampling.data_init:
            dataloader2 = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, drop_last=False)
            data_iter2 = iter(dataloader2)

        vid_mse, vid_ssim, vid_lpips = [], [], []
        vid_mse2, vid_ssim2, vid_lpips2 = [], [], []
        real_embeddings, real_embeddings2, real_embeddings_uncond = [], [], []
        fake_embeddings, fake_embeddings2, fake_embeddings_uncond = [], [], []

        T2 = Transforms.Compose([Transforms.Resize((128, 128)),
                     Transforms.ToTensor(),
                     Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                         std=(0.5, 0.5, 0.5))])
        model_lpips = eval_models.PerceptualLoss(model='net-lin',net='alex', device=self.config.device) # already in test mode and dataparallel
        #model_lpips = torch.nn.DataParallel(model_lpips)
        #model_lpips.eval()
        # Sampler
        sampler = self.get_sampler()

        for i, (real_, _) in tqdm(enumerate(dataloader), total=min(max_data_iter, len(dataloader)), desc="\nvideo_gen dataloader"):

            if i >= max_data_iter: # stop early
                break

            real_ = data_transform(self.config, real_)

            # (1) Conditional Video Predition/Interpolation : Calc MSE,etc. and FVD on fully cond model i.e. prob_mask_cond=0.0
            # This is prediction if future = 0, else this is interpolation

            logging.info(f"(1) Video {'Pred' if future == 0 else 'Interp'}")

            # Video Prediction
            if future == 0:
                num_frames_pred = self.config.sampling.num_frames_pred
                logging.info(f"PREDICTING {num_frames_pred} frames, using a {self.config.data.num_frames} frame model conditioned on {self.config.data.num_frames_cond} frames, subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")
            # Video Interpolation
            else:
                num_frames_pred = self.config.data.num_frames
                logging.info(f"INTERPOLATING {num_frames_pred} frames, using a {self.config.data.num_frames} frame model conditioned on {self.config.data.num_frames_cond} cond + {future} future frames, subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")

            real, cond, cond_mask = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred,
                                                    prob_mask_cond=0.0, prob_mask_future=0.0, conditional=conditional)
            real = inverse_data_transform(self.config, real)
            cond_original = inverse_data_transform(self.config, cond.clone())
            cond = cond.to(self.config.device)

            # z
            init_samples_shape = (real.shape[0], self.config.data.channels*self.config.data.num_frames,
                                  self.config.data.image_size, self.config.data.image_size)
            if self.version == "SMLD":
                z = torch.rand(init_samples_shape, device=self.config.device)
                z = data_transform(self.config, z)
            elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                if getattr(self.config.model, 'gamma', False):
                    used_k, used_theta = net.k_cum[0], net.theta_t[0]
                    z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                    z = z - used_k*used_theta
                else:
                    z = torch.randn(init_samples_shape, device=self.config.device)

            # init_samples
            if self.config.sampling.data_init:
                try:
                    real_init, _ = next(data_iter2)
                except StopIteration:
                    if self.config.data.dataset.upper() == 'FFHQ':
                        dataloader2 = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                    data_iter2 = iter(dataloader2)
                    real_init, _ = next(data_iter2)
                real_init = data_transform(self.config, real_init)
                real_init, _, _ = conditioning_fn(self.config, real_init, conditional=conditional)
                real_init = real_init.to(self.config.device)
                real_init1 = real_init[:, :self.config.data.channels*self.config.data.num_frames]
                if self.version == "SMLD":
                    z = torch.randn_like(real_init1)
                    init_samples = real_init1 + float(self.config.model.sigma_begin) * z
                elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                    alpha = net.alphas[0]
                    z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                    init_samples = alpha.sqrt() * real_init1 + (1 - alpha).sqrt() * z
            else:
                init_samples = z

            if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                n_iter_frames = num_frames_pred
            else:
                n_iter_frames = ceil(num_frames_pred / self.config.data.num_frames)

            pred_samples = []

            for i_frame in tqdm(range(n_iter_frames), desc="Generating video frames"):

                mynet = scorenet

                # Generate samples
                gen_samples = sampler(init_samples if i_frame==0 or getattr(self.config.sampling, 'init_prev_t', -1) <= 0 else gen_samples,
                                      mynet, cond=cond, cond_mask=cond_mask,
                                      n_steps_each=self.config.sampling.n_steps_each, step_lr=self.config.sampling.step_lr,
                                      verbose=True if not train else False, final_only=True, denoise=self.config.sampling.denoise,
                                      subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                      clip_before=getattr(self.config.sampling, 'clip_before', True),
                                      t_min=getattr(self.config.sampling, 'init_prev_t', -1), log=True if not train else False,
                                      gamma=getattr(self.config.model, 'gamma', False))
                gen_samples = gen_samples[-1].reshape(gen_samples[-1].shape[0], self.config.data.channels*self.config.data.num_frames,
                                                      self.config.data.image_size, self.config.data.image_size)
                pred_samples.append(gen_samples.to('cpu'))

                if i_frame == n_iter_frames - 1:
                    continue

                # Continues only if prediction, and future = 0

                # Autoregressively setup conditioning
                # cond -> [cond[n:], pred[:n]]
                if cond is None: # first frames are the cond
                    cond = gen_samples
                elif getattr(self.config.sampling, 'one_frame_at_a_time', False):
                    cond = torch.cat([cond[:, self.config.data.channels:], gen_samples[:, :self.config.data.channels]], dim=1)
                else:
                    cond = torch.cat([cond[:, self.config.data.channels*self.config.data.num_frames:],
                                      gen_samples[:, self.config.data.channels*max(0, self.config.data.num_frames - self.config.data.num_frames_cond):]
                                     ], dim=1)

                # resample new random init
                if self.version == "SMLD":
                    z = torch.rand(init_samples_shape, device=self.config.device)
                    z = data_transform(self.config, z)
                elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                    if getattr(self.config.model, 'gamma', False):
                        used_k, used_theta = net.k_cum[0], net.theta_t[0]
                        z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                        z = z - used_k*used_theta
                    else:
                        z = torch.randn(init_samples_shape, device=self.config.device)

                # init_samples
                if self.config.sampling.data_init:
                    if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                        real_init1 = real_init[self.config.data.channels*(i_frame+1):self.config.data.channels*(i_frame+1+self.config.data.num_frames)]
                    else:
                        real_init1 = real_init[(i_frame+1)*self.config.data.channels*self.config.data.num_frames:(i_frame+2)*self.config.data.channels*self.config.data.num_frames]
                    if self.version == "SMLD":
                        z = torch.randn_like(real_init1)
                        init_samples = real_init1 + float(self.config.model.sigma_begin) * z
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        alpha = net.alphas[0]
                        z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                        init_samples = alpha.sqrt() * real_init1 + (1 - alpha).sqrt() * z
                else:
                    init_samples = z

            pred = torch.cat(pred_samples, dim=1)[:, :self.config.data.channels*num_frames_pred]
            pred = inverse_data_transform(self.config, pred)
            # pred has length of multiple of n (because we repeat data sample n times)
            
            if real.shape[1] < pred.shape[1]: # We cannot calculate MSE, PSNR, SSIM
                print("-------- Warning: Cannot calculate metrics because predicting beyond the training data range --------")
                for ii in range(len(pred)):
                    vid_mse.append(0)
                    vid_ssim.append(0)
                    vid_lpips.append(0)
            else:
                # Calculate MSE, PSNR, SSIM
                for ii in range(len(pred)):
                    mse, avg_ssim, avg_distance = 0, 0, 0
                    for jj in range(num_frames_pred):

                        # MSE (and PSNR)
                        pred_ij = pred[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                        real_ij = real[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                        mse += F.mse_loss(real_ij, pred_ij)

                        pred_ij_pil = Transforms.ToPILImage()(pred_ij).convert("RGB")
                        real_ij_pil = Transforms.ToPILImage()(real_ij).convert("RGB")

                        # SSIM
                        pred_ij_np_grey = np.asarray(pred_ij_pil.convert('L'))
                        real_ij_np_grey = np.asarray(real_ij_pil.convert('L'))
                        if self.config.data.dataset.upper() == "STOCHASTICMOVINGMNIST" or self.config.data.dataset.upper() == "MOVINGMNIST":
                            # ssim is the only metric extremely sensitive to gray being compared to b/w 
                            pred_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(pred_ij)).convert("RGB").convert('L'))
                            real_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(real_ij)).convert("RGB").convert('L'))
                        avg_ssim += ssim(pred_ij_np_grey, real_ij_np_grey, data_range=255, gaussian_weights=True, use_sample_covariance=False)

                        # Calculate LPIPS
                        pred_ij_LPIPS = T2(pred_ij_pil).unsqueeze(0).to(self.config.device)
                        real_ij_LPIPS = T2(real_ij_pil).unsqueeze(0).to(self.config.device)
                        avg_distance += model_lpips.forward(real_ij_LPIPS, pred_ij_LPIPS)

                    vid_mse.append(mse / num_frames_pred)
                    vid_ssim.append(avg_ssim / num_frames_pred)
                    vid_lpips.append(avg_distance.data.item() / num_frames_pred)


            # (2) Conditional Video Predition, if (1) was Interpolation : Calc MSE,etc. and FVD on fully cond model i.e. prob_mask_cond=0.0
            # unless prob_mask_sync is True, in which case perform (3) uncond gen

            second_calc = False
            if future > 0 and prob_mask_future > 0.0 and not self.prob_mask_sync:

                second_calc = True
                logging.info(f"(2) Video Pred")

                num_frames_pred = self.config.sampling.num_frames_pred

                logging.info(f"PREDICTING {num_frames_pred} frames, using a {self.config.data.num_frames} frame model conditioned on {self.config.data.num_frames_cond} frames, subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")

                real2, cond, cond_mask = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred,
                                                         prob_mask_cond=0.0, prob_mask_future=1.0, conditional=conditional)
                real2 = inverse_data_transform(self.config, real2)
                cond_original2 = inverse_data_transform(self.config, cond.clone())
                cond = cond.to(self.config.device)

                # z
                init_samples_shape = (real.shape[0], self.config.data.channels*self.config.data.num_frames,
                                      self.config.data.image_size, self.config.data.image_size)
                if self.version == "SMLD":
                    z = torch.rand(init_samples_shape, device=self.config.device)
                    z = data_transform(self.config, z)
                elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                    if getattr(self.config.model, 'gamma', False):
                        used_k, used_theta = net.k_cum[0], net.theta_t[0]
                        z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                        z = z - used_k*used_theta
                    else:
                        z = torch.randn(init_samples_shape, device=self.config.device)

                # init_samples
                if self.config.sampling.data_init:
                    try:
                        real_init, _ = next(data_iter2)
                    except StopIteration:
                        if self.config.data.dataset.upper() == 'FFHQ':
                            dataloader2 = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                        data_iter2 = iter(dataloader2)
                        real_init, _ = next(data_iter2)
                    real_init = data_transform(self.config, real_init)
                    real_init, _, _ = conditioning_fn(self.config, real_init, conditional=conditional)
                    real_init = real_init.to(self.config.device)
                    real_init1 = real_init[:, :self.config.data.channels*self.config.data.num_frames]
                    if self.version == "SMLD":
                        z = torch.randn_like(real_init1)
                        init_samples = real_init1 + float(self.config.model.sigma_begin) * z
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        alpha = net.alphas[0]
                        z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                        init_samples = alpha.sqrt() * real_init1 + (1 - alpha).sqrt() * z
                else:
                    init_samples = z

                if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                    n_iter_frames = num_frames_pred
                else:
                    n_iter_frames = ceil(num_frames_pred / self.config.data.num_frames)

                pred_samples = []

                for i_frame in tqdm(range(n_iter_frames), desc="Generating video frames"):

                    mynet = scorenet

                    # Generate samples
                    gen_samples = sampler(init_samples if i_frame==0 or getattr(self.config.sampling, 'init_prev_t', -1) <= 0 else gen_samples,
                                          mynet, cond=cond, cond_mask=cond_mask,
                                          n_steps_each=self.config.sampling.n_steps_each, step_lr=self.config.sampling.step_lr,
                                          verbose=True if not train else False, final_only=True, denoise=self.config.sampling.denoise,
                                          subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                          clip_before=getattr(self.config.sampling, 'clip_before', True),
                                          t_min=getattr(self.config.sampling, 'init_prev_t', -1), log=True if not train else False,
                                          gamma=getattr(self.config.model, 'gamma', False))
                    gen_samples = gen_samples[-1].reshape(gen_samples[-1].shape[0], self.config.data.channels*self.config.data.num_frames,
                                                          self.config.data.image_size, self.config.data.image_size)
                    pred_samples.append(gen_samples.to('cpu'))

                    if i_frame == n_iter_frames - 1:
                        continue

                    # Autoregressively setup conditioning
                    # cond -> [cond[n:], pred[:n]]
                    if cond is None: # first frames are the cond
                        cond = gen_samples
                    elif getattr(self.config.sampling, 'one_frame_at_a_time', False):
                        cond = torch.cat([cond[:, self.config.data.channels:],
                                          gen_samples[:, :self.config.data.channels],
                                          cond[:, -self.config.data.channels*future:]   # future frames are always there, but always 0
                                         ], dim=1)
                    else:
                        cond = torch.cat([cond[:, self.config.data.channels*self.config.data.num_frames:-self.config.data.channels*future],
                                          gen_samples[:, self.config.data.channels*max(0, self.config.data.num_frames - self.config.data.num_frames_cond):],
                                          cond[:, -self.config.data.channels*future:]   # future frames are always there, but always 0
                                         ], dim=1)

                    # resample new random init
                    if self.version == "SMLD":
                        z = torch.rand(init_samples_shape, device=self.config.device)
                        z = data_transform(self.config, z)
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        if getattr(self.config.model, 'gamma', False):
                            used_k, used_theta = net.k_cum[0], net.theta_t[0]
                            z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                            z = z - used_k*used_theta
                        else:
                            z = torch.randn(init_samples_shape, device=self.config.device)

                    # init_samples
                    if self.config.sampling.data_init:
                        if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                            real_init1 = real_init[self.config.data.channels*(i_frame+1):self.config.data.channels*(i_frame+1+self.config.data.num_frames)]
                        else:
                            real_init1 = real_init[(i_frame+1)*self.config.data.channels*self.config.data.num_frames:(i_frame+2)*self.config.data.channels*self.config.data.num_frames]
                        if self.version == "SMLD":
                            z = torch.randn_like(real_init1)
                            init_samples = real_init1 + float(self.config.model.sigma_begin) * z
                        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                            alpha = net.alphas[0]
                            z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                            init_samples = alpha.sqrt() * real_init1 + (1 - alpha).sqrt() * z
                    else:
                        init_samples = z

                pred2 = torch.cat(pred_samples, dim=1)[:, :self.config.data.channels*num_frames_pred]
                pred2 = inverse_data_transform(self.config, pred2)
                # pred has length of multiple of n (because we repeat data sample n times)

                if real.shape[1] < pred.shape[1]: # We cannot calculate MSE, PSNR, SSIM
                    print("-------- Warning: Cannot calculate metrics because predicting beyond the training data range --------")
                    for ii in range(len(pred)):
                        vid_mse.append(0)
                        vid_ssim.append(0)
                        vid_lpips.append(0)
                else:
                    # Calculate MSE, PSNR, SSIM
                    for ii in range(len(pred2)):
                        mse, avg_ssim, avg_distance = 0, 0, 0
                        for jj in range(num_frames_pred):

                            # MSE (and PSNR)
                            pred_ij = pred2[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                            real_ij = real2[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                            mse += F.mse_loss(real_ij, pred_ij)

                            pred_ij_pil = Transforms.ToPILImage()(pred_ij).convert("RGB")
                            real_ij_pil = Transforms.ToPILImage()(real_ij).convert("RGB")

                            # SSIM
                            pred_ij_np_grey = np.asarray(pred_ij_pil.convert('L'))
                            real_ij_np_grey = np.asarray(real_ij_pil.convert('L'))
                            if self.config.data.dataset.upper() == "STOCHASTICMOVINGMNIST" or self.config.data.dataset.upper() == "MOVINGMNIST":
                                # ssim is the only metric extremely sensitive to gray being compared to b/w 
                                pred_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(pred_ij)).convert("RGB").convert('L'))
                                real_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(real_ij)).convert("RGB").convert('L'))
                            avg_ssim += ssim(pred_ij_np_grey, real_ij_np_grey, data_range=255, gaussian_weights=True, use_sample_covariance=False)

                            # Calculate LPIPS
                            pred_ij_LPIPS = T2(pred_ij_pil).unsqueeze(0).to(self.config.device)
                            real_ij_LPIPS = T2(real_ij_pil).unsqueeze(0).to(self.config.device)
                            avg_distance += model_lpips.forward(real_ij_LPIPS, pred_ij_LPIPS)

                        vid_mse2.append(mse / num_frames_pred)
                        vid_ssim2.append(avg_ssim / num_frames_pred)
                        vid_lpips2.append(avg_distance.data.item() / num_frames_pred)

            # FVD

            logging.info(f"fvd1 {calc_fvd1}, fvd2 {calc_fvd2}, fvd3 {calc_fvd3}")
            pred_uncond = None
            if calc_fvd1 or calc_fvd2 or calc_fvd3:

                # (3) Unconditional Video Generation: We must redo the predictions with no input conditioning for unconditional FVD
                if calc_fvd3:

                    logging.info(f"(3) Video Gen - Uncond - FVD")

                    # If future = 0, we must make more since first ones are empty frames!
                    # Else, run only 1 iteration, and make only num_frames
                    num_frames_pred = self.config.data.num_frames_cond + self.config.sampling.num_frames_pred

                    logging.info(f"GENERATING (Uncond) {num_frames_pred} frames, using a {self.config.data.num_frames} frame model (conditioned on {self.config.data.num_frames_cond} cond + {self.config.data.num_frames_future} futr frames), subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")

                    # We mask cond
                    _, cond_fvd, cond_mask_fvd = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred,
                                                                 prob_mask_cond=1.0, prob_mask_future=1.0, conditional=conditional)
                    cond_fvd = cond_fvd.to(self.config.device)

                    # z
                    init_samples_shape = (cond_fvd.shape[0], self.config.data.channels*self.config.data.num_frames,
                                          self.config.data.image_size, self.config.data.image_size)
                    if self.version == "SMLD":
                        z = torch.rand(init_samples_shape, device=self.config.device)
                        z = data_transform(self.config, z)
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        if getattr(self.config.model, 'gamma', False):
                            used_k, used_theta = net.k_cum[0], net.theta_t[0]
                            z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                            z = z - used_k*used_theta
                        else:
                            z = torch.randn(init_samples_shape, device=self.config.device)
                            # z = data_transform(self.config, z)

                    # init_samples
                    if self.config.sampling.data_init:
                        try:
                            real_init, _ = next(data_iter2)
                        except StopIteration:
                            if self.config.data.dataset.upper() == 'FFHQ':
                                dataloader2 = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                            data_iter2 = iter(dataloader2)
                            real_init, _ = next(data_iter2)
                        real_init = data_transform(self.config, real_init)
                        real_init, _, _ = conditioning_fn(self.config, real_init, conditional=conditional)
                        real_init = real_init.to(self.config.device)
                        real_init1 = real_init[:, :self.config.data.channels*self.config.data.num_frames]
                        if self.version == "SMLD":
                            z = torch.randn_like(real_init1)
                            init_samples = real_init1 + float(self.config.model.sigma_begin) * z
                        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                            alpha = net.alphas[0]
                            z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                            init_samples = alpha.sqrt() * real_init1 + (1 - alpha).sqrt() * z
                    else:
                        init_samples = z

                    if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                        n_iter_frames = num_frames_pred
                    else:
                        n_iter_frames = ceil(num_frames_pred / self.config.data.num_frames)

                    pred_samples = []
                    for i_frame in tqdm(range(n_iter_frames), desc="Generating video frames"):

                        # Generate samples
                        gen_samples = sampler(init_samples if i_frame==0 or getattr(self.config.sampling, 'init_prev_t', -1) <= 0 else gen_samples,
                                              scorenet, cond=cond_fvd, cond_mask=cond_mask_fvd,
                                              n_steps_each=self.config.sampling.n_steps_each, step_lr=self.config.sampling.step_lr,
                                              verbose=True if not train else False, final_only=True, denoise=self.config.sampling.denoise,
                                              subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                              clip_before=getattr(self.config.sampling, 'clip_before', True),
                                              t_min=getattr(self.config.sampling, 'init_prev_t', -1), log=True if not train else False,
                                              gamma=getattr(self.config.model, 'gamma', False))
                        gen_samples = gen_samples[-1].reshape(gen_samples[-1].shape[0], self.config.data.channels*self.config.data.num_frames,
                                                              self.config.data.image_size, self.config.data.image_size)
                        pred_samples.append(gen_samples.to('cpu'))

                        if i_frame == n_iter_frames - 1:
                            continue

                        # cond -> [cond[n:], real[:n]]
                        if future == 0:
                            if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                                cond_fvd = torch.cat([cond_fvd[:, self.config.data.channels:], gen_samples[:, :self.config.data.channels]], dim=1)
                            else:
                                cond_fvd = torch.cat([cond_fvd[:, self.config.data.channels*self.config.data.num_frames:],
                                                      gen_samples[:, self.config.data.channels*max(0, self.config.data.num_frames - self.config.data.num_frames_cond):]
                                                     ], dim=1)
                        else:
                            if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                                cond_fvd = torch.cat([cond_fvd[:, self.config.data.channels:],
                                                      gen_samples[:, :self.config.data.channels],
                                                      cond_fvd[:, -self.config.data.channels*future:]   # future frames are always there, but always 0
                                                     ], dim=1)
                            else:
                                cond_fvd = torch.cat([cond_fvd[:, self.config.data.channels*self.config.data.num_frames:-self.config.data.channels*future],
                                                      gen_samples[:, self.config.data.channels*max(0, self.config.data.num_frames - self.config.data.num_frames_cond):],
                                                      cond_fvd[:, -self.config.data.channels*future:]   # future frames are always there, but always 0
                                                     ], dim=1)

                        # Make cond_mask one
                        if i_frame == 0:
                            cond_mask_fvd = cond_mask_fvd.fill_(1)
                        # resample new random init
                        if self.version == "SMLD":
                            z = torch.rand(init_samples_shape, device=self.config.device)
                            z = data_transform(self.config, z)
                        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                            if getattr(self.config.model, 'gamma', False):
                                used_k, used_theta = net.k_cum[0], net.theta_t[0]
                                z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                                z = z - used_k*used_theta
                            else:
                                z = torch.randn(init_samples_shape, device=self.config.device)

                        # init_samples
                        if self.config.sampling.data_init:
                            if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                                real_init1 = real_init[self.config.data.channels*(i_frame+1):self.config.data.channels*(i_frame+1+self.config.data.num_frames)]
                            else:
                                real_init1 = real_init[(i_frame+1)*self.config.data.channels*self.config.data.num_frames:(i_frame+2)*self.config.data.channels*self.config.data.num_frames]
                            if self.version == "SMLD":
                                z = torch.randn_like(real_init1)
                                init_samples = real_init1 + float(self.config.model.sigma_begin) * z
                            elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                                alpha = net.alphas[0]
                                z = z / (1 - used_alphas).sqrt() if getattr(self.config.model, 'gamma', False) else z
                                init_samples = alpha.sqrt() * real_init1 + (1 - alpha).sqrt() * z
                        else:
                            init_samples = z

                    pred_uncond = torch.cat(pred_samples, dim=1)[:, :self.config.data.channels*num_frames_pred]
                    pred_uncond = inverse_data_transform(self.config, pred_uncond)

                def to_i3d(x):
                    x = x.reshape(x.shape[0], -1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                    if self.config.data.channels == 1:
                        x = x.repeat(1, 1, 3, 1, 1) # hack for greyscale images
                    x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
                    return x

                if (calc_fvd1 or (calc_fvd3 and not second_calc)) and real.shape[1] >= pred.shape[1]:

                    # real
                    if future == 0:
                        real_fvd = torch.cat([
                            cond_original[:, :self.config.data.num_frames_cond*self.config.data.channels],
                            real
                        ], dim=1)[::preds_per_test]    # Ignore the repeated ones
                    else:
                        real_fvd = torch.cat([
                            cond_original[:, :self.config.data.num_frames_cond*self.config.data.channels],
                            real,
                            cond_original[:, -future*self.config.data.channels:]
                        ], dim=1)[::preds_per_test]    # Ignore the repeated ones
                    real_fvd = to_i3d(real_fvd)
                    real_embeddings.append(get_fvd_feats(real_fvd, i3d=i3d, device=self.config.device))

                    # fake
                    if future == 0:
                        fake_fvd = torch.cat([
                            cond_original[:, :self.config.data.num_frames_cond*self.config.data.channels], pred], dim=1)
                    else:
                        fake_fvd = torch.cat([
                            cond_original[:, :self.config.data.num_frames_cond*self.config.data.channels],
                            pred,
                            cond_original[:, -future*self.config.data.channels:]
                        ], dim=1)
                    fake_fvd = to_i3d(fake_fvd)
                    fake_embeddings.append(get_fvd_feats(fake_fvd, i3d=i3d, device=self.config.device))

                # fake2 : fvd_cond if fvd was fvd_interp
                if (second_calc and (calc_fvd2 or calc_fvd3)) and real.shape[1] >= pred.shape[1]: # only cond, but real has all frames req for interp

                    # real2
                    real_fvd2 = torch.cat([
                        cond_original2[:, :self.config.data.num_frames_cond*self.config.data.channels],
                        real2
                    ], dim=1)[::preds_per_test]    # Ignore the repeated ones
                    real_fvd2 = to_i3d(real_fvd2)
                    real_embeddings2.append(get_fvd_feats(real_fvd2, i3d=i3d, device=self.config.device))

                    # fake2
                    fake_fvd2 = torch.cat([
                        cond_original2[:, :self.config.data.num_frames_cond*self.config.data.channels],
                        pred2
                    ], dim=1)
                    fake_fvd2 = to_i3d(fake_fvd2)
                    fake_embeddings2.append(get_fvd_feats(fake_fvd2, i3d=i3d, device=self.config.device))

                # (3) fake 3: uncond
                if calc_fvd3:
                    # real uncond
                    real_embeddings_uncond.append(real_embeddings2[-1] if second_calc else real_embeddings[-1])

                    # fake uncond
                    fake_fvd_uncond = torch.cat([pred_uncond], dim=1) # We don't want to input the zero-mask
                    fake_fvd_uncond = to_i3d(fake_fvd_uncond)
                    fake_embeddings_uncond.append(get_fvd_feats(fake_fvd_uncond, i3d=i3d, device=self.config.device))

            if i == 0 or preds_per_test == 1: # Save first mini-batch or save them all
                cond = cond_original

                no_metrics = False
                if real.shape[1] < pred.shape[1]: # Pad with zeros to prevent bugs
                    no_metrics = True
                    real = torch.cat([real, torch.zeros(real.shape[0], pred.shape[1]-real.shape[1], real.shape[2], real.shape[3])], dim=1)

                if future > 0:
                    cond, futr = torch.tensor_split(cond, (self.config.data.num_frames_cond*self.config.data.channels,), dim=1)

                # Save gif
                gif_frames_cond = []
                gif_frames_pred, gif_frames_pred2, gif_frames_pred3 = [], [], []
                gif_frames_futr = []

                # cond : # we show conditional frames, and real&pred side-by-side
                for t in range(cond.shape[1]//self.config.data.channels):
                    cond_t = cond[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                    frame = torch.cat([cond_t, 0.5*torch.ones(*cond_t.shape[:-1], 2), cond_t], dim=-1)
                    frame = frame.permute(0, 2, 3, 1).numpy()
                    frame = np.stack([putText(f.copy(), f"{t+1:2d}p", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                    nrow = ceil(np.sqrt(2*cond.shape[0])/2)
                    gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                    gif_frames_cond.append((gif_frame*255).astype('uint8'))
                    if t == 0:
                        gif_frames_cond.append((gif_frame*255).astype('uint8'))
                    del frame, gif_frame

                # pred
                for t in range(pred.shape[1]//self.config.data.channels):
                    real_t = real[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                    pred_t = pred[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                    frame = torch.cat([real_t, 0.5*torch.ones(*pred_t.shape[:-1], 2), pred_t], dim=-1)
                    frame = frame.permute(0, 2, 3, 1).numpy()
                    frame = np.stack([putText(f.copy(), f"{t+1:02d}", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                    nrow = ceil(np.sqrt(2*pred.shape[0])/2)
                    gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                    gif_frames_pred.append((gif_frame*255).astype('uint8'))
                    if t == pred.shape[1]//self.config.data.channels - 1 and future == 0:
                        gif_frames_pred.append((gif_frame*255).astype('uint8'))
                    del frame, gif_frame

                # pred2
                if second_calc:
                    for t in range(pred2.shape[1]//self.config.data.channels):
                        real_t = real2[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                        pred_t = pred2[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                        frame = torch.cat([real_t, 0.5*torch.ones(*pred_t.shape[:-1], 2), pred_t], dim=-1)
                        frame = frame.permute(0, 2, 3, 1).numpy()
                        frame = np.stack([putText(f.copy(), f"{t+1:02d}", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                        nrow = ceil(np.sqrt(2*pred.shape[0])/2)
                        gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                        gif_frames_pred2.append((gif_frame*255).astype('uint8'))
                        if t == pred2.shape[1]//self.config.data.channels - 1:
                            gif_frames_pred2.append((gif_frame*255).astype('uint8'))
                        del frame, gif_frame

                # pred_uncond
                if pred_uncond is not None:
                    for t in range(pred_uncond.shape[1]//self.config.data.channels):
                        frame = pred_uncond[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                        frame = frame.permute(0, 2, 3, 1).numpy()
                        frame = np.stack([putText(f.copy(), f"{t+1:02d}", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                        nrow = ceil(np.sqrt(2*pred_uncond.shape[0])/2)
                        gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                        gif_frames_pred3.append((gif_frame*255).astype('uint8'))
                        if t == pred_uncond.shape[1]//self.config.data.channels - 1:
                            gif_frames_pred3.append((gif_frame*255).astype('uint8'))
                        del frame, gif_frame

                # futr
                if future > 0: # if conditional, we show conditional frames, and real&pred, and future frames side-by-side
                    for t in range(futr.shape[1]//self.config.data.channels):
                        futr_t = futr[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                        frame = torch.cat([futr_t, 0.5*torch.ones(*futr_t.shape[:-1], 2), futr_t], dim=-1)
                        frame = frame.permute(0, 2, 3, 1).numpy()
                        frame = np.stack([putText(f.copy(), f"{t+1:2d}f", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                        nrow = ceil(np.sqrt(2*futr.shape[0])/2)
                        gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                        gif_frames_futr.append((gif_frame*255).astype('uint8'))
                        if t == futr.shape[1]//self.config.data.channels - 1:
                            gif_frames_futr.append((gif_frame*255).astype('uint8'))
                        del frame, gif_frame

                # Save gif
                if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_pred_{ckpt}_{i}.gif"),
                                     [*gif_frames_cond, *gif_frames_pred], fps=4)
                elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_interp_{ckpt}_{i}.gif"),
                                     [*gif_frames_cond, *gif_frames_pred, *gif_frames_futr], fps=4)
                elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_interp_{ckpt}_{i}.gif"),
                                     [*gif_frames_cond, *gif_frames_pred, *gif_frames_futr], fps=4)
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_pred_{ckpt}_{i}.gif"),
                                     [*gif_frames_cond, *gif_frames_pred2], fps=4)
                elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_pred_{ckpt}_{i}.gif"),
                                     [*gif_frames_cond, *gif_frames_pred], fps=4)
                    if len(gif_frames_pred3) > 0:
                        imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_gen_{ckpt}_{i}.gif"),
                                         gif_frames_pred3, fps=4)
                elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:     # (1) Interp + (2) Pred + (3) Gen
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_interp_{ckpt}_{i}.gif"),
                                     [*gif_frames_cond, *gif_frames_pred, *gif_frames_futr], fps=4)
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_pred_{ckpt}_{i}.gif"),
                                     [*gif_frames_cond, *gif_frames_pred2], fps=4)
                    if len(gif_frames_pred3) > 0:
                        imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_gen_{ckpt}_{i}.gif"),
                                         gif_frames_pred3, fps=4)
                elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:     # (1) Interp + (3) Gen
                    imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_interp_{ckpt}_{i}.gif"),
                                     [*gif_frames_cond, *gif_frames_pred, *gif_frames_futr], fps=4)
                    if len(gif_frames_pred3) > 0:
                        imageio.mimwrite(os.path.join(self.args.log_sample_path if train else self.args.video_folder, f"videos_gen_{ckpt}_{i}.gif"),
                                         gif_frames_pred3, fps=4)

                del gif_frames_cond, gif_frames_pred, gif_frames_pred2, gif_frames_pred3, gif_frames_futr

                # Stretch out multiple frames horizontally

                def save_pred(pred, real):
                    if train:
                        torch.save({"cond": cond, "pred": pred, "real": real},
                                   os.path.join(self.args.log_sample_path, f"videos_pred_{ckpt}.pt"))
                    else:
                        torch.save({"cond": cond, "pred": pred, "real": real},
                                   os.path.join(self.args.video_folder, f"videos_pred_{ckpt}.pt"))
                    cond_im = stretch_image(cond, self.config.data.channels, self.config.data.image_size)
                    pred_im = stretch_image(pred, self.config.data.channels, self.config.data.image_size)
                    real_im = stretch_image(real, self.config.data.channels, self.config.data.image_size)
                    padding_hor = 0.5*torch.ones(*real_im.shape[:-1], 2)
                    real_data = torch.cat([cond_im, padding_hor, real_im], dim=-1)
                    pred_data = torch.cat([0.5*torch.ones_like(cond_im), padding_hor, pred_im], dim=-1)
                    padding_ver = 0.5*torch.ones(*real_im.shape[:-2], 2, real_data.shape[-1])
                    data = torch.cat([real_data, padding_ver, pred_data], dim=-2)
                    # Save
                    nrow = ceil(np.sqrt((self.config.data.num_frames_cond+self.config.sampling.num_frames_pred)*pred.shape[0])/(self.config.data.num_frames_cond+self.config.sampling.num_frames_pred))
                    image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
                    if train:
                        save_image(image_grid, os.path.join(self.args.log_sample_path, f"videos_stretch_pred_{ckpt}_{i}.png"))
                    else:
                        save_image(image_grid, os.path.join(self.args.video_folder, f"videos_stretch_pred_{ckpt}_{i}.png"))

                def save_interp(pred, real):
                    if train:
                        torch.save({"cond": cond, "pred": pred, "real": real, "futr": futr},
                                   os.path.join(self.args.log_sample_path, f"videos_interp_{ckpt}.pt"))
                    else:
                        torch.save({"cond": cond, "pred": pred, "real": real, "futr": futr},
                                   os.path.join(self.args.video_folder, f"videos_interp_{ckpt}.pt"))
                    cond_im = stretch_image(cond, self.config.data.channels, self.config.data.image_size)
                    pred_im = stretch_image(pred, self.config.data.channels, self.config.data.image_size)
                    real_im = stretch_image(real, self.config.data.channels, self.config.data.image_size)
                    futr_im = stretch_image(futr, self.config.data.channels, self.config.data.image_size)
                    padding_hor = 0.5*torch.ones(*real_im.shape[:-1], 2)
                    real_data = torch.cat([cond_im, padding_hor, real_im, padding_hor, futr_im], dim=-1)
                    pred_data = torch.cat([0.5*torch.ones_like(cond_im), padding_hor, pred_im, padding_hor, 0.5*torch.ones_like(futr_im)], dim=-1)
                    padding_ver = 0.5*torch.ones(*real_im.shape[:-2], 2, real_data.shape[-1])
                    data = torch.cat([real_data, padding_ver, pred_data], dim=-2)
                    # Save
                    nrow = ceil(np.sqrt((self.config.data.num_frames_cond+self.config.sampling.num_frames_pred+future)*pred.shape[0])/(self.config.data.num_frames_cond+self.config.sampling.num_frames_pred+future))
                    image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
                    if train:
                        save_image(image_grid, os.path.join(self.args.log_sample_path, f"videos_stretch_interp_{ckpt}_{i}.png"))
                    else:
                        save_image(image_grid, os.path.join(self.args.video_folder, f"videos_stretch_interp_{ckpt}_{i}.png"))

                def save_gen(pred):
                    if pred is None:
                        return
                    if train:
                        torch.save({"gen": pred}, os.path.join(self.args.log_sample_path, f"videos_gen_{ckpt}.pt"))
                    else:
                        torch.save({"gen": pred}, os.path.join(self.args.video_folder, f"videos_gen_{ckpt}.pt"))
                    data = stretch_image(pred, self.config.data.channels, self.config.data.image_size)
                    # Save
                    nrow = ceil(np.sqrt((self.config.data.num_frames_cond+self.config.sampling.num_frames_pred)*pred.shape[0])/(self.config.data.num_frames_cond+self.config.sampling.num_frames_pred))
                    image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
                    if train:
                        save_image(image_grid, os.path.join(self.args.log_sample_path, f"videos_stretch_gen_{ckpt}_{i}.png"))
                    else:
                        save_image(image_grid, os.path.join(self.args.video_folder, f"videos_stretch_gen_{ckpt}_{i}.png"))

                if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
                    save_pred(pred, real)

                elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation
                    save_interp(pred, real)

                elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred
                    save_interp(pred, real)
                    save_pred(pred2, real2)

                elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen
                    save_pred(pred, real)
                    save_gen(pred_uncond)

                elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:     # (1) Interp + (2) Pred + (3) Gen
                    save_interp(pred, real)
                    save_pred(pred2, real2)
                    save_gen(pred_uncond)

                elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:     # (1) Interp + (2) Pred + (3) Gen
                    save_interp(pred, real)
                    save_gen(pred_uncond)

        if no_metrics:
            return None

        # Calc MSE, PSNR, SSIM, LPIPS
        mse_list = np.array(vid_mse).reshape(-1, preds_per_test).min(-1)
        psnr_list = (10 * np.log10(1 / np.array(vid_mse))).reshape(-1, preds_per_test).max(-1)
        ssim_list = np.array(vid_ssim).reshape(-1, preds_per_test).max(-1)
        lpips_list = np.array(vid_lpips).reshape(-1, preds_per_test).min(-1)

        def image_metric_stuff(metric):
            avg_metric, std_metric = metric.mean().item(), metric.std().item()
            conf95_metric = avg_metric - float(st.norm.interval(alpha=0.95, loc=avg_metric, scale=st.sem(metric))[0])
            return avg_metric, std_metric, conf95_metric

        avg_mse, std_mse, conf95_mse = image_metric_stuff(mse_list)
        avg_psnr, std_psnr, conf95_psnr = image_metric_stuff(psnr_list)
        avg_ssim, std_ssim, conf95_ssim = image_metric_stuff(ssim_list)
        avg_lpips, std_lpips, conf95_lpips = image_metric_stuff(lpips_list)

        vid_metrics = {'ckpt': ckpt, 'preds_per_test': preds_per_test,
                        'mse': avg_mse, 'mse_std': std_mse, 'mse_conf95': conf95_mse,
                        'psnr': avg_psnr, 'psnr_std': std_psnr, 'psnr_conf95': conf95_psnr,
                        'ssim': avg_ssim, 'ssim_std': std_ssim, 'ssim_conf95': conf95_ssim,
                        'lpips': avg_lpips, 'lpips_std': std_lpips, 'lpips_conf95': conf95_lpips}

        def fvd_stuff(fake_embeddings, real_embeddings):
            avg_fvd = frechet_distance(fake_embeddings, real_embeddings)
            if preds_per_test > 1:
                fvds_list = []
                # Calc FVD for 5 random trajs (each), and average that FVD
                trajs = np.random.choice(np.arange(preds_per_test), (preds_per_test,), replace=False)
                for traj in trajs:
                    fvds_list.append(frechet_distance(fake_embeddings[traj::preds_per_test], real_embeddings))
                fvd_traj_mean, fvd_traj_std  = float(np.mean(fvds_list)), float(np.std(fvds_list))
                fvd_traj_conf95 = fvd_traj_mean - float(st.norm.interval(alpha=0.95, loc=fvd_traj_mean, scale=st.sem(fvds_list))[0])
            else:
                fvd_traj_mean, fvd_traj_std, fvd_traj_conf95 = -1, -1, -1
            return avg_fvd, fvd_traj_mean, fvd_traj_std, fvd_traj_conf95

        # Calc FVD
        if calc_fvd1 or calc_fvd2 or calc_fvd3:

            if calc_fvd1:
                # (1) Video Pred/Interp
                real_embeddings = np.concatenate(real_embeddings)
                fake_embeddings = np.concatenate(fake_embeddings)
                avg_fvd, fvd_traj_mean, fvd_traj_std, fvd_traj_conf95 = fvd_stuff(fake_embeddings, real_embeddings)
                vid_metrics.update({'fvd': avg_fvd, 'fvd_traj_mean': fvd_traj_mean, 'fvd_traj_std': fvd_traj_std, 'fvd_traj_conf95': fvd_traj_conf95})

        if second_calc:
            mse2 = np.array(vid_mse2).reshape(-1, preds_per_test).min(-1)
            psnr2 = (10 * np.log10(1 / np.array(vid_mse2))).reshape(-1, preds_per_test).max(-1)
            ssim2 = np.array(vid_ssim2).reshape(-1, preds_per_test).max(-1)
            lpips2 = np.array(vid_lpips2).reshape(-1, preds_per_test).min(-1)

            avg_mse2, std_mse2, conf95_mse2 = image_metric_stuff(mse2)
            avg_psnr2, std_psnr2, conf95_psnr2 = image_metric_stuff(psnr2)
            avg_ssim2, std_ssim2, conf95_ssim2 = image_metric_stuff(ssim2)
            avg_lpips2, std_lpips2, conf95_lpips2 = image_metric_stuff(lpips2)

            vid_metrics.update({'mse2': avg_mse2, 'mse2_std': std_mse2, 'mse2_conf95': conf95_mse2,
                                'psnr2': avg_psnr2, 'psnr2_std': std_psnr2, 'psnr2_conf95': conf95_psnr2,
                                'ssim2': avg_ssim2, 'ssim2_std': std_ssim2, 'ssim2_conf95': conf95_ssim2,
                                'lpips2': avg_lpips2, 'lpips2_std': std_lpips2, 'lpips2_conf95': conf95_lpips2})

            # (2) Video Pred if 1 was Interp
            if calc_fvd2:
                real_embeddings2 = np.concatenate(real_embeddings2)
                fake_embeddings2 = np.concatenate(fake_embeddings2)
                avg_fvd2, fvd2_traj_mean, fvd2_traj_std, fvd2_traj_conf95 = fvd_stuff(fake_embeddings2, real_embeddings2)
                vid_metrics.update({'fvd2': avg_fvd2, 'fvd2_traj_mean': fvd2_traj_mean, 'fvd2_traj_std': fvd2_traj_std, 'fvd2_traj_conf95': fvd2_traj_conf95})

        # (3) uncond
        if calc_fvd3:
            real_embeddings_uncond = np.concatenate(real_embeddings_uncond)
            fake_embeddings_uncond = np.concatenate(fake_embeddings_uncond)
            avg_fvd3, fvd3_traj_mean, fvd3_traj_std, fvd3_traj_conf95 = fvd_stuff(fake_embeddings_uncond, real_embeddings_uncond)
            vid_metrics.update({'fvd3': avg_fvd3, 'fvd3_traj_mean': fvd3_traj_mean, 'fvd3_traj_std': fvd3_traj_std, 'fvd3_traj_conf95': fvd3_traj_conf95})

        if not train and (calc_fvd1 or calc_fvd2 or calc_fvd3):
            np.savez(os.path.join(self.args.video_folder, f"video_embeddings_{ckpt}.npz"),
                     real_embeddings=real_embeddings,
                     fake_embeddings=fake_embeddings,
                     real_embeddings2=real_embeddings2,
                     fake_embeddings2=fake_embeddings2,
                     real_embeddings3=real_embeddings_uncond,
                     fake_embeddings3=fake_embeddings_uncond)

        if train:
            elapsed = str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3]
        else:
            elapsed = str(datetime.timedelta(seconds=(time.time() - self.start_time)))[:-3]
        format_p = lambda dd : ", ".join([f"{k}:{v:.4f}" if k != 'ckpt' and k != 'preds_per_test' and k != 'time' else f"{k}:{v:7d}" if k == 'ckpt' else f"{k}:{v:3d}" if k == 'preds_per_test' else f"{k}:{v}" for k, v in dd.items()])
        logging.info(f"elapsed: {elapsed}, {format_p(vid_metrics)}")
        logging.info(f"elapsed: {elapsed}, mem:{get_proc_mem():.03f}GB, GPUmem: {get_GPU_mem():.03f}GB")

        if train:
            return vid_metrics

        else:

            logging.info(f"elapsed: {elapsed}, Writing metrics to {os.path.join(self.args.video_folder, 'vid_metrics.yml')}")
            vid_metrics['time'] = elapsed

            if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction

                vid_metrics['pred_mse'], vid_metrics['pred_psnr'], vid_metrics['pred_ssim'], vid_metrics['pred_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
                vid_metrics['pred_mse_std'], vid_metrics['pred_psnr_std'], vid_metrics['pred_ssim_std'], vid_metrics['pred_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
                vid_metrics['pred_mse_conf95'], vid_metrics['pred_psnr_conf95'], vid_metrics['pred_ssim_conf95'], vid_metrics['pred_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
                if calc_fvd1:
                    vid_metrics['pred_fvd'], vid_metrics['pred_fvd_traj_mean'], vid_metrics['pred_fvd_traj_std'], vid_metrics['pred_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

            elif self.condp == 0.0 and self.futrf > 0 and self.futrp == 0.0:    # (1) Interpolation

                vid_metrics['interp_mse'], vid_metrics['interp_psnr'], vid_metrics['interp_ssim'], vid_metrics['interp_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
                vid_metrics['interp_mse_std'], vid_metrics['interp_psnr_std'], vid_metrics['interp_ssim_std'], vid_metrics['interp_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
                vid_metrics['interp_mse_conf95'], vid_metrics['interp_psnr_conf95'], vid_metrics['interp_ssim_conf95'], vid_metrics['interp_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
                if calc_fvd1:
                    vid_metrics['interp_fvd'], vid_metrics['interp_fvd_traj_mean'], vid_metrics['interp_fvd_traj_std'], vid_metrics['interp_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

            elif self.condp == 0.0 and self.futrf > 0 and self.futrp > 0.0:     # (1) Interp + (2) Pred

                vid_metrics['interp_mse'], vid_metrics['interp_psnr'], vid_metrics['interp_ssim'], vid_metrics['interp_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
                vid_metrics['interp_mse_std'], vid_metrics['interp_psnr_std'], vid_metrics['interp_ssim_std'], vid_metrics['interp_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
                vid_metrics['interp_mse_conf95'], vid_metrics['interp_psnr_conf95'], vid_metrics['interp_ssim_conf95'], vid_metrics['interp_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
                if calc_fvd1:
                    vid_metrics['interp_fvd'], vid_metrics['interp_fvd_traj_mean'], vid_metrics['interp_fvd_traj_std'], vid_metrics['interp_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

                if second_calc:
                    vid_metrics['pred_mse'], vid_metrics['pred_psnr'], vid_metrics['pred_ssim'], vid_metrics['pred_lpips'] = vid_metrics['mse2'], vid_metrics['psnr2'], vid_metrics['ssim2'], vid_metrics['lpips2']
                    vid_metrics['pred_mse_std'], vid_metrics['pred_psnr_std'], vid_metrics['pred_ssim_std'], vid_metrics['pred_lpips_std'] = vid_metrics['mse2_std'], vid_metrics['psnr2_std'], vid_metrics['ssim2_std'], vid_metrics['lpips2_std']
                    vid_metrics['pred_mse_conf95'], vid_metrics['pred_psnr_conf95'], vid_metrics['pred_ssim_conf95'], vid_metrics['pred_lpips_conf95'] = vid_metrics['mse2_conf95'], vid_metrics['psnr2_conf95'], vid_metrics['ssim2_conf95'], vid_metrics['lpips2_conf95']
                if calc_fvd2:
                        vid_metrics['pred_fvd'], vid_metrics['pred_fvd_traj_mean'], vid_metrics['pred_fvd_traj_std'], vid_metrics['pred_fvd_traj_conf95'] = vid_metrics['fvd2'], vid_metrics['fvd2_traj_mean'], vid_metrics['fvd2_traj_std'], vid_metrics['fvd2_traj_conf95']

            elif self.condp > 0.0 and self.futrf == 0:                         # (1) Pred + (3) Gen

                vid_metrics['pred_mse'], vid_metrics['pred_psnr'], vid_metrics['pred_ssim'], vid_metrics['pred_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
                vid_metrics['pred_mse_std'], vid_metrics['pred_psnr_std'], vid_metrics['pred_ssim_std'], vid_metrics['pred_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
                vid_metrics['pred_mse_conf95'], vid_metrics['pred_psnr_conf95'], vid_metrics['pred_ssim_conf95'], vid_metrics['pred_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
                if calc_fvd1:
                    vid_metrics['pred_fvd'], vid_metrics['pred_fvd_traj_mean'], vid_metrics['pred_fvd_traj_std'], vid_metrics['pred_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

                if calc_fvd3:
                    vid_metrics['gen_fvd'], vid_metrics['gen_fvd_traj_mean'], vid_metrics['gen_fvd_traj_std'], vid_metrics['gen_fvd_traj_conf95'] = vid_metrics['fvd3'], vid_metrics['fvd3_traj_mean'], vid_metrics['fvd3_traj_std'], vid_metrics['fvd3_traj_conf95']

            elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and not self.prob_mask_sync:     # (1) Interp + (2) Pred + (3) Gen

                vid_metrics['interp_mse'], vid_metrics['interp_psnr'], vid_metrics['interp_ssim'], vid_metrics['interp_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
                vid_metrics['interp_mse_std'], vid_metrics['interp_psnr_std'], vid_metrics['interp_ssim_std'], vid_metrics['interp_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
                vid_metrics['interp_mse_conf95'], vid_metrics['interp_psnr_conf95'], vid_metrics['interp_ssim_conf95'], vid_metrics['interp_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
                if calc_fvd1:
                    vid_metrics['interp_fvd'], vid_metrics['interp_fvd_traj_mean'], vid_metrics['interp_fvd_traj_std'], vid_metrics['interp_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

                if second_calc:
                    vid_metrics['pred_mse'], vid_metrics['pred_psnr'], vid_metrics['pred_ssim'], vid_metrics['pred_lpips'] = vid_metrics['mse2'], vid_metrics['psnr2'], vid_metrics['ssim2'], vid_metrics['lpips2']
                    vid_metrics['pred_mse_std'], vid_metrics['pred_psnr_std'], vid_metrics['pred_ssim_std'], vid_metrics['pred_lpips_std'] = vid_metrics['mse2_std'], vid_metrics['psnr2_std'], vid_metrics['ssim2_std'], vid_metrics['lpips2_std']
                    vid_metrics['pred_mse_conf95'], vid_metrics['pred_psnr_conf95'], vid_metrics['pred_ssim_conf95'], vid_metrics['pred_lpips_conf95'] = vid_metrics['mse2_conf95'], vid_metrics['psnr2_conf95'], vid_metrics['ssim2_conf95'], vid_metrics['lpips2_conf95']
                    if calc_fvd2:
                        vid_metrics['pred_fvd'], vid_metrics['pred_fvd_traj_mean'], vid_metrics['pred_fvd_traj_std'], vid_metrics['pred_fvd_traj_conf95'] = vid_metrics['fvd2'], vid_metrics['fvd2_traj_mean'], vid_metrics['fvd2_traj_std'], vid_metrics['fvd2_traj_conf95']

                if calc_fvd3:
                    vid_metrics['gen_fvd'], vid_metrics['gen_fvd_traj_mean'], vid_metrics['gen_fvd_traj_std'], vid_metrics['gen_fvd_traj_conf95'] = vid_metrics['fvd3'], vid_metrics['fvd3_traj_mean'], vid_metrics['fvd3_traj_std'], vid_metrics['fvd3_traj_conf95']

            elif self.condp > 0.0 and self.futrf > 0 and self.futrp > 0.0 and self.prob_mask_sync:  # (1) Interp + (3) Gen

                vid_metrics['interp_mse'], vid_metrics['interp_psnr'], vid_metrics['interp_ssim'], vid_metrics['interp_lpips'] = vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
                vid_metrics['interp_mse_std'], vid_metrics['interp_psnr_std'], vid_metrics['interp_ssim_std'], vid_metrics['interp_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], vid_metrics['lpips_std']
                vid_metrics['interp_mse_conf95'], vid_metrics['interp_psnr_conf95'], vid_metrics['interp_ssim_conf95'], vid_metrics['interp_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics['ssim_conf95'], vid_metrics['lpips_conf95']
                if calc_fvd1:
                    vid_metrics['interp_fvd'], vid_metrics['interp_fvd_traj_mean'], vid_metrics['interp_fvd_traj_std'], vid_metrics['interp_fvd_traj_conf95'] = vid_metrics['fvd'], vid_metrics['fvd_traj_mean'], vid_metrics['fvd_traj_std'], vid_metrics['fvd_traj_conf95']

                if calc_fvd3:
                    vid_metrics['gen_fvd'], vid_metrics['gen_fvd_traj_mean'], vid_metrics['gen_fvd_traj_std'], vid_metrics['gen_fvd_traj_conf95'] = vid_metrics['fvd3'], vid_metrics['fvd3_traj_mean'], vid_metrics['fvd3_traj_std'], vid_metrics['fvd3_traj_conf95']

            logging.info(f"elapsed: {elapsed}, {format_p(vid_metrics)}")
            self.write_to_yaml(os.path.join(self.args.video_folder, 'vid_metrics.yml'), vid_metrics)

    def test(self):
        scorenet = get_model(self.config)
        scorenet = torch.nn.DataParallel(scorenet)

        if self.config.data.dataset.upper() == 'FFHQ':
            test_dataloader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.test.batch_size, self.config.data.image_size)
        else:
            dataset, test_dataset = get_dataset(self.args.data_path, self.config)
            test_dataloader = DataLoader(test_dataset, batch_size=self.config.test.batch_size, shuffle=True,
                                         num_workers=self.config.data.num_workers, drop_last=True)

        conditional = self.config.data.num_frames_cond > 0
        cond = None
        verbose = False
        for ckpt in tqdm(range(self.config.test.begin_ckpt, self.config.test.end_ckpt + 1, getattr(self.config.test, "freq", 5000)),
                              desc="processing ckpt:"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pt'),
                                map_location=self.config.device)
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(scorenet)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(scorenet)
            else:
                scorenet.load_state_dict(states[0])

            scorenet.eval()

            step = 0
            mean_loss = 0.
            mean_grad_norm = 0.
            average_grad_scale = 0.
            for x, y in test_dataloader:
                step += 1

                x = x.to(self.config.device)
                x = data_transform(self.config, x)

                x, cond, cond_mask = conditioning_fn(self.config, x, num_frames_pred=self.config.data.num_frames,
                                                     prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                     prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0),
                                                     conditional=conditional)

                with torch.no_grad():
                    test_loss = anneal_dsm_score_estimation(scorenet, x, labels=None, cond=cond, cond_mask=cond_mask,
                                                            loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                            gamma=getattr(self.config.model, 'gamma', False),
                                                            L1=getattr(self.config.training, 'L1', False),
                                                            all_frames=getattr(self.config.model, 'output_all_frames', False))
                    if verbose:
                        logging.info("step: {}, test_loss: {}".format(step, test_loss.item()))

                    mean_loss += test_loss.item()

            mean_loss /= step
            mean_grad_norm /= step
            average_grad_scale /= step

            logging.info("ckpt: {}, average test loss: {}".format(
                ckpt, mean_loss
            ))

    def fast_fid(self):
        ### Test the fids of ensembled checkpoints.
        ### Shouldn't be used for models with ema
        if self.config.fast_fid.ensemble:
            if self.config.model.ema:
                raise RuntimeError("Cannot apply ensembling to models with EMA.")
            self.fast_ensemble_fid()
            return

        scorenet = get_model(self.config)
        scorenet = torch.nn.DataParallel(scorenet)

        net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Sampler
        sampler = self.get_sampler()

        # sigmas = get_sigmas(self.config)
        # sigmas = sigmas_th.cpu().numpy()

        # If FFHQ tfrecord, reset dataloader
        if self.config.data.dataset.upper() == 'FFHQ':
            dataloader = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
        else:
            dataset, _ = get_dataset(self.args.data_path, self.config)
            dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                    num_workers=self.config.data.num_workers)
        data_iter = iter(dataloader)

        conditional = self.config.data.num_frames_cond > 0
        cond = None
        if conditional:
            if self.config.data.dataset.upper() == 'FFHQ':
                dataloader_cond = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
            else:
                dataset_cond, _ = get_dataset(self.args.data_path, self.config)
                dataloader_cond = DataLoader(dataset_cond, batch_size=self.config.sampling.batch_size, shuffle=True,
                                             num_workers=self.config.data.num_workers)
            data_iter_cond = iter(dataloader_cond)

        fids, precisions, recalls = {}, {}, {}
        for ckpt in tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, getattr(self.config.fast_fid, "freq", 5000)),
                              desc="Ckpt"):

            # Check for features
            if os.path.exists(os.path.join(self.args.image_folder, 'feats_{}.pt'.format(ckpt))):
                gen_samples = os.path.join(self.args.image_folder, 'feats_{}.pt'.format(ckpt))
                save_feats_path = None

            # Check for samples
            elif os.path.exists(os.path.join(self.args.image_folder, 'samples_{}.pt'.format(ckpt))):
                gen_samples = torch.load(os.path.join(self.args.image_folder, 'samples_{}.pt'.format(ckpt)))
                save_feats_path = os.path.join(self.args.image_folder, 'feats_{}.pt'.format(ckpt))

            # Generate samples
            else:

                states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pt'),
                                    map_location=self.config.device)

                if self.config.model.ema:
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(scorenet)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(scorenet)
                else:
                    scorenet.load_state_dict(states[0])

                scorenet.eval()

                num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
                # output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
                # os.makedirs(output_path, exist_ok=True)
                for i in tqdm(range(num_iters), desc="samples"):

                    # z
                    init_samples_shape = (self.config.fast_fid.batch_size, self.config.data.channels*self.config.data.num_frames,
                                          self.config.data.image_size, self.config.data.image_size)
                    if self.version == "SMLD":
                        z = torch.rand(init_samples_shape, device=self.config.device)
                        z = data_transform(self.config, z)
                    elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                        if getattr(self.config.model, 'gamma', False):
                            used_k, used_theta = net.k_cum[0], net.theta_t[0]
                            z = Gamma(torch.full(real.shape, used_k), torch.full(real.shape, 1 / used_theta)).sample().to(self.config.device)
                            z = z - used_k*used_theta
                        else:
                            z = torch.randn(init_samples_shape, device=self.config.device)

                    init_samples = z

                    if conditional:
                        try:
                            samples, _ = next(data_iter_cond)
                        except StopIteration:
                            if self.config.data.dataset.upper() == 'FFHQ':
                                dataloader_cond = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                            data_iter_cond = iter(dataloader_cond)
                            samples, _ = next(data_iter_cond)
                        samples = samples.to(self.config.device)
                        samples = data_transform(self.config, samples)
                        samples, cond, cond_mask = conditioning_fn(self.config, samples, conditional=conditional)
                        _, cond = samples[:len(init_samples)], cond[:len(init_samples)]

                    all_samples = sampler(init_samples, scorenet, cond=cond, cond_mask=cond_mask, final_only=True,
                                          n_steps_each=self.config.fast_fid.n_steps_each,
                                          step_lr=self.config.fast_fid.step_lr,
                                          verbose=self.config.fast_fid.verbose,
                                          denoise=self.config.sampling.denoise,
                                          subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                          clip_before=getattr(self.config.sampling, 'clip_before', True), log=True,
                                          gamma=getattr(self.config.model, 'gamma', False)).to('cpu')

                    final_samples = all_samples[-1].reshape(all_samples[-1].shape[0], self.config.data.channels*self.config.data.num_frames,
                                                            self.config.data.image_size, self.config.data.image_size)
                    final_samples = inverse_data_transform(self.config, final_samples)
                    gen_samples = final_samples if i == 0 else torch.cat([gen_samples, final_samples], dim=0)

                # Expand it out
                gen_samples = gen_samples.reshape(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)

                # Save samples
                torch.save(gen_samples, os.path.join(self.args.image_folder, 'samples_{}.pt'.format(ckpt)))
                nrow = ceil(np.sqrt(self.config.data.num_frames*100))
                image_grid = make_grid(gen_samples[:self.config.data.num_frames*100], nrow=nrow, pad_value=0.5)
                save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(ckpt)))

                save_feats_path = os.path.join(self.args.image_folder, 'feats_{}.pt'.format(ckpt))

            # FID
            if self.args.no_pr:
                stats_path = get_stats_path(getattr(self.config.fast_fid, 'dataset', self.config.data.dataset).upper(),
                                            self.args.stats_dir, download=self.args.stats_download)
                fid = get_fid(stats_path, gen_samples, self.config.device)
                fids[ckpt] = fid
                print("ckpt: {}, fid: {}".format(ckpt, fid))
                self.write_to_pickle(os.path.join(self.args.image_folder, 'fids.pickle'), fids)
                self.write_to_yaml(os.path.join(self.args.image_folder, 'fids.yml'), fids)

            # FID, precision, recall
            else:
                ds_feats_path = get_feats_path(getattr(self.config.fast_fid, 'dataset', self.config.data.dataset).upper(),
                                               self.args.feats_dir)
                k = self.config.fast_fid.pr_nn_k
                fid, precision, recall = get_fid_PR(ds_feats_path, gen_samples, self.config.device,
                                                    k=k, save_feats_path=save_feats_path)
                fids[ckpt], precisions[ckpt], recalls[ckpt] = fid, precision, recall
                print("ckpt: {}, fid: {}, precision: {}, recall: {}".format(ckpt, fid, precision, recall))

                self.write_to_pickle(os.path.join(self.args.image_folder, 'fids.pickle'), fids)
                self.write_to_yaml(os.path.join(self.args.image_folder, 'fids.yml'), fids)
                self.write_to_pickle(os.path.join(self.args.image_folder, f'precisions_k{k}.pickle'), precisions)
                self.write_to_yaml(os.path.join(self.args.image_folder, f'precisions_k{k}.yml'), precisions)
                self.write_to_pickle(os.path.join(self.args.image_folder, f'recalls_k{k}.pickle'), recalls)
                self.write_to_yaml(os.path.join(self.args.image_folder, f'recalls_k{k}.yml'), recalls)

    def fast_ensemble_fid(self):
        num_ensembles = 5
        scorenets = [NCSN(self.config).to(self.config.device) for _ in range(num_ensembles)]
        scorenets = [torch.nn.DataParallel(scorenet) for scorenet in scorenets]

        # sigmas = get_sigmas(self.config)
        # sigmas = sigmas_th.cpu().numpy()

        net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Sampler
        sampler = self.get_sampler()

        conditional = self.config.data.num_frames_cond > 0
        cond = None
        if conditional:
            if self.config.data.dataset.upper() == 'FFHQ':
                dataloader_cond = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
            else:
                dataset_cond, _ = get_dataset(self.args.data_path, self.config)
                dataloader_cond = DataLoader(dataset_cond, batch_size=self.config.sampling.batch_size, shuffle=True,
                                             num_workers=self.config.data.num_workers)
            data_iter_cond = iter(dataloader_cond)

        fids = {}
        for ckpt in tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, self.config.fast_fid.freq),
                              desc="processing ckpt"):
            begin_ckpt = max(self.config.fast_fid.begin_ckpt, ckpt - (num_ensembles - 1) * self.config.fast_fid.freq)
            index = 0
            for i in range(begin_ckpt, ckpt + self.config.fast_fid.freq, self.config.fast_fid.freq):
                states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{i}.pt'),
                                    map_location=self.config.device)
                scorenets[index].load_state_dict(states[0])
                scorenets[index].eval()
                index += 1

            def scorenet(x, labels, cond=None):
                num_ckpts = (ckpt - begin_ckpt) // self.config.fast_fid.freq + 1
                return sum([scorenets[i](x, labels, cond) for i in range(num_ckpts)]) / num_ckpts

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):

                # z
                init_samples_shape = (self.config.fast_fid.batch_size, self.config.data.channels*self.config.data.num_frames,
                                      self.config.data.image_size, self.config.data.image_size)
                if self.version == "SMLD":
                    z = torch.rand(init_samples_shape, device=self.config.device)
                    z = data_transform(self.config, z)
                elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
                    if getattr(self.config.model, 'gamma', False):
                        used_k, used_theta = net.k_cum[0], net.theta_t[0]
                        z = Gamma(torch.full(real.shape, used_k), torch.full(real.shape, 1 / used_theta)).sample().to(self.config.device)
                        z = z - used_k*used_theta
                    else:
                        z = torch.randn(init_samples_shape, device=self.config.device)

                init_samples = z

                if conditional:
                    try:
                        samples, _ = next(data_iter_cond)
                    except StopIteration:
                        if self.config.data.dataset.upper() == 'FFHQ':
                            dataloader_cond = FFHQ_TFRecordsDataLoader([self.args.data_path], self.config.sampling.batch_size, self.config.data.image_size)
                        data_iter_cond = iter(dataloader_cond)
                        samples, _ = next(data_iter_cond)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    samples, cond, _ = conditioning_fn(self.config, samples, conditional=conditional)
                    samples, cond = samples[:len(init_samples)], cond[:len(init_samples)]

                all_samples = sampler(init_samples, scorenet, cond=cond,
                                      n_steps_each=self.config.fast_fid.n_steps_each,
                                      step_lr=self.config.fast_fid.step_lr,
                                      verbose=self.config.fast_fid.verbose,
                                      denoise=self.config.sampling.denoise,
                                      subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                      clip_before=getattr(self.config.sampling, 'clip_before', True),
                                      log=True, gamma=getattr(self.config.model, 'gamma', False)).to('cpu')

                final_samples = inverse_data_transform(self.config, all_samples[-1])

                # Expand it out
                final_samples = final_samples.reshape(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)

            # FID
            if self.args.no_pr:
                stats_path = get_stats_path(getattr(self.config.fast_fid, 'dataset', self.config.data.dataset).upper(),
                                            self.args.stats_dir, download=self.args.stats_download)
                fid = get_fid(stats_path, final_samples, self.config.device)
                fids[ckpt] = fid
                print("ckpt: {}, fid: {}".format(ckpt, fid))
                self.write_to_pickle(os.path.join(self.args.image_folder, 'fids.pickle'), fids)
                self.write_to_yaml(os.path.join(self.args.image_folder, 'fids.yml'), fids)

            # FID, precision, recall
            else:
                feats_path = get_feats_path(getattr(self.config.fast_fid, 'dataset', self.config.data.dataset).upper(),
                                            self.args.feats_dir)
                k = self.config.fast_fid.pr_nn_k
                fid, precision, recall = get_fid_PR(feats_path, final_samples, self.config.device, k=k)
                fids[ckpt], precisions[ckpt], recalls[ckpt] = fid, precision, recall
                print("ckpt: {}, fid: {}, precision: {}, recall: {}".format(ckpt, fid, precision, recall))

                self.write_to_pickle(os.path.join(self.args.image_folder, 'fids.pickle'), fids)
                self.write_to_yaml(os.path.join(self.args.image_folder, 'fids.yml'), fids)
                self.write_to_pickle(os.path.join(self.args.image_folder, f'precisions_k{k}.pickle'), precisions)
                self.write_to_yaml(os.path.join(self.args.image_folder, f'precisions_k{k}.yml'), precisions)
                self.write_to_pickle(os.path.join(self.args.image_folder, f'recalls_k{k}.pickle'), recalls)
                self.write_to_yaml(os.path.join(self.args.image_folder, f'recalls_k{k}.yml'), recalls)

    def get_sampler(self):
        # Sampler
        if self.version == "SMLD":
            consistent = getattr(self.config.sampling, 'consistent', False)
            sampler = anneal_Langevin_dynamics_consistent if consistent else anneal_Langevin_dynamics
        elif self.version == "DDPM":
            sampler = partial(ddpm_sampler, config=self.config)
        elif self.version == "DDIM":
            sampler = partial(ddim_sampler, config=self.config)
        elif self.version == "FPNDM":
            sampler = partial(FPNDM_sampler, config=self.config)

        return sampler

    def init_meters(self):
        success = self.load_meters()
        if not success:
            self.epochs = RunningAverageMeter()
            self.losses_train, self.losses_test = RunningAverageMeter(), RunningAverageMeter()
            self.lr_meter, self.grad_norm = RunningAverageMeter(), RunningAverageMeter()
            self.time_train, self.time_elapsed = RunningAverageMeter(), RunningAverageMeter()
            self.time_train_prev = self.time_elapsed_prev = 0
            self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds = RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter()
            self.mses2, self.psnrs2, self.ssims2, self.lpipss2, self.fvds2 = RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter()
            self.fvds3 = RunningAverageMeter()
            self.best_mse = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_psnr = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_ssim = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_lpips = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_mse2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_psnr2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_ssim2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_lpips2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd3 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}

    def load_meters(self):
        meters_pkl = os.path.join(self.args.log_path, 'meters.pkl')
        if not os.path.exists(meters_pkl):
            print(f"{meters_pkl} does not exist! Returning.")
            return False
        with open(meters_pkl, "rb") as f:
            a = pickle.load(f)
        # Load
        self.epochs = a['epochs']
        self.losses_train = a['losses_train']
        self.losses_test = a['losses_test']
        self.lr_meter = a['lr_meter']
        self.grad_norm = a['grad_norm']
        self.time_train = a['time_train']
        self.time_train_prev = a['time_train'].val or 0
        self.time_elapsed = a['time_elapsed']
        self.time_elapsed_prev = a['time_elapsed'].val or 0
        try:
            self.mses = a['mses']
            self.psnrs = a['psnrs']
            self.ssims = a['ssims']
            self.lpipss = a['lpips']
            self.fvds = a['fvds']
            self.best_mse = a['best_mse']
            self.best_psnr = a['best_psnr']
            self.best_ssim = a['best_ssim']
            self.best_lpips = a['best_lpips']
            self.best_fvd = a['best_fvd']
        except:
            self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds = RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter()
            self.best_mse = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_psnr = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_ssim = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_lpips = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
        try:
            self.mses2 = a['mses2']
            self.psnrs2 = a['psnrs2']
            self.ssims2 = a['ssims2']
            self.lpipss2 = a['lpips2']
            self.fvds2 = a['fvds2']
            self.fvds3 = a['fvds3']
            self.best_mse2 = a['best_mse2']
            self.best_psnr2 = a['best_psnr2']
            self.best_ssim2 = a['best_ssim2']
            self.best_lpips2 = a['best_lpips2']
            self.best_fvd2 = a['best_fvd2']
            self.best_fvd3 = a['best_fvd3']
        except:
            self.mses2, self.psnrs2, self.ssims2, self.lpipss2, self.fvds2 = RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter()
            self.fvds3 = RunningAverageMeter()
            self.best_mse2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_psnr2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_ssim2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_lpips2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd3 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
        return True

    def save_meters(self):
        meters_pkl = os.path.join(self.args.log_path, 'meters.pkl')
        with open(meters_pkl, "wb") as f:
            pickle.dump({
                'epochs': self.epochs,
                'losses_train': self.losses_train,
                'losses_test': self.losses_test,
                'lr_meter' : self.lr_meter,
                'grad_norm' : self.grad_norm,
                'time_train': self.time_train,
                'time_elapsed': self.time_elapsed,
                'mses': self.mses,
                'psnrs': self.psnrs,
                'ssims': self.ssims,
                'lpips': self.lpipss,
                'fvds': self.fvds,
                'best_mse': self.best_mse,
                'best_psnr': self.best_psnr,
                'best_ssim': self.best_ssim,
                'best_lpips': self.best_lpips,
                'best_fvd': self.best_fvd,
                'mses2': self.mses2,
                'psnrs2': self.psnrs2,
                'ssims2': self.ssims2,
                'lpips2': self.lpipss2,
                'fvds2': self.fvds2,
                'best_mse2': self.best_mse2,
                'best_psnr2': self.best_psnr2,
                'best_ssim2': self.best_ssim2,
                'best_lpips2': self.best_lpips2,
                'best_fvd2': self.best_fvd2,
                'best_fvd3': self.best_fvd3,
                },
                f, protocol=pickle.HIGHEST_PROTOCOL)

    def write_to_pickle(self, pickle_file, my_dict):
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as handle:
                old_dict = pickle.load(handle)
            for key in my_dict.keys():
                old_dict[key] = my_dict[key]
            my_dict = {}
            for key in sorted(old_dict.keys()):
                my_dict[key] = old_dict[key]
        with open(pickle_file, 'wb') as handle:
            pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def write_to_yaml(self, yaml_file, my_dict):
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                old_dict = yaml.load(f, Loader=yaml.FullLoader)
            for key in my_dict.keys():
                old_dict[key] = my_dict[key]
            my_dict = {}
            for key in sorted(old_dict.keys()):
                my_dict[key] = old_dict[key]
        with open(yaml_file, 'w') as f:
            yaml.dump(my_dict, f, default_flow_style=False)
