# SMLD: s = -1/sigma * z
# DDPM: s = -1/sqrt(1 - alpha) * z
# All `scorenet` models return z, not s!

import torch
import logging
import numpy as np

from functools import partial
from scipy.stats import hmean
from torch.distributions.gamma import Gamma
from tqdm import tqdm
from . import pndm


def get_sigmas(config):

    T = getattr(config.model, 'num_classes')

    if config.model.sigma_dist == 'geometric':
        return torch.logspace(np.log10(config.model.sigma_begin), np.log10(config.model.sigma_end),
                              T).to(config.device)

    elif config.model.sigma_dist == 'linear':
        return torch.linspace(config.model.sigma_begin, config.model.sigma_end,
                              T).to(config.device)

    elif config.model.sigma_dist == 'cosine':
        t = torch.linspace(T, 0, T+1)/T
        s = 0.008
        f = torch.cos((t + s)/(1 + s) * np.pi/2)**2
        return f[:-1]/f[-1]

    else:
        raise NotImplementedError('sigma distribution not supported')


@torch.no_grad()
def FPNDM_sampler(x_mod, scorenet, cond=None, final_only=False, denoise=True, subsample_steps=None,
                 verbose=False, log=True, clip_before=True, t_min=-1, gamma=False, **kwargs):

    net = scorenet.module if hasattr(scorenet, 'module') else scorenet

    # schedule = getattr(config.model, 'sigma_dist', 'linear')
    # if schedule == 'linear':
    #     betas = get_sigmas(config)
    #     alphas = torch.cumprod(1 - betas.flip(0), 0).flip(0)
    #     alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
    # elif schedule == 'cosine':
    #     alphas = get_sigmas(config)
    #     alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
    #     betas = (1 - alphas/alphas_prev).clip_(0, 0.999)
    alphas, alphas_prev, betas = net.alphas, net.alphas_prev, net.betas
    steps = np.arange(len(betas))

    net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

    alphas_old = alphas.flip(0)
    
    # New ts (see https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py)
    skip = len(alphas) // subsample_steps
    steps = range(0, len(alphas), skip)
    steps_next = [-1] + list(steps[:-1])

    #steps_next = list(steps[1:]) + [steps[-1] + 1]
    #print(steps)
    #print(steps_next)
    #alphas_old = torch.cat([alphas_old, torch.tensor([1.0]).to(alphas)])

    steps = torch.tensor(steps, device=alphas.device)
    steps_next = torch.tensor(steps_next, device=alphas.device)
    #alphas = alphas.index_select(0, steps)
    alphas_next = alphas.index_select(0, steps_next + 1)
    alphas = alphas.index_select(0, steps + 1)
    #print(alphas_next)
    #print(alphas)

    images = []
    scorenet = partial(scorenet, cond=cond)

    L = len(steps)
    ets = []
    for i, step in enumerate(steps):

        t_ = (steps[i] * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        t_next = (steps_next[i] * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        #print(alphas_next[i])
        #print(alphas[i])
        #print(alphas_old[t_next.long() + 1][0])
        #print(alphas_old[t_.long() + 1][0])
        x_mod, ets = pndm.gen_order_4(x_mod, t_, t_next, model=scorenet, alphas_cump=alphas_old, ets=ets, clip_before=clip_before)

        if not final_only:
            images.append(x_mod.to('cpu'))

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)


@torch.no_grad()
def ddim_sampler(x_mod, scorenet, cond=None, final_only=False, denoise=True, subsample_steps=None,
                 verbose=False, log=True, clip_before=True, t_min=-1, gamma=False, **kwargs):

    net = scorenet.module if hasattr(scorenet, 'module') else scorenet

    # schedule = getattr(config.model, 'sigma_dist', 'linear')
    # if schedule == 'linear':
    #     betas = get_sigmas(config)
    #     alphas = torch.cumprod(1 - betas.flip(0), 0).flip(0)
    #     alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
    # elif schedule == 'cosine':
    #     alphas = get_sigmas(config)
    #     alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
    #     betas = (1 - alphas/alphas_prev).clip_(0, 0.999)
    alphas, alphas_prev, betas = net.alphas, net.alphas_prev, net.betas
    if gamma:
        ks_cum, thetas = net.k_cum, net.theta_t
    steps = np.arange(len(betas))

    net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

    # New ts (see https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py)
    if subsample_steps is not None:
        if subsample_steps < len(alphas):
            skip = len(alphas) // subsample_steps
            steps = range(0, len(alphas), skip)
            steps = torch.tensor(steps, device=alphas.device)
            # new alpha, beta, alpha_prev
            alphas = alphas.index_select(0, steps)
            alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
            betas = 1.0 - torch.div(alphas, alphas_prev) # for some reason we lose a bit of precision here
            if gamma:
                ks_cum = ks_cum.index_select(0, steps)
                thetas = thetas.index_select(0, steps)

    images = []
    scorenet = partial(scorenet, cond=cond)
    x_transf = False

    L = len(steps)
    for i, step in enumerate(steps):

        if step < t_min*len(alphas): # otherwise, wait until it happens
            continue

        if not x_transf and t_min > 0: # we must add noise to the previous frame
            if gamma:
                z = Gamma(torch.full(x_mod.shape[1:], ks_cum[i]),
                          torch.full(x_mod.shape[1:], 1 / thetas[i])).sample((x_mod.shape[0],)).to(x_mod.device)
                z = (z - ks_cum[i]*thetas[i])/((1 - alphas[i]).sqrt())
            else:
                z = torch.randn_like(x_mod)
            x_mod = alphas[i].sqrt() * x_mod + (1 - alphas[i]).sqrt() * z
        x_transf = True

        c_beta, c_alpha, c_alpha_prev = betas[i], alphas[i], alphas_prev[i]
        labels = (step * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        grad = scorenet(x_mod, labels)

        #print(alphas_prev[i])
        #print(alphas[i])

        x0 = (1 / c_alpha.sqrt()) * (x_mod - (1 - c_alpha).sqrt() * grad)
        if clip_before:
            x0 = x0.clip_(-1, 1)
        x_mod = c_alpha_prev.sqrt() * x0 + (1 - c_alpha_prev).sqrt() * grad

        if not final_only:
            images.append(x_mod.to('cpu'))

        if i == 0 or (i+1) % max(L//10, 1) == 0:

            if verbose or log:
                grad = -1/(1 - c_alpha).sqrt() * grad
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()
                grad_mean_norm = torch.norm(grad.mean(dim=0).reshape(-1)) ** 2 * (1 - c_alpha)

            if verbose:
                print("{}: {}/{}, grad_norm: {}, image_norm: {}, grad_mean_norm: {}".format(
                    "DDIM gamma" if gamma else "DDIM", i+1, L, grad_norm.item(), image_norm.item(), grad_mean_norm.item()))
            if log:
                logging.info("{}: {}/{}, grad_norm: {}, image_norm: {}, grad_mean_norm: {}".format(
                    "DDIM gamma" if gamma else "DDIM", i+1, L, grad_norm.item(), image_norm.item(), grad_mean_norm.item()))

        # # If last step, don't add noise
        # last_step = i + 1 == L
        # if last_step:
        #     continue

    # Denoise
    if denoise: # x + batch_mul(std ** 2, score_fn(x, eps_t))
        last_noise = ((L - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod = x_mod - (1 - alphas[-1]).sqrt() * scorenet(x_mod, last_noise)
        if not final_only:
            images.append(x_mod.to('cpu'))

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)


@torch.no_grad()
def ddpm_sampler(x_mod, scorenet, cond=None, just_beta=False, final_only=False, denoise=True, subsample_steps=None,
                 same_noise=False, noise_val=None, frac_steps=None, verbose=False, log=False, clip_before=True, 
                 t_min=-1, gamma=False, **kwargs):

    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    # schedule = getattr(config.model, 'sigma_dist', 'linear')
    # if schedule == 'linear':
    #     betas = get_sigmas(config)
    #     alphas = torch.cumprod(1 - betas.flip(0), 0).flip(0)
    #     alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
    # elif schedule == 'cosine':
    #     alphas = get_sigmas(config)
    #     alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
    #     betas = (1 - alphas/alphas_prev).clip_(0, 0.999)
    alphas, alphas_prev, betas = net.alphas, net.alphas_prev, net.betas
    steps = np.arange(len(betas))
    if gamma:
        ks_cum, thetas = net.k_cum, net.theta_t

    net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

    # New ts (see https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py)
    if subsample_steps is not None:
        if subsample_steps < len(alphas):
            skip = len(alphas) // subsample_steps
            steps = range(0, len(alphas), skip)
            steps = torch.tensor(steps, device=alphas.device)
            # new alpha, beta, alpha_prev
            alphas = alphas.index_select(0, steps)
            alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
            betas = 1.0 - torch.div(alphas, alphas_prev) # for some reason we lose a bit of precision here
            if gamma:
                ks_cum = ks_cum.index_select(0, steps)
                thetas = thetas.index_select(0, steps)

    # Subsample steps : keep range but decrease number of steps
    #if subsample_steps is not None:
    #    steps = torch.round(torch.linspace(0, len(alphas)-1, subsample_steps)).long()
    #    alphas = alphas[steps]
    #    alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
    #    betas = 1 - alphas/alphas_prev

    # Frac steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        alphas = alphas[steps]
        alphas_prev = alphas_prev[steps]
        betas = betas[steps]
        if gamma:
            ks_cum = ks_cum[steps]
            thetas = thetas[steps]

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []
    scorenet = partial(scorenet, cond=cond)
    x_transf = False

    L = len(steps)
    for i, step in enumerate(steps):

        if step < t_min*len(alphas): # wait until it happens
            continue

        if not x_transf and t_min > 0: # we must add noise to the previous frame
            if gamma:
                z = Gamma(torch.full(x_mod.shape[1:], ks_cum[i]),
                          torch.full(x_mod.shape[1:], 1 / thetas[i])).sample((x_mod.shape[0],)).to(x_mod.device)
                z = (z - ks_cum[i]*thetas[i]) / (1 - alphas[i]).sqrt()
            else:
                z = torch.randn_like(x_mod)
            x_mod = alphas[i].sqrt() * x_mod + (1 - alphas[i]).sqrt() * z
        x_transf = True

        c_beta, c_alpha, c_alpha_prev = betas[i], alphas[i], alphas_prev[i]
        labels = (step * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        grad = scorenet(x_mod, labels)

        # x_mod = 1 / (1 - c_beta).sqrt() * (x_mod + c_beta / (1 - c_alpha).sqrt() * grad)
        x0 = (1 / c_alpha.sqrt()) * (x_mod - (1 - c_alpha).sqrt() * grad)
        if clip_before:
            x0 = x0.clip_(-1, 1)
        x_mod = (c_alpha_prev.sqrt() * c_beta / (1 - c_alpha)) * x0 + ((1 - c_beta).sqrt() * (1 - c_alpha_prev) / (1 - c_alpha)) * x_mod

        if not final_only:
            images.append(x_mod.to('cpu'))

        if i == 0 or (i+1) % max(L//10, 1) == 0:

            if verbose or log:
                grad = -1/(1 - c_alpha).sqrt() * grad
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()
                grad_mean_norm = torch.norm(grad.mean(dim=0).reshape(-1)) ** 2 * (1 - c_alpha)

            if verbose:
                print("{}: {}/{}, grad_norm: {}, image_norm: {}, grad_mean_norm: {}".format(
                    "DDPM gamma" if gamma else "DDPM", i+1, L, grad_norm.item(), image_norm.item(), grad_mean_norm.item()))
            if log:
                logging.info("{}: {}/{}, grad_norm: {}, image_norm: {}, grad_mean_norm: {}".format(
                    "DDPM gamma" if gamma else "DDPM", i+1, L, grad_norm.item(), image_norm.item(), grad_mean_norm.item()))

        # If last step, don't add noise
        last_step = i + 1 == L
        if last_step:
            continue

        # Add noise
        if same_noise:
            noise = noise_val
        else:
            if gamma:
                z = Gamma(torch.full(x_mod.shape[1:], ks_cum[i]),
                          torch.full(x_mod.shape[1:], 1 / thetas[i])).sample((x_mod.shape[0],)).to(x_mod.device)
                noise = (z - ks_cum[i]*thetas[i])/((1 - alphas[i]).sqrt())
            else:
                noise = torch.randn_like(x_mod)
        if just_beta:
            x_mod += c_beta.sqrt() * noise
        else:
            x_mod += ((1 - c_alpha_prev) / (1 - c_alpha) * c_beta).sqrt() * noise

    # Denoise
    if denoise:
        last_noise = ((L - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod = x_mod - (1 - alphas[-1]).sqrt() * scorenet(x_mod, last_noise)
        if not final_only:
            images.append(x_mod.to('cpu'))

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)


@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, cond=None, n_steps_each=200, step_lr=0.000008,
                             final_only=False, denoise=True, harm_mean=False,
                             same_noise=False, noise_val=None, frac_steps=None,
                             verbose=False, log=False, t_min=-1, **kwargs):
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    sigmas = net.sigmas
    steps = np.arange(len(sigmas))

    L = len(sigmas)
    if harm_mean:
        sigmas_hmean = hmean(sigmas.cpu())

    # Sub steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        sigmas = sigmas[steps]

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []

    scorenet = partial(scorenet, cond=cond)

    for c, sigma in enumerate(sigmas):
        labels = (torch.ones(x_mod.shape[0], device=x_mod.device) * c).long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):

            grad = scorenet(x_mod, labels)
            if harm_mean:
                grad = grad * sigmas_hmean / sigma

            if same_noise:
                noise = noise_val
            else:
                z = torch.randn_like(x_mod)
                noise = z
            x_mod = x_mod - step_size / sigma * grad + (step_size * 2.).sqrt() * noise

            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            snr = (step_size / 2.).sqrt() * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).reshape(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))

            if (c == 0 and s == 0) or (c*n_steps_each+s+1) % max((L*n_steps_each)//10, 1) == 0:
                if verbose:
                    print("ALS level: {:.04f}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        (c*n_steps_each+s+1)/(L*n_steps_each), step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
                if log:
                    logging.info("ALS level: {:.04f}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        (c*n_steps_each+s+1)/(L*n_steps_each), step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = ((len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod = x_mod - sigmas[-1] * scorenet(x_mod, last_noise)
        if not final_only:
            images.append(x_mod.to('cpu'))

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)


@torch.no_grad()
def sparse_anneal_Langevin_dynamics(x_mod_sparse, sparsity, scorenet, cond=None, n_steps_each=200, step_lr=0.000008,
                                    final_only=False, denoise=True, harm_mean=False,
                                    same_noise=False, noise_val=None, frac_steps=None,
                                    verbose=False, log=False, **kwargs):
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    sigmas = net.sigmas
    steps = np.arange(len(sigmas))

    L = len(sigmas)
    if harm_mean:
        sigmas_hmean = hmean(sigmas.cpu())

    # Sub steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        sigmas = sigmas[steps]

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []
    x_mod = x_mod_sparse.clone()

    scorenet = partial(scorenet, cond=cond)

    for c, sigma in enumerate(sigmas):
        labels = (torch.ones(x_mod.shape[0], device=x_mod.device) * c).long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):

            grad = scorenet(x_mod, labels)
            if harm_mean:
                grad = grad * sigmas_hmean / sigma

            if same_noise:
                noise = noise_val
            else:
                z = torch.randn_like(x_mod)
                noise = z

            x_mod = x_mod - step_size / sigma * grad + (step_size * 2.).sqrt() * noise
            x_mod_sparse = x_mod_sparse - step_size / sigma * (1/sparsity * grad) + (step_size * 2.).sqrt() * (sparsity * noise)

            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            snr = (step_size / 2.).sqrt() * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).reshape(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod_sparse.to('cpu'))

            if (c == 0 and s == 0) or (c*n_steps_each+s+1) % max((L*n_steps_each)//10, 1) == 0:
                if verbose:
                    print("ALS level: {:.04f}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        (c*n_steps_each+s+1)/(L*n_steps_each), step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
                if log:
                    logging.info("ALS level: {:.04f}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        (c*n_steps_each+s+1)/(L*n_steps_each), step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = ((len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod_sparse = x_mod_sparse - sigmas[-1] * sparsity * scorenet(x_mod, last_noise)
        if not final_only:
            images.append(x_mod_sparse.to('cpu'))

    if final_only:
        return x_mod_sparse.unsqueeze(0)
    else:
        return torch.stack(images)


@torch.no_grad()
def anneal_Langevin_dynamics_consistent(x_mod, scorenet, cond=None, n_steps_each=200, step_lr=0.000008,
                                        final_only=False, denoise=True, harm_mean=False,
                                        same_noise=False, noise_val=None, frac_steps=None,
                                        verbose=False, log=False, **kwargs):
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    sigmas = net.sigmas
    steps = np.arange(len(sigmas))

    L = len(sigmas)
    sigma_begin = sigmas[0].cpu().item()
    sigma_end = sigmas[-1].cpu().item()
    consistent_sigmas = np.geomspace(sigma_begin, sigma_end, (L - 1) * n_steps_each + 1)

    smallest_invgamma = consistent_sigmas[-1] / consistent_sigmas[-2]
    lowerbound = sigmas[-1] ** 2 * (1 - smallest_invgamma)
    higherbound = sigmas[-1] ** 2 * (1 + smallest_invgamma)
    assert lowerbound < step_lr < higherbound, f"Could not satisfy {lowerbound} < {step_lr} < {higherbound}"

    eta = step_lr / (sigmas[-1] ** 2)

    if harm_mean:
        sigmas_hmean = hmean(consistent_sigmas)

    # Sub steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        consistent_sigmas = consistent_sigmas[steps]

    consistent_L = len(consistent_sigmas)
    iter_consistent_sigmas = iter(consistent_sigmas)
    next_sigma = next(iter_consistent_sigmas)

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []

    scorenet = partial(scorenet, cond=cond)

    for c in range(consistent_L):

        c_sigma = next_sigma
        used_sigmas = torch.tensor([c_sigma]*len(x_mod)).reshape(len(x_mod), *([1] * len(x_mod.shape[1:]))).float().to(x_mod.device)
        grad = scorenet(x_mod, used_sigmas, y_is_label=False)

        if harm_mean:
            grad = grad * sigmas_hmean / used_sigmas

        x_mod -= eta * c_sigma * grad
        if not final_only:
            images.append(x_mod.to('cpu'))

        last_step = c + 1 == consistent_L
        if last_step:

            if denoise:
                last_noise = ((len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
                x_mod = x_mod - sigmas[-1] * scorenet(x_mod, last_noise)
                if not final_only:
                    images.append(x_mod.to('cpu'))

                continue

        next_sigma = next(iter_consistent_sigmas)
        gamma = c_sigma/next_sigma
        beta = (1 - (gamma*(1 - eta))**2).sqrt()
        if same_noise:
            noise = noise_val
        else:
            z = torch.randn_like(x_mod)
            noise = z
        x_mod += beta * next_sigma * noise

        if c % n_steps_each == 0:

            if verbose or log:
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                snr = eta * gamma * c_sigma / beta * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).reshape(-1)) ** 2 * c_sigma ** 2

            if verbose:
                print("CAS level: {:.04f}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c/consistent_L, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
            if log:
                logging.info("CAS level: {:.04f}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c/consistent_L, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)

@torch.no_grad()
def sparse_anneal_Langevin_dynamics_consistent(x_mod_sparse, sparsity, scorenet, cond=None, n_steps_each=200, step_lr=0.000008,
                                               final_only=False, denoise=True, harm_mean=False,
                                               same_noise=False, noise_val=None, frac_steps=None,
                                               verbose=False, log=False, **kwargs):
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    sigmas = net.sigmas
    steps = np.arange(len(sigmas))

    L = len(sigmas)
    sigma_begin = sigmas[0].cpu().item()
    sigma_end = sigmas[-1].cpu().item()
    consistent_sigmas = np.geomspace(sigma_begin, sigma_end, (L - 1) * n_steps_each + 1)

    smallest_invgamma = consistent_sigmas[-1] / consistent_sigmas[-2]
    lowerbound = sigmas[-1] ** 2 * (1 - smallest_invgamma)
    higherbound = sigmas[-1] ** 2 * (1 + smallest_invgamma)
    assert lowerbound < step_lr < higherbound, f"Could not satisfy {lowerbound} < {step_lr} < {higherbound}"

    eta = step_lr / (sigmas[-1] ** 2)

    if harm_mean:
        sigmas_hmean = hmean(consistent_sigmas)

    # Sub steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        consistent_sigmas = consistent_sigmas[steps]

    consistent_L = len(consistent_sigmas)
    iter_consistent_sigmas = iter(consistent_sigmas)
    next_sigma = next(iter_consistent_sigmas)

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []
    x_mod = x_mod_sparse.clone()

    scorenet = partial(scorenet, cond=cond)

    for c in range(consistent_L):

        c_sigma = next_sigma
        used_sigmas = torch.tensor([c_sigma]*len(x_mod)).reshape(len(x_mod), *([1] * len(x_mod.shape[1:]))).float().to(x_mod.device)
        grad = scorenet(x_mod, used_sigmas, y_is_label=False)

        if harm_mean:
            grad = grad * sigmas_hmean / used_sigmas

        x_mod += eta * c_sigma**2 * grad
        if not final_only:
            images.append(x_mod.to('cpu'))

        last_step = c + 1 == consistent_L
        if last_step:

            if denoise:
                last_noise = ((len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
                x_mod = x_mod + sigmas[-1] * scorenet(x_mod, last_noise)
                x_mod_sparse = x_mod_sparse + sigmas[-1] * 1/sparsity * scorenet(x_mod, last_noise)
                if not final_only:
                    images.append(x_mod.to('cpu'))

                continue

        next_sigma = next(iter_consistent_sigmas)
        gamma = c_sigma/next_sigma
        beta = (1 - (gamma*(1 - eta))**2).sqrt()
        if same_noise:
            noise = noise_val
        else:
            z = torch.randn_like(x_mod)
            noise = z
        x_mod += next_sigma * beta * noise
        x_mod_sparse += next_sigma * beta * sparsity * noise

        if c % n_steps_each == 0:

            if verbose or log:
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                snr = eta * gamma * c_sigma / beta * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).reshape(-1)) ** 2 * c_sigma ** 2

            if verbose:
                print("CAS level: {:.04f}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c/consistent_L, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
            if log:
                logging.info("CAS level: {:.04f}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c/consistent_L, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if final_only:
        return x_mod_sparse.unsqueeze(0)
    else:
        return torch.stack(images)


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, image_size, n_steps_each=100,
                                        step_lr=0.000008, log=False, cond=None, **kwargs):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """
    sigmas = scorenet.module.sigmas if hasattr(scorenet, 'module') else scorenet.sigmas

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().reshape(-1, 3, image_size, image_size)
    x_mod = x_mod.reshape(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]

    scorenet = partial(scorenet, cond=cond)

    for c, sigma in tqdm(enumerate(sigmas), total=len(sigmas)):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for s in range(n_steps_each):
            images.append(x_mod.to('cpu'))
            corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
            x_mod[:, :, :, :cols] = corrupted_half_image
            noise = torch.randn_like(x_mod) * (step_size * 2.).sqrt()
            grad = scorenet(x_mod, labels)
            x_mod = x_mod + step_size * grad + noise
            print("class: {}, step_size: {}, mean {}, max {}".format(
                c, step_size, grad.abs().mean(), grad.abs().max()))
            if log:
                logging.info("class: {}, step_size: {}, mean {}, max {}".format(
                    c, step_size, grad.abs().mean(), grad.abs().max()))

    return torch.stack(images)


@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, log=False, cond=None, **kwargs):
    sigmas = scorenet.module.sigmas if hasattr(scorenet, 'module') else scorenet.sigmas

    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    scorenet = partial(scorenet, cond=cond)

    for c, sigma in tqdm(enumerate(sigmas), total=len(sigmas)):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * (step_size * 2.).sqrt()

            snr = (step_size / 2.).sqrt() * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))
            if log:
                logging.info("level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                    c, step_size, image_norm.item(), grad_norm.item(), snr.item()))

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)
