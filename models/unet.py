import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.gamma import Gamma

from . import get_sigmas

__all__ = ['UNet_SMLD', 'UNet_DDPM']


def default_init(module, scale):
    if scale == 0:
        scale = 1e-10
    # tf: sqrt(3*sc/((f_i+f_o)/2)) = sqrt(6*sc/(f_i+f_o)) = sqrt(sc)sqrt(6/(f_i+f_o)) => gain = sqrt(sc)
    torch.nn.init.xavier_uniform_(module.weight, math.sqrt(scale))
    torch.nn.init.zeros_(module.bias)


def init_weights(module, scale=1, module_is_list=False):
    if module_is_list:
        for module_ in module.modules():
            if isinstance(module_, nn.Conv2d) or isinstance(module_, nn.Linear):
                default_init(module_, scale)
    else:
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            default_init(module, scale)


class Swish(nn.Module):
    """
    Swish out-performs Relu for deep NN (more than 40 layers). Although, the performance of relu and swish model
    degrades with increasing batch size, swish performs better than relu.
    https://jmlb.github.io/ml/2017/12/31/swish_activation_function/ (December 31th 2017)
    """

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def Normalize(num_channels):
    return nn.GroupNorm(eps=1e-6, num_groups=32, num_channels=num_channels)


class Nin(nn.Module):
    """ Shared weights """

    def __init__(self, channel_in: int, channel_out: int, init_scale=1.):
        super().__init__()
        self.channel_out = channel_out
        self.weights = nn.Parameter(torch.zeros(channel_out, channel_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weights, math.sqrt(1e-10 if init_scale == 0. else init_scale))
        self.bias = nn.Parameter(torch.zeros(channel_out), requires_grad=True)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        bs, _, width, _ = x.shape
        res = torch.bmm(self.weights.repeat(bs, 1, 1), x.flatten(2)) + self.bias.unsqueeze(0).unsqueeze(-1)
        return res.view(bs, self.channel_out, width, width)


class ResnetBlock(nn.Module):
    def __init__(self, channel_in, channel_out, dropout, tembdim, conditional=False):
        super().__init__()
        self.dropout = dropout
        self.nonlinearity = Swish()
        self.normalize0 = Normalize(channel_in)
        self.conv0 = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1)
        init_weights(self.conv0)
        self.conditional = conditional

        if conditional:
            self.dense = nn.Linear(tembdim, channel_out)
            init_weights(self.dense)

        self.normalize1 = Normalize(channel_out)
        self.conv1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
        init_weights(self.conv1, scale=0)

        if channel_in != channel_out:
            self.nin = Nin(channel_in, channel_out)
        else:
            self.nin = nn.Identity()
        self.channel_in = channel_in

    def forward(self, x, temb=None):
        h = self.nonlinearity(self.normalize0(x))
        h = self.conv0(h)
        if temb is not None and self.conditional:
            h += self.dense(temb).unsqueeze(-1).unsqueeze(-1)

        h = self.nonlinearity(self.normalize1(h))
        return self.nin(x) + self.conv1(self.dropout(h))


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Q = Nin(channels, channels)
        self.K = Nin(channels, channels)
        self.V = Nin(channels, channels)
        self.OUT = Nin(channels, channels, init_scale=0.)  # ensure identity at init

        self.normalize = Normalize(channels)
        self.c = channels

    def forward(self, x):
        h = self.normalize(x)
        q, k, v = self.Q(h), self.K(h), self.V(h)
        w = torch.einsum('abcd,abef->acdef', q, k) * (1 / math.sqrt(self.c))

        batch_size, width, *_ = w.shape
        w = F.softmax(w.view(batch_size, width, width, width * width), dim=-1)
        w = w.view(batch_size, *[width] * 4)
        h = torch.einsum('abcde,afde->afbc', w, v)
        return x + self.OUT(h)


class Upsample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        init_weights(self.conv)

    def forward(self, x):
        return self.conv(self.up(x))


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def get_timestep_embedding(timesteps, embedding_dim: int = 128):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.ZeroPad2d((0, 1, 0, 0))(emb)

    assert [*emb.shape] == [timesteps.shape[0],
                            embedding_dim], f"{emb.shape}, {str([timesteps.shape[0], embedding_dim])}"
    return emb


def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()

        self.locals = [config]

        self.config = config
        self.n_channels = n_channels = config.data.channels
        self.ch = ch = config.model.ngf
        self.mode = mode = getattr(config, 'mode', 'deep')
        assert mode in ['deep', 'deeper', 'deepest']
        self.dropout = nn.Dropout2d(p=getattr(config.model, 'dropout', 0.0))
        self.time_conditional = time_conditional = getattr(config.model, 'time_conditional', False)

        self.version = getattr(config.model, 'version', 'SMLD').upper()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled

        self.num_frames = num_frames = getattr(config.data, 'num_frames', 1)
        self.num_frames_cond = num_frames_cond = getattr(config.data, 'num_frames_cond', 0) + getattr(config.data, "num_frames_future", 0)

        # TODO make sure channel is in dimensions 1 [bs x c x 32 x 32]
        ResnetBlock_ = partialclass(ResnetBlock, dropout=self.dropout, tembdim=ch * 4, conditional=time_conditional)

        if mode == 'deepest':
            ch_mult = [ch * n for n in (1, 2, 2, 2, 4, 4)]
        elif mode == 'deeper':
            ch_mult = [ch * n for n in (1, 2, 2, 4, 4)]
        else:
            ch_mult = [ch * n for n in (1, 2, 2, 2)]

        # DOWN
        self.downblocks = nn.ModuleList()
        self.downblocks.append(nn.Conv2d(n_channels*(num_frames + num_frames_cond), ch, kernel_size=3, padding=1, stride=1))
        prev_ch = ch_mult[0]
        ch_size = [ch]
        for i, ich in enumerate(ch_mult):
            for firstarg in [prev_ch, ich]:
                self.downblocks.append(ResnetBlock_(firstarg, ich))
                ch_size += [ich]
                if i == 1:
                    self.downblocks.append(AttnBlock(ich))

            if i != len(ch_mult) - 1:
                self.downblocks.append(nn.Conv2d(ich, ich, kernel_size=3, stride=2, padding=1))
                ch_size += [ich]
            prev_ch = ich
        init_weights(self.downblocks, module_is_list=True)

        # MIDDLE
        self.middleblocks = nn.ModuleList()
        self.middleblocks.append(ResnetBlock_(ch_mult[-1], ch_mult[-1]))
        self.middleblocks.append(AttnBlock(ch_mult[-1]))
        self.middleblocks.append(ResnetBlock_(ch_mult[-1], ch_mult[-1]))

        # UP
        self.upblocks = nn.ModuleList()
        prev_ich = ch_mult[-1]
        for i, ich in reversed(list(enumerate(ch_mult))):
            for _ in range(3):
                self.upblocks.append(ResnetBlock_(prev_ich + ch_size.pop(), ich))
                if i == 1:
                    self.upblocks.append(AttnBlock(ich))
                prev_ich = ich
            if i != 0:
                self.upblocks.append(Upsample(ich))

        self.normalize = Normalize(ch)
        self.nonlinearity = Swish()
        self.out = nn.Conv2d(ch, n_channels*(num_frames + num_frames_cond) if getattr(config.model, 'output_all_frames', False) else n_channels*num_frames, kernel_size=3, stride=1, padding=1)
        init_weights(self.out, scale=0)

        self.temb_dense = nn.Sequential(
            nn.Linear(ch, ch * 4),
            self.nonlinearity,
            nn.Linear(ch * 4, ch * 4),
            self.nonlinearity
        )
        init_weights(self.temb_dense, module_is_list=True)

    # noinspection PyArgumentList
    def forward(self, x, y=None, cond=None):

        if y is not None and self.time_conditional:
            temb = get_timestep_embedding(y, self.ch)
            temb = self.temb_dense(temb)
        else:
            temb = None

        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        if not self.logit_transform and not self.rescaled:
            x = 2 * x - 1.

        hs = []
        for module in self.downblocks:
            if isinstance(module, ResnetBlock):
                x = module(x, temb)
            else:
                x = module(x)

            if isinstance(module, AttnBlock):
                hs.pop()
            hs += [x]

        for module in self.middleblocks:
            if isinstance(module, ResnetBlock):
                x = module(x, temb)
            else:
                x = module(x)

        for module in self.upblocks:
            if isinstance(module, ResnetBlock):
                x = module(torch.cat((x, hs.pop()), dim=1), temb)
            else:
                x = module(x)
        x = self.nonlinearity(self.normalize(x))
        output = self.out(x)

        if getattr(self.config.model, 'output_all_frames', False) and cond is not None:
            _, output = torch.split(output, [self.num_frames_cond*self.config.data.channels,self.num_frames*self.config.data.channels], dim=1)

        return output


class UNet_SMLD(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.version = getattr(config.model, 'version', 'SMLD').upper()
        assert self.version == "SMLD", f"models/unet : version is not SMLD! Given: {self.version}"

        self.config = config
        self.unet = UNet(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.noise_in_cond = getattr(config.model, 'noise_in_cond', False)

    def forward(self, x, y, cond=None, labels=None):

        if self.noise_in_cond and cond is not None: # We add noise to cond
            sigmas = self.sigmas
            # if labels is None:
            #     labels = torch.randint(0, len(sigmas), (cond.shape[0],), device=cond.device)
            labels = y
            used_sigmas = sigmas[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:])))
            z = torch.randn_like(cond)
            cond = cond + used_sigmas * z

        return self.unet(x, y, cond)


class UNet_DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.version = getattr(config.model, 'version', 'DDPM').upper()
        assert self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM", f"models/unet : version is not DDPM or DDIM! Given: {self.version}"

        self.config = config
        self.unet = UNet(config)

        self.schedule = getattr(config.model, 'sigma_dist', 'linear')
        if self.schedule == 'linear':
            self.register_buffer('betas', get_sigmas(config))   # large to small, doesn't match paper, match code instead
            self.register_buffer('alphas', torch.cumprod(1 - self.betas.flip(0), 0).flip(0))    # flip for small-to-large, then flip back
            self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
        elif self.schedule == 'cosine':
            self.register_buffer('alphas', get_sigmas(config))  # large to small, doesn't match paper, match code instead
            self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
            self.register_buffer('betas', (1 - self.alphas/self.alphas_prev).clip_(0, 0.999))
        self.gamma = getattr(config.model, 'gamma', False)
        if self.gamma:
            self.theta_0 = 0.001
            self.register_buffer('k', self.betas/(self.alphas*(self.theta_0 ** 2)))  # large to small, doesn't match paper, match code instead
            self.register_buffer('k_cum', torch.cumsum(self.k.flip(0), 0).flip(0))  # flip for small-to-large, then flip back
            self.register_buffer('theta_t', torch.sqrt(self.alphas)*self.theta_0)

        self.noise_in_cond = getattr(config.model, 'noise_in_cond', False)

    def forward(self, x, y, cond=None, labels=None, cond_mask=None):
        if self.noise_in_cond and cond is not None: # We add noise to cond
            alphas = self.alphas
            # if labels is None:
            #     labels = torch.randint(0, len(alphas), (cond.shape[0],), device=cond.device)
            labels = y
            used_alphas = alphas[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:])))
            if self.gamma:
                used_k = self.k_cum[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
                used_theta = self.theta_t[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
                z = Gamma(used_k, 1 / used_theta).sample()
                z = (z - used_k*used_theta)/(1 - used_alphas).sqrt()
            else:
                z = torch.randn_like(cond)
            cond = used_alphas.sqrt() * cond + (1 - used_alphas).sqrt() * z

        return self.unet(x, y, cond)
