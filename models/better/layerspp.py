# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from . import layers
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import functools

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


# https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py
# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    # def __init__(self, config_text, norm_nc, label_nc):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        param_free_norm_type = 'group'
        ks = 3

        if param_free_norm_type == 'group':
            num_groups = min(norm_nc // 4, 32)
            while(norm_nc % num_groups != 0): # must find another value
                num_groups -= 1
            self.param_free_norm = nn.GroupNorm(num_groups=num_groups, num_channels=norm_nc, affine=False, eps=1e-6)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class MySPADE(nn.Module):
    # def __init__(self, config_text, norm_nc, label_nc):
    def __init__(self, norm_nc, label_nc, param_free_norm_type='group', act=nn.ReLU(), conv=conv3x3,
                 spade_dim=128, is3d=False, num_frames=0, num_frames_cond=0, conv1x1=None):
        super().__init__()

        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))

        self.norm_nc = norm_nc
        self.label_nc = label_nc
        self.param_free_norm_type = param_free_norm_type
        self.act = act
        self.conv = conv
        self.spade_dim = spade_dim
        self.is3d = is3d
        self.num_frames = num_frames
        self.num_frames_cond = num_frames_cond
        self.conv1x1 = conv1x1

        self.N = N = 1 if not is3d else num_frames   # useful for 3D conv

        # # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128 if not is3d else 64

        if param_free_norm_type == 'group':
            num_groups = min(norm_nc // 4, 32)
            while(norm_nc % num_groups != 0): # must find another value
                num_groups -= 1
            self.param_free_norm = nn.GroupNorm(num_groups=num_groups, num_channels=norm_nc, affine=False, eps=1e-6)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        in_ch = label_nc
        if self.is3d:
          channels = int(label_nc/num_frames_cond)
          in_ch = int(label_nc/num_frames_cond*num_frames)
          self.converter = conv1x1(label_nc, int(channels * num_frames))

        self.mlp_shared = nn.Sequential(conv(in_ch, spade_dim//N*N), act)
        self.mlp_gamma = conv(spade_dim//N*N, norm_nc*N)
        self.mlp_beta = conv(spade_dim//N*N, norm_nc*N)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
        normalized = self.param_free_norm(x).reshape(B, -1, H, W)

        if self.is3d:
          # Use 1x1 Conv3D to convert segmap from num_frames_cond (Nc) to num_frames (N) :
          # B,CNc,H,W -> B,C,Nc,H,W -> B,Nc,C,H,W ---1x1Conv3D--> B,N,C,H,W -> B,C,N,H,W -> B,CN,H,W
          B, CN, H, W = segmap.shape
          C = CN // self.num_frames_cond
          segmap = self.converter(segmap.reshape(B, C, -1, H, W).permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)).reshape(B, -1, C, H, W).permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[-2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


# Added multi-head attention similar to https://github.com/openai/guided-diffusion/blob/912d5776a64a33e3baf3cff7eb1bcba9d9b9354c/guided_diffusion/unet.py#L361
class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0., n_heads=1, n_head_channels=-1):
    super().__init__()
    num_groups = min(channels // 4, 32)
    while (channels % num_groups != 0): # must find another value
        num_groups -= 1
    self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale
    if n_head_channels == -1:
        self.n_heads = n_heads
    else:
        if channels < n_head_channels:
          self.n_heads = 1
        else:
          assert channels % n_head_channels == 0
          self.n_heads = channels // n_head_channels

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    C = C // self.n_heads

    w = torch.einsum('bchw,bcij->bhwij', q.reshape(B * self.n_heads, C, H, W), k.reshape(B * self.n_heads, C, H, W)) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B * self.n_heads, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B * self.n_heads, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v.reshape(B * self.n_heads, C, H, W))
    h = h.reshape(B, C * self.n_heads, H, W)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, up=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      if not self.with_conv:
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = self.Conv2d_0(x)

    return h


class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, down=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      if self.with_conv:
        x = F.pad(x, (0, 1, 0, 1))
        x = self.Conv_0(x)
      else:
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      if not self.with_conv:
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = self.Conv2d_0(x)

    return x


class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0., is3d=False, n_frames=1, 
               pseudo3d=False, act3d=False):
    super().__init__()

    if pseudo3d or is3d:
      from . import layers3d
      conv3x3_3d = layers3d.ddpm_conv3x3_3d
      conv1x1_3d = layers3d.ddpm_conv1x1_3d
      conv3x3_pseudo3d = layers3d.ddpm_conv3x3_pseudo3d
      conv1x1_pseudo3d = layers3d.ddpm_conv1x1_pseudo3d

    if pseudo3d:
      conv3x3_ = functools.partial(conv3x3_pseudo3d, n_frames=n_frames, act=act if act3d else None)
      conv1x1_ = functools.partial(conv1x1_pseudo3d, n_frames=n_frames, act=act if act3d else None)
    elif is3d:
      conv3x3_ = functools.partial(conv3x3_3d, n_frames=n_frames)
      conv1x1_ = functools.partial(conv1x1_3d, n_frames=n_frames)
    else:
      conv3x3_ = conv3x3
      conv1x1_ = conv1x1

    out_ch = out_ch if out_ch else in_ch
    num_groups = min(in_ch // 4, 32)
    while (in_ch % num_groups != 0):
      num_groups -= 1
    self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=in_ch, eps=1e-6)
    self.Conv_0 = conv3x3_(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    num_groups = min(out_ch // 4, 32)
    while (in_ch % num_groups != 0):
      num_groups -= 1
    self.GroupNorm_1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3_(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3_(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))
    h = self.Conv_0(h)
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockDDPMppSPADE(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, spade_dim=128,
               dropout=0.1, skip_rescale=False, init_scale=0., is3d=False, n_frames=1, num_frames_cond=0, cond_ch=0,
               pseudo3d=False, act3d=False):
    super().__init__()

    if pseudo3d or is3d:
      from . import layers3d
      conv3x3_3d = layers3d.ddpm_conv3x3_3d
      conv1x1_3d = layers3d.ddpm_conv1x1_3d
      conv3x3_pseudo3d = layers3d.ddpm_conv3x3_pseudo3d
      conv1x1_pseudo3d = layers3d.ddpm_conv1x1_pseudo3d

    if pseudo3d:
      conv3x3_ = functools.partial(conv3x3_pseudo3d, n_frames=n_frames, act=act if act3d else None)
      conv1x1_ = functools.partial(conv1x1_pseudo3d, n_frames=n_frames, act=act if act3d else None)
      conv1x1_cond = functools.partial(conv1x1_pseudo3d, n_frames=cond_ch//num_frames_cond, act=act if act3d else None)
    elif is3d:
      conv3x3_ = functools.partial(conv3x3_3d, n_frames=n_frames)
      conv1x1_ = functools.partial(conv1x1_3d, n_frames=n_frames)
      conv1x1_cond = functools.partial(conv1x1_3d, n_frames=cond_ch//num_frames_cond)
    else:
      conv3x3_ = conv3x3
      conv1x1_ = conv1x1
      conv1x1_cond = conv1x1

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = MySPADE(norm_nc=(in_ch//n_frames if is3d else in_ch), label_nc=cond_ch, param_free_norm_type='group', act=act, conv=conv3x3_,
                               spade_dim=spade_dim, is3d=is3d, num_frames=n_frames, num_frames_cond=num_frames_cond, conv1x1=conv1x1_cond)
    self.Conv_0 = conv3x3_(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.GroupNorm_1 = MySPADE(norm_nc=(out_ch//n_frames if is3d else out_ch), label_nc=cond_ch, param_free_norm_type='group', act=act, conv=conv3x3_,
                               spade_dim=spade_dim, is3d=is3d, num_frames=n_frames, num_frames_cond=num_frames_cond, conv1x1=conv1x1_cond)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3_(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3_(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None, cond=None):
    h = self.act(self.GroupNorm_0(x, cond))
    h = self.Conv_0(h)
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h, cond))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


def get_norm(norm, ch, affine=True):
  """Get activation functions from the opt file."""
  if norm == 'none':
    return nn.Identity()
  elif norm == 'batch':
    return nn.BatchNorm1d(ch, affine = affine)
  elif norm == 'evo':
    return EvoNorm2D(ch = ch, affine = affine, eps = 1e-5, groups = min(ch // 4, 32))
  elif norm == 'group':
    num_groups=min(ch // 4, 32)
    while(ch % num_groups != 0): # must find another value
      num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=ch, eps=1e-5, affine=affine)
  elif norm == 'layer':
    return nn.LayerNorm(normalized_shape=ch, eps=1e-5, elementwise_affine=affine)
  elif norm == 'instance':
    return nn.InstanceNorm2d(num_features=ch, eps=1e-5, affine=affine)
  else:
    raise NotImplementedError('norm choice does not exist')


class get_act_norm(nn.Module): # order is norm -> act
  def __init__(self, act, act_emb, norm, ch, emb_dim=None, spectral=False, is3d=False, n_frames=1,
               num_frames_cond=0, cond_ch=0, spade_dim=128, cond_conv=None, cond_conv1=None):
    super(get_act_norm, self).__init__()
    
    self.norm = norm
    self.act = act
    self.act_emb = act_emb
    self.is3d = is3d
    self.n_frames = n_frames
    self.cond_ch = cond_ch
    if emb_dim is not None:
      if self.is3d:
        out_dim = 2*(ch // self.n_frames)
      else:
        out_dim = 2*ch
      if spectral:
        self.Dense_0 = torch.nn.utils.spectral_norm(nn.Linear(emb_dim, out_dim))
      else:
        self.Dense_0 = nn.Linear(emb_dim, out_dim)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)
      affine = False # We remove scale/intercept after normalization since we will learn it with [temb, yemb]
    else:
      affine = True

    if norm == 'spade':
      self.Norm_0 = MySPADE(norm_nc=(ch // n_frames) if is3d else ch, label_nc=cond_ch, param_free_norm_type='group', act=act, conv=cond_conv,
                            spade_dim=spade_dim, is3d=is3d, num_frames=n_frames, num_frames_cond=num_frames_cond, conv1x1=cond_conv1)
    else:
      self.Norm_0 = get_norm(norm, (ch // n_frames) if is3d else ch, affine)

  def forward(self, x, emb=None, cond=None):
    if emb is not None:
      #emb = torch.cat([temb, yemb], dim=1) # Combine embeddings
      emb_out = self.Dense_0(self.act_emb(emb))[:, :, None, None] # Linear projection
      # ada-norm as in https://github.com/openai/guided-diffusion
      scale, shift = torch.chunk(emb_out, 2, dim=1)
      if self.is3d:
        B, CN, H, W = x.shape
        N = self.n_frames
        scale = scale.reshape(B, -1, 1, 1, 1)
        shift = shift.reshape(B, -1, 1, 1, 1)
        x = x.reshape(B, -1, N, H, W)
      if self.norm == 'spade':
        emb_norm = self.Norm_0(x, cond)
        emb_norm = emb_norm.reshape(B, -1, N, H, W) if self.is3d else emb_norm
      else:
        emb_norm = self.Norm_0(x)
      x = emb_norm * (1 + scale) + shift
      if self.is3d:
        x = x.reshape(B, -1, H, W)
    else:
      if self.is3d:
        B, CN, H, W = x.shape
        N = self.n_frames
        x = x.reshape(B, -1, N, H, W)
      if self.norm == 'spade':
        x = self.Norm_0(x, cond)
      else:
        x = self.Norm_0(x)
        x = x.reshape(B, CN, H, W) if self.is3d else x
    x = self.act(x)
    return(x)


# Group normalization added
class ResnetBlockBigGANppGN(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0., is3d=False, n_frames=1, pseudo3d=False, act3d=False):
    super().__init__()

    if pseudo3d or is3d:
      from . import layers3d
      conv3x3_3d = layers3d.ddpm_conv3x3_3d
      conv1x1_3d = layers3d.ddpm_conv1x1_3d
      conv3x3_pseudo3d = layers3d.ddpm_conv3x3_pseudo3d
      conv1x1_pseudo3d = layers3d.ddpm_conv1x1_pseudo3d

    if pseudo3d:
      conv3x3_ = functools.partial(conv3x3_pseudo3d, n_frames=n_frames, act=act if act3d else None)
      conv1x1_ = functools.partial(conv1x1_pseudo3d, n_frames=n_frames, act=act if act3d else None)
    elif is3d:
      conv3x3_ = functools.partial(conv3x3_3d, n_frames=n_frames)
      conv1x1_ = functools.partial(conv1x1_3d, n_frames=n_frames)
    else:
      conv3x3_ = conv3x3
      conv1x1_ = conv1x1

    out_ch = out_ch if out_ch else in_ch
    self.actnorm0 = get_act_norm(act, act, 'group', in_ch, emb_dim=temb_dim, spectral=False, is3d=is3d, n_frames=n_frames)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.Conv_0 = conv3x3_(in_ch, out_ch)

    self.actnorm1 = get_act_norm(act, act, 'group', out_ch, emb_dim=temb_dim, spectral=False, is3d=is3d, n_frames=n_frames)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3_(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1_(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.actnorm0(x, temb)

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    h = self.Conv_0(h)
    h = self.actnorm1(h, temb)
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


# SPADE normalization added
class ResnetBlockBigGANppSPADE(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False, spade_dim=128,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1), n_frames=1, num_frames_cond=0, cond_ch=0,
               skip_rescale=True, init_scale=0., is3d=False, pseudo3d=False, act3d=False):
    super().__init__()

    if pseudo3d or is3d:
      from . import layers3d
      conv3x3_3d = layers3d.ddpm_conv3x3_3d
      conv1x1_3d = layers3d.ddpm_conv1x1_3d
      conv3x3_pseudo3d = layers3d.ddpm_conv3x3_pseudo3d
      conv1x1_pseudo3d = layers3d.ddpm_conv1x1_pseudo3d

    if pseudo3d:
      conv3x3_ = functools.partial(conv3x3_pseudo3d, n_frames=n_frames, act=act if act3d else None)
      conv1x1_ = functools.partial(conv1x1_pseudo3d, n_frames=n_frames, act=act if act3d else None)
      conv1x1_cond = functools.partial(conv1x1_pseudo3d, n_frames=cond_ch//num_frames_cond, act=act if act3d else None)
    elif is3d:
      conv3x3_ = functools.partial(conv3x3_3d, n_frames=n_frames)
      conv1x1_ = functools.partial(conv1x1_3d, n_frames=n_frames)
      conv1x1_cond = functools.partial(conv1x1_3d, n_frames=cond_ch//num_frames_cond)
    else:
      conv3x3_ = conv3x3
      conv1x1_ = conv1x1
      conv1x1_cond = conv1x1

    out_ch = out_ch if out_ch else in_ch
    self.actnorm0 = get_act_norm(act, act, 'spade', in_ch, emb_dim=temb_dim, spectral=False, is3d=is3d, n_frames=n_frames,
                                 num_frames_cond=num_frames_cond, cond_ch=cond_ch, spade_dim=spade_dim, cond_conv=conv3x3_, cond_conv1=conv1x1_cond)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.Conv_0 = conv3x3_(in_ch, out_ch)

    self.actnorm1 = get_act_norm(act, act, 'spade', out_ch, emb_dim=temb_dim, spectral=False, is3d=is3d, n_frames=n_frames,
                                 num_frames_cond=num_frames_cond, cond_ch=cond_ch, spade_dim=spade_dim, cond_conv=conv3x3_, cond_conv1=conv1x1_cond)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3_(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1_(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None, cond=None):
    h = self.actnorm0(x, temb, cond)

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    h = self.Conv_0(h)
    h = self.actnorm1(h, temb, cond)
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    # Shortcut
    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
