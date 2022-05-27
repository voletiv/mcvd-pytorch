## 3d layers equivalents for conv and attention

from . import layers, layerspp
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

default_init = layers.default_init
contract_inner = layers.contract_inner

class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 3, 1, 2)


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0., n_heads=1, n_head_channels=-1):
    super().__init__()
    num_groups = min(channels // 4, 32)
    while(channels % num_groups != 0): # must find another value
      num_groups -= 1
    self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels,
                                  eps=1e-6)
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

class NIN1d(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 2, 1)

class AttnBlockpp1d(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM. in 1D"""

  def __init__(self, channels, skip_rescale=False, init_scale=0., n_heads=1, n_head_channels=-1):
    super().__init__()
    num_groups = min(channels // 4, 32)
    while (channels % num_groups != 0):
      num_groups -= 1
    self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN1d(channels, channels)
    self.NIN_1 = NIN1d(channels, channels)
    self.NIN_2 = NIN1d(channels, channels)
    self.NIN_3 = NIN1d(channels, channels, init_scale=init_scale)
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
    B, C, T = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    C = C // self.n_heads
    w = torch.einsum('bct,bci->bti', q.reshape(B * self.n_heads, C, T), k.reshape(B * self.n_heads, C, T)) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B * self.n_heads, T, T))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B * self.n_heads, T, T))
    h = torch.einsum('bti,bci->bct', w, v.reshape(B * self.n_heads, C, T))
    h = h.reshape(B, C * self.n_heads, T)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class NIN3d(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 4, 1) # BxCxNxHxW to BxNxHxWxC
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 4, 1, 2, 3)


class AttnBlockpp3d_old(nn.Module): # over time, height, width; crazy memory demands like 9GB for one att block!!! Not worth it
  """Channel-wise 3d self-attention block."""

  def __init__(self, channels, skip_rescale=False, init_scale=0., n_heads=1, n_head_channels=-1, n_frames=1):
    super().__init__()
    self.N = n_frames
    self.channels = self.Cin = channels // n_frames
    num_groups = min(self.channels // 4, 32)
    while (self.channels % num_groups != 0):
      num_groups -= 1
    self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=self.channels,
                                  eps=1e-6)
    self.NIN_0 = NIN3d(self.channels, self.channels)
    self.NIN_1 = NIN3d(self.channels, self.channels)
    self.NIN_2 = NIN3d(self.channels, self.channels)
    self.NIN_3 = NIN3d(self.channels, self.channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale
    if n_head_channels == -1:
      self.n_heads = n_heads
    else:
      if self.channels < n_head_channels:
        self.n_heads = 1
      else:
        assert self.channels % n_head_channels == 0
        self.n_heads = self.channels // n_head_channels

  def forward(self, x):
    # to 3d shape
    B, CN, H, W = x.shape
    C = self.Cin
    N = self.N
    x = x.reshape(B, C, N, H, W)

    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    C = C // self.n_heads

    w = torch.einsum('bcnhw,bcnij->bnhwij', q.reshape(B * self.n_heads, C, N, H, W), k.reshape(B * self.n_heads, C, N, H, W)) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B * self.n_heads, N, H, W, N * H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B * self.n_heads, N, H, W, N, H, W))
    h = torch.einsum('bnhwijk,bcijk->bcnhw', w, v.reshape(B * self.n_heads, C, N, H, W))
    h = h.reshape(B, C * self.n_heads, N, H, W)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      x = x + h
    else:
      x = (x + h) / np.sqrt(2.)
    return x.reshape(B, C*N, H, W)

class AttnBlockpp3d(nn.Module):
  """Channel-wise 3d self-attention block."""
  # Because doing attn over space-time is very memory demanding, we do space, then time
  # 1) space-only attn block
  # 2) time-only attn block

  def __init__(self, channels, skip_rescale=False, init_scale=0., n_heads=1, n_head_channels=-1, n_frames=1, act=None):
    super().__init__()
    self.N = n_frames
    self.channels = self.Cin = channels // n_frames
    self.space_att = AttnBlockpp(channels=self.channels, skip_rescale=skip_rescale, init_scale=init_scale, n_heads=n_heads, n_head_channels=n_head_channels)
    self.time_att = AttnBlockpp1d(channels=self.channels, skip_rescale=skip_rescale, init_scale=init_scale, n_heads=n_heads, n_head_channels=n_head_channels)
    self.act = act

  def forward(self, x):
    B, CN, H, W = x.shape
    C = self.Cin
    N = self.N
    
    # Space attention
    x = x.reshape(B, C, N, H, W).permute(0, 2, 1, 3, 4).reshape(B*N, C, H, W)
    x = self.space_att(x)
    x = x.reshape(B, N, C, H, W).permute(0, 2, 1, 3, 4) # B, C, N, H, W

    if self.act is not None:
      x = self.act(x)

    # Time attention
    x = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, C, N)
    x = self.time_att(x)
    x = x.reshape(B, H, W, C, N).permute(0, 3, 4, 1, 2).reshape(B, C*N, H, W)

    return x

class MyConv3d(nn.Module):
  """3d convolution."""

  def __init__(self, in_planes, out_planes, kernel_size, stride=1, bias=True, init_scale=1., padding=0, dilation=1, n_frames=1):
    super().__init__()
    self.N = n_frames
    self.Cin = in_planes // n_frames
    self.Cout = out_planes // n_frames
    self.conv = nn.Conv3d(self.Cin, self.Cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
    self.conv.weight.data = default_init(init_scale)(self.conv.weight.data.shape)
    nn.init.zeros_(self.conv.bias)

  def forward(self, x):
    # to 3d shape
    B, CN, H, W = x.shape
    x = x.reshape(B, self.Cin, self.N, H, W)
    x = self.conv(x)
    x = x.reshape(B, self.Cout*self.N, H, W)
    return x

def ddpm_conv1x1_3d(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0, n_frames=1):
  """1x1 convolution with DDPM initialization."""
  conv = MyConv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias, n_frames=n_frames)
  return conv

def ddpm_conv3x3_3d(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1, n_frames=1):
  """3x3 convolution with DDPM initialization."""
  conv = MyConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias, n_frames=n_frames)
  return conv


class PseudoConv3d(nn.Module):
  """Pseudo3d convolution."""
  # Because doing conv over space-time is very memory demanding, we do space, then time
  # 1) space-only conv2d
  # activation function
  # 2) time-only conv1d

  def __init__(self, in_planes, out_planes, kernel_size, stride=1, bias=True, init_scale=1., padding=0, dilation=1, n_frames=1, act=None):
    super().__init__()
    self.N = n_frames
    self.Cin = in_planes // n_frames
    self.Cout = out_planes // n_frames

    self.space_conv = nn.Conv2d(self.Cin, self.Cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
    self.space_conv.weight.data = default_init(init_scale)(self.space_conv.weight.data.shape)
    nn.init.zeros_(self.space_conv.bias)

    self.time_conv = nn.Conv1d(self.Cout, self.Cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
    self.time_conv.weight.data = default_init(init_scale)(self.time_conv.weight.data.shape)
    nn.init.zeros_(self.time_conv.bias)

    self.act=act

  def forward(self, x):
    B, CN, H, W = x.shape
    C = self.Cin
    N = self.N

    # Space conv2d B*N, C, H, W
    x = x.reshape(B, C, N, H, W).permute(0, 2, 1, 3, 4).reshape(B*N, C, H, W)
    x = self.space_conv(x)
    C = self.Cout
    x = x.reshape(B, N, C, H, W).permute(0, 2, 1, 3, 4) # B, C, N, H, W

    if self.act is not None:
      x = self.act(x)

    # Time conv1d B*H*W, C, N
    x = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, C, N)
    x = self.time_conv(x)
    x = x.reshape(B, H, W, C, N).permute(0, 3, 4, 1, 2).reshape(B, C*N, H, W)

    return x

def ddpm_conv1x1_pseudo3d(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0, n_frames=1, act=None):
  """1x1 Pseudo convolution with DDPM initialization."""
  conv = PseudoConv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias, n_frames=n_frames, act=act)
  return conv

def ddpm_conv3x3_pseudo3d(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1, n_frames=1, act=None):
  """3x3 Pseudo convolution with DDPM initialization."""
  conv = PseudoConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias, n_frames=n_frames, act=act)
  return conv
