"""Optics-guided Transformer (OGT) and Deformable Multi-head Attention (DefMHA)."""
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dcn import DeformableConv2d as DefConv


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Attention(nn.Module):
    """Deformable Convolution Multi-head Attention (DefMHA).

    Q/K/V are produced by 1x1 convolutions followed by a 3x3 depthwise
    deformable convolution so that the sampling offsets adapt to dirt shapes.
    """

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, padding=0, bias=bias)
        self.qkv_defconv = DefConv(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                   bias=True, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_defconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1,
                                padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class TransformerBlock(nn.Module):
    """Optics-guided Transformer (OGT) block.

    Features are split along channels into `scale` groups of `uPlane`
    channels (Res2Net-style). Group i is fed to DefMHA attention i and the
    output is accumulated onto group i+1, enabling multi-scale perception of
    dirt shape / density / scale variation.

    When ``for_PSF`` is True (used by the Blur Correction Block), the block
    additionally predicts a set of small blur/deblur kernels from the
    features (a `n_kernal` group of `dim x dim` maps pooled to k_size x k_size).
    """

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 uPlane=3, scale=4, for_PSF=False, n_kernal=4, k_size=3):
        super(TransformerBlock, self).__init__()
        uPlane = num_heads
        scale = int(dim / uPlane)

        self.scale = scale
        self.uPlane = uPlane
        self.for_PSF = for_PSF

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        attn = []
        for _ in range(self.scale - 1):
            attn.append(Attention(self.uPlane, num_heads, bias))
        self.attn = nn.ModuleList(attn)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        if self.for_PSF:
            self.conv1x1 = nn.Conv2d(dim, dim * n_kernal, kernel_size=1, bias=bias)
            self.repeat_conv = nn.Sequential(
                nn.Conv2d(dim * n_kernal, n_kernal * dim * dim, kernel_size=1, bias=False),
                nn.AdaptiveAvgPool2d((k_size, k_size)))

    def forward(self, x):
        x_nm1 = self.norm1(x)
        spx = torch.split(x_nm1, self.uPlane, 1)
        if self.scale > 1:
            for i in range(self.scale - 1):
                if i == 0:
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.attn[i](sp)
                if i == 0:
                    out = sp
                else:
                    out = torch.cat((out, sp), 1)
            out = torch.cat((out, spx[self.scale - 1]), 1)

        x = x + out
        x = x + self.ffn(self.norm2(x))

        if self.for_PSF:
            bs, ch, W, H = x.shape
            out = self.conv1x1(x)
            if W > H:
                out = out[:, :, 0:H, :]
            elif W < H:
                out = out[:, :, :, 0:W]
            out = self.repeat_conv(out)
            x = out

        return x
