"""ReDNet: Removing Dirt Network.

Overall structure: a physics-inspired Removing Dirt Pipeline (ReDP) followed
by a Restormer-based detail refinement network.

ReDP implements the inverse of the dirty window imaging model
    Y = X * F_out(P, G) + F_in(P, G)   =>   X = (Y - F_in) / F_out
with blocks:
  - SEB  (Shape Estimation Block, OGT with 1 head)
  - DSVB (Density and Scale Variation Block, OGT with 1 head)
  - BCB  (Blur Correction Block, OGT with 4 heads + kernel prediction)
  - IFB1 / IFB2 (Illumination Function Blocks, OGT with 4 heads)
  - channel SE modules between blocks.
"""
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ogt_arch import TransformerBlock as OGTBlock
from .restormer_arch import Restormer
from .se_module import ChannelSELayer as ChannelSE


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super(ConvBlock, self).__init__()
        if out_ch is None:
            out_ch = in_ch
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.conv3(x))


class TypeEst(nn.Module):
    """Shape Estimation Block (SEB)."""

    def __init__(self, in_ch, out_ch, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 uPlane=3, scale=4):
        super(TypeEst, self).__init__()
        self.transBlock = OGTBlock(in_ch, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                                   uPlane=uPlane, scale=scale)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        out = self.transBlock(x)
        out = self.conv(out)
        return out


class DenseEst(nn.Module):
    """Density and Scale Variation Block (DSVB)."""

    def __init__(self, in_ch, out_ch, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 uPlane=3, scale=4):
        super(DenseEst, self).__init__()
        self.transBlock = OGTBlock(in_ch, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                                   uPlane=uPlane, scale=scale)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        out = self.transBlock(x)
        out = self.conv(out)
        return out


class KernalCorr(nn.Module):
    """Applies the small kernels predicted by the BCB (blur / de-blur process)."""

    def __init__(self, in_ch, out_ch=None, n_kernals=4):
        super(KernalCorr, self).__init__()
        self.n_kernals = n_kernals
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, PSF_map):
        bs, ch, H, W = PSF_map.shape
        PSF_map = rearrange(PSF_map, 'b (c nin) h w -> b c nin h w', nin=int(sqrt(ch / self.n_kernals)))

        kernals = PSF_map.chunk(self.n_kernals, dim=1)

        for b in range(bs):
            for i in range(self.n_kernals):
                if i == 0:
                    out_i = F.conv2d(x[b:b + 1, :, :, :], kernals[i][b, :, :, :, :], padding=1)
                elif i < self.n_kernals // 2:
                    out_i = F.conv2d(out_i, kernals[i][b, :, :, :, :], padding=1)
                else:
                    out_i = F.conv_transpose2d(out_i, kernals[i][b, :, :, :, :], padding=1)
            if b == 0:
                out = out_i
            else:
                out = torch.cat([out, out_i], dim=0)

        blur_map = self.conv(out)
        return blur_map


class BlurEst(nn.Module):
    """Blur Correction Block (BCB): an OGT predicts the PSF kernels, then
    small kernel convolutions / transposed convolutions emulate the blur and
    de-blur functions."""

    def __init__(self, in_ch, out_ch, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 uPlane=3, scale=4, n_kernal=4):
        super(BlurEst, self).__init__()
        self.transBlock = OGTBlock(in_ch, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                                   uPlane=uPlane, scale=scale, for_PSF=True, n_kernal=n_kernal, k_size=3)
        self.kernalCorrBlock = KernalCorr(in_ch, out_ch, n_kernals=n_kernal)

    def forward(self, x):
        PSF_map = self.transBlock(x)
        out = self.kernalCorrBlock(x, PSF_map)
        return out


class IlluminationEst(nn.Module):
    """Illumination Function Block (IFB)."""

    def __init__(self, in_ch, out_ch, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 uPlane=3, scale=4):
        super(IlluminationEst, self).__init__()
        self.transBlock = OGTBlock(in_ch, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                                   uPlane=uPlane, scale=scale)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        out = self.transBlock(x)
        out = self.conv(out)
        return out


class ChAttenuation(nn.Module):
    """Channel attention (SE) followed by 1x1 conv, used between ReDP blocks."""

    def __init__(self, in_ch, out_ch):
        super(ChAttenuation, self).__init__()
        self.se1 = ChannelSE(in_ch, 2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        out = self.se1(x)
        out = self.conv1(out)
        return out


class ReDNet(nn.Module):
    """ReDNet = ReDP (physics pipeline) + Restormer (RefineNet).

    Args:
        in_ch (int): input channels.
        out_ch (int): output channels.
        dim (int): base width. The ReDP latent channels equal ``dim // 3``
            (16 for the default dim=48, following the paper).
        num_blocks (list): Restormer encoder block counts.
        num_refinement_blocks (int): Restormer refinement block count.
        heads (list): attention head counts. heads[0] is used by SEB/DSVB,
            heads[2] by BCB/IFBs (1, 1, 4, 4 in the paper).
    """

    def __init__(self,
                 in_ch=3,
                 out_ch=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8, 16],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8, 16],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(ReDNet, self).__init__()

        n_feat = dim // 3

        # SEB + DSVB
        self.type_est = TypeEst(in_ch, n_feat, heads[0], ffn_expansion_factor, bias,
                                LayerNorm_type, uPlane=1, scale=3)
        self.dense_est = DenseEst(in_ch, n_feat, heads[0], ffn_expansion_factor, bias,
                                  LayerNorm_type, uPlane=1, scale=3)
        self.se1 = ChAttenuation(n_feat + n_feat, n_feat)

        # BCB
        self.blur_est = BlurEst(n_feat, n_feat, heads[2], ffn_expansion_factor, bias,
                                LayerNorm_type, uPlane=heads[2], scale=4, n_kernal=4)
        self.se_blur = ChAttenuation(n_feat, 3)

        # IFBs: F_in and F_out estimation
        self.se1_5 = ChAttenuation(6, n_feat)
        self.illum_est = IlluminationEst(n_feat, n_feat, heads[2], ffn_expansion_factor, bias,
                                         LayerNorm_type, uPlane=heads[2], scale=4)
        self.se2 = ChAttenuation(n_feat, 3)

        self.se1_52 = ChAttenuation(6, n_feat)
        self.illum_est22 = IlluminationEst(n_feat, n_feat, heads[2], ffn_expansion_factor, bias,
                                           LayerNorm_type, uPlane=heads[2], scale=4)
        self.se22 = ChAttenuation(n_feat, 3)

        # RefineNet (Restormer)
        self.recoNet = Restormer(3, 3)

    def forward(self, x):
        """Returns ``(restored image, estimated dirt pattern)``."""
        type_fea = self.type_est(x)
        dense_feat = self.dense_est(x)

        cat1 = self.se1(torch.cat([type_fea, dense_feat], dim=1))

        blur_feat = self.blur_est(cat1)
        dirt_pattern = self.se_blur(blur_feat)

        # F_out (denominator) and F_in (additive term)
        illu_in = self.se1_5(torch.cat([x, dirt_pattern], dim=1))
        illum = self.illum_est(illu_in)
        ones_f1p = self.se2(illum) + 1e-7

        illu_in2 = self.se1_52(torch.cat([x, dirt_pattern], dim=1))
        illum2 = self.illum_est22(illu_in2)
        f2p = self.se22(illum2)

        # inverse of the imaging model: X = (Y - F_in) / F_out
        out1 = (x - f2p) / ones_f1p
        out = self.recoNet(out1) + out1

        return out, dirt_pattern
