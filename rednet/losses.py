"""Training losses for ReDNet.

The released model is trained with a combination of a perceptual LPIPS loss
(VGG backbone, taming-transformers implementation) and the Focal Frequency
Loss (Jiang et al., ICCV 2021).
"""
import os
from collections import namedtuple

import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision import models as tv_models


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    https://arxiv.org/pdf/2012.12821.pdf
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False,
                 log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])
        y = torch.stack(patch_list, 1)
        freq = torch.fft.fft2(y, norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        return self.loss_weight * self.loss_formulation(pred_freq, target_freq, matrix=matrix)


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class VGG16Features(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG16Features, self).__init__()
        try:
            vgg_pretrained_features = tv_models.vgg16(
                weights=tv_models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features
        except AttributeError:  # older torchvision
            vgg_pretrained_features = tv_models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LPIPS(nn.Module):
    """Learned perceptual metric with VGG backbone and linear calibration
    weights (taming-transformers style)."""

    CKPT_URLS = [
        'https://huggingface.co/Zongliang-Wu/ReDNet/resolve/main/pretrained/lpips_vgg.pth',
        'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1',
    ]

    def __init__(self, use_dropout=True, loss_weight=1.0, reduction='mean', ckpt_path=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16Features(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained(ckpt_path)
        for param in self.parameters():
            param.requires_grad = False

    def _resolve_ckpt(self, ckpt_path=None):
        candidates = []
        if ckpt_path is not None:
            candidates.append(ckpt_path)
        if os.environ.get('REDNET_LPIPS_CKPT'):
            candidates.append(os.environ['REDNET_LPIPS_CKPT'])
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'pretrained', 'lpips_vgg.pth'))
        for c in candidates:
            if os.path.isfile(c):
                return c
        # try to download
        try:
            import urllib.request
            target = candidates[-1]
            os.makedirs(os.path.dirname(target), exist_ok=True)
            for url in self.CKPT_URLS:
                try:
                    print(f'Downloading LPIPS linear weights from {url} ...')
                    urllib.request.urlretrieve(url, target)
                    return target
                except Exception as e:
                    print(f'  failed: {e}')
        except Exception:
            pass
        raise FileNotFoundError(
            'Cannot find the LPIPS linear-calibration weights (lpips_vgg.pth / vgg.pth). '
            'Place it at ./pretrained/lpips_vgg.pth or set REDNET_LPIPS_CKPT.')

    def load_from_pretrained(self, ckpt_path=None):
        ckpt = self._resolve_ckpt(ckpt_path)
        state = torch.load(ckpt, map_location='cpu')
        state = state.get('params', state) if isinstance(state, dict) else state
        state = state.get('lpips', state) if isinstance(state, dict) and 'lpips' in state else state
        if isinstance(state, dict):
            renamed = {}
            for k, v in state.items():
                k = k.replace('lin0.model', 'lin0.model').replace('model.1.', 'model.')
                renamed[k] = v
            state = renamed
        self.load_state_dict(state, strict=False)
        print(f'Loaded pretrained LPIPS loss from {ckpt}')

    def forward(self, input, target, **kwargs):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val.mean()


class DirtLoss(nn.Module):
    """Total reconstruction loss used to train the released ReDNet model:
    ``L = LPIPS(pred, gt) + FocalFrequencyLoss(pred, gt)``."""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(DirtLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_rec_perceptual = LPIPS()
        self.loss_freq = FocalFrequencyLoss()

    def forward(self, pred, target, pred_mask=None, target_mask=None, **kwargs):
        loss = self.loss_rec_perceptual(pred, target) + self.loss_freq(pred, target)
        return self.loss_weight * loss
