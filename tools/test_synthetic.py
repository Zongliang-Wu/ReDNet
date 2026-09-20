"""Evaluate ReDNet on the synthetic test set (Tab. III protocol).

Metrics: PSNR / SSIM / LPIPS. By default SSIM uses data_range=2.0, which
reproduces the metric convention of the paper (see rednet/utils.py); pass
--standard_ssim for the modern convention (data_range=1.0).

Usage:
    python tools/test_synthetic.py \
        --weights pretrained/rednet_dirt.pth \
        --dirty_dir datasets/syn_test/dirty \
        --gt_dir datasets/syn_test/gt \
        --save_dir results/syn --save_images
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rednet.archs.rednet_arch import ReDNet
from rednet.utils import load_img, save_img, pad_to_multiple, psnr, ssim, compute_lpips


def main():
    parser = argparse.ArgumentParser(description='ReDNet synthetic test evaluation')
    parser.add_argument('--weights', default='pretrained/rednet_dirt.pth', type=str)
    parser.add_argument('--dirty_dir', required=True, type=str, help='folder of dirty test images')
    parser.add_argument('--gt_dir', required=True, type=str, help='folder of clean ground truths')
    parser.add_argument('--save_dir', default='results/syn', type=str)
    parser.add_argument('--no_lpips', action='store_true', help='skip LPIPS (avoids the lpips package)')
    parser.add_argument('--save_images', action='store_true', help='save restored images (required for the LPIPS protocol)')
    parser.add_argument('--standard_ssim', action='store_true',
                        help='use the modern SSIM convention (data_range=1) instead of the paper convention (data_range=2)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ReDNet(in_ch=3, out_ch=3, num_blocks=[4, 6, 6, 8, 16], num_refinement_blocks=4,
                   ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
    ckpt = torch.load(args.weights, map_location='cpu')
    state = ckpt.get('params', ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f'==> Loaded weights: {args.weights}')

    from natsort import natsorted
    from glob import glob
    files = natsorted(glob(os.path.join(args.dirty_dir, '*.png')) +
                      glob(os.path.join(args.dirty_dir, '*.jpg')))
    gt_files = natsorted(glob(os.path.join(args.gt_dir, '*.png')) +
                         glob(os.path.join(args.gt_dir, '*.jpg')))
    assert len(files) == len(gt_files), f'{len(files)} dirty vs {len(gt_files)} gt images'
    print(f'==> Evaluating {len(files)} image pairs')

    if args.save_images:
        os.makedirs(args.save_dir, exist_ok=True)

    ssim_dr = 1.0 if args.standard_ssim else 2.0
    psnr_sum = ssim_sum = lpips_sum = 0.0
    n_lpips = 0
    with torch.no_grad():
        for f, gf in zip(files, gt_files):
            img = load_img(f).astype(np.float32) / 255.
            inp = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()[None].to(device)
            inp = pad_to_multiple(inp, factor=8)
            h, w = img.shape[:2]
            out, _ = model(inp)
            out = torch.clamp(out[:, :, :h, :w], 0, 1)[0].permute(1, 2, 0).cpu().numpy()

            gt = load_img(gf).astype(np.float32) / 255.
            # paper convention: normalize each image by its own maximum
            o = out / out.max()
            g = gt / gt.max()
            psnr_sum += psnr(o, g)
            ssim_sum += ssim(o, g, data_range=ssim_dr)
            if not args.no_lpips:
                if args.save_images:
                    spath = os.path.join(args.save_dir, os.path.splitext(os.path.basename(f))[0] + '.png')
                    save_img(spath, out)
                    lpips_sum += compute_lpips(spath, gf)
                else:
                    lpips_sum += compute_lpips((out * 255).round().astype(np.uint8),
                                               (gt * 255).round().astype(np.uint8))
                n_lpips += 1
            elif args.save_images:
                save_img(os.path.join(args.save_dir, os.path.splitext(os.path.basename(f))[0] + '.png'), out)

    n = len(files)
    print('================ Results ================')
    print(f'PSNR: {psnr_sum / n:.4f} dB')
    print(f'SSIM: {ssim_sum / n:.4f}  (data_range={ssim_dr:g})')
    if n_lpips:
        print(f'LPIPS: {lpips_sum / n_lpips:.4f}')


if __name__ == '__main__':
    main()
