"""Evaluate ReDNet on the real-world dirty window benchmark (Tab. IV protocol).

Two modes:
1. Full-reference (paired): the 3 released dirty/GT pairs. Images are
   restored at native resolution (1200x900); PSNR / SSIM / LPIPS are
   computed with the paper metric convention (SSIM data_range=2, images
   normalized by their own maximum -- see rednet/utils.py).
2. No-reference: a folder of real dirty images is restored and saved.

Usage:
    # full-reference benchmark (Tab. IV): use the real-benchmark checkpoint
    python tools/test_real.py --mode fr \
        --weights pretrained/rednet_dirt_real.pth \
        --dirty_dir datasets/real_test/paired/dirty \
        --gt_dir datasets/real_test/paired/gt --save_dir results/real_fr

    # 67 no-reference images (figure-quality checkpoint)
    python tools/test_real.py --mode nr \
        --weights pretrained/rednet_dirt.pth \
        --dirty_dir datasets/real_test/noref --save_dir results/real_nr
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rednet.archs.rednet_arch import ReDNet
from rednet.utils import load_img, save_img, psnr, ssim, compute_lpips, restore_image


def main():
    parser = argparse.ArgumentParser(description='ReDNet real-world test evaluation')
    parser.add_argument('--weights', default='pretrained/rednet_dirt.pth', type=str)
    parser.add_argument('--mode', choices=['fr', 'nr'], default='fr',
                        help='fr: full-reference paired data; nr: no-reference data')
    parser.add_argument('--dirty_dir', required=True, type=str)
    parser.add_argument('--gt_dir', default=None, type=str, help='required for --mode fr')
    parser.add_argument('--save_dir', default='results/real', type=str)
    parser.add_argument('--no_lpips', action='store_true')
    parser.add_argument('--standard_ssim', action='store_true',
                        help='use the modern SSIM convention (data_range=1) instead of the paper convention (data_range=2)')
    args = parser.parse_args()
    assert args.mode == 'nr' or args.gt_dir is not None, '--gt_dir is required for --mode fr'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ReDNet(in_ch=3, out_ch=3, num_blocks=[4, 6, 6, 8, 16], num_refinement_blocks=4,
                   ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt.get('params', ckpt))
    model.to(device).eval()
    print(f'==> Loaded weights: {args.weights}')

    from natsort import natsorted
    from glob import glob
    files = natsorted(glob(os.path.join(args.dirty_dir, '*.png')) +
                      glob(os.path.join(args.dirty_dir, '*.jpg')))
    print(f'==> Restoring {len(files)} real images')
    os.makedirs(args.save_dir, exist_ok=True)

    ssim_dr = 1.0 if args.standard_ssim else 2.0
    psnr_sum = ssim_sum = lpips_sum = 0.0
    n = 0
    for f in files:
        img = load_img(f).astype(np.float32) / 255.
        out, _ = restore_image(model, img, device=device)
        spath = os.path.join(args.save_dir, os.path.splitext(os.path.basename(f))[0] + '.png')
        save_img(spath, out)

        if args.mode == 'fr':
            gt_path = os.path.join(args.gt_dir, os.path.basename(f))
            gt = load_img(gt_path).astype(np.float32) / 255.
            if gt.shape[:2] != out.shape[:2]:
                import cv2
                gt = cv2.resize(gt, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
            o = out / out.max()
            g = gt / gt.max()
            p = psnr(o, g)
            s = ssim(o, g, data_range=ssim_dr)
            psnr_sum += p
            ssim_sum += s
            line = f'{os.path.basename(f)}: PSNR {p:.4f}, SSIM {s:.4f}'
            if not args.no_lpips:
                l = compute_lpips(spath, gt_path)
                lpips_sum += l
                line += f', LPIPS {l:.4f}'
            print(line)
            n += 1

    if args.mode == 'fr' and n:
        print('============ Results (full-reference) ============')
        print(f'PSNR: {psnr_sum / n:.4f} dB')
        print(f'SSIM: {ssim_sum / n:.4f}  (data_range={ssim_dr:g})')
        if not args.no_lpips:
            print(f'LPIPS: {lpips_sum / n:.4f}')


if __name__ == '__main__':
    main()
