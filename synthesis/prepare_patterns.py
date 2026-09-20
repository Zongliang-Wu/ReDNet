"""Prepare 256x256 dirt-pattern masks from a raw pattern library.

Two pattern sources are used in the paper:
  1. Patterns rendered with the `dust synthesis tool` of Gu et al. (2007),
     "Dirty Glass: Rendering Contamination on Transparent Surfaces" --
     large rendered dirt fields are cropped into square patches.
  2. Self-collected patterns: photographs taken through real dirty windows
     with a whiteboard behind, preprocessed (color conversion, vignetting
     correction, inversion, cropping) and stored as grayscale masks where
     white = dirt.

This script turns raw patterns into `--size` masks: four corner crops of
``--crop`` pixels (512/1024, depending on the source resolution) are taken
from each pattern and resized, following the original pipeline.

Usage:
    python synthesis/prepare_patterns.py --pattern_dir /path/to/patterns \
        --output_dir patterns_256 --crop 512 --size 256
"""
import argparse
import os

import cv2
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def corner_crops(mask, W_size):
    """Four corner crops of size W_size (the original sampling scheme)."""
    h, w = mask.shape[:2]
    W_size = min(W_size, h, w)
    return [
        mask[0:W_size, 0:W_size],
        mask[-W_size - 1:-1, 0:W_size],
        mask[-W_size - 1:-1, -W_size - 1:-1],
        mask[0:W_size, -W_size - 1:-1],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern_dir', required=True, help='raw pattern images (white = dirt)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--crop', type=int, default=512, help='corner crop size before resize')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--invert', choices=['auto', 'yes', 'no'], default='auto',
                        help='dirt must be WHITE on black. "auto" inverts images whose mean '
                             'brightness exceeds 0.5 (e.g., raw whiteboard captures with dark dirt)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.pattern_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    idx = 0
    for f in files:
        img = cv2.cvtColor(cv2.imread(os.path.join(args.pattern_dir, f)), cv2.COLOR_BGR2RGB)
        mask = rgb2gray(img).astype(np.float32)
        if args.invert == 'yes' or (args.invert == 'auto' and mask.mean() > 127.5):
            mask = 255.0 - mask
        for crop in corner_crops(mask, args.crop):
            crop = cv2.resize(crop, (args.size, args.size))
            cv2.imwrite(os.path.join(args.output_dir, f'{idx}.png'), crop)
            idx += 1
    print(f'Saved {idx} pattern masks to {args.output_dir}')


if __name__ == '__main__':
    main()
