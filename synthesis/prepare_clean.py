"""Prepare clean 256x256 patches from DIV2K / Flickr2K images.

Each source image is rotated to portrait, center-square cropped, and resized
to ``--size`` (256 by default), following the dataset construction of the
paper. Suggested split (matching the released data): Flickr2K images
0001-2000 for training, 2001+ for testing (100 random images).

Usage:
    python synthesis/prepare_clean.py --input_dir /path/to/Flickr2K_HR \
        --output_dir datasets_clean/train --size 256
"""
import argparse
import os

import cv2
import numpy as np


def process(img, size):
    H, W, C = img.shape
    if W > H:
        img = np.transpose(img, [1, 0, 2])
        H, W = img.shape[:2]
    mid = W // 2
    half = H // 2
    img = img[:, mid - half:mid + half, :]
    return cv2.resize(img, (size, size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='DIV2K_HR or Flickr2K_HR folder')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    for i, f in enumerate(files):
        img = cv2.cvtColor(cv2.imread(os.path.join(args.input_dir, f)), cv2.COLOR_BGR2RGB)
        out = process(img, args.size)
        cv2.imwrite(os.path.join(args.output_dir, f'{i}.png'),
                    cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f'Saved {len(files)} clean patches to {args.output_dir}')


if __name__ == '__main__':
    main()
