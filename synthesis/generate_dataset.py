"""Generate the synthetic (dirty, clean, mask) dataset.

Pairs each clean patch with a dirt-pattern mask and synthesizes the dirty
image with the imaging model in ``dust_syn.py``. The number of triplets is
``n_clean x patterns_per_image``; use ``--n_train 4000 --n_test 100`` to
match the released dataset size.

Usage:
    python synthesis/generate_dataset.py \
        --clean_dir datasets_clean/train --pattern_dir patterns_256 \
        --output_dir datasets/syn_train --n 4000 --seed 100
"""
import argparse
import os

import cv2
import numpy as np
from natsort import natsorted

from dust_syn import dusty_syn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', required=True, help='clean 256x256 patches')
    parser.add_argument('--pattern_dir', required=True, help='256x256 dirt pattern masks')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--n', type=int, default=4000, help='number of triplets to generate')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--include_no_dirt', action='store_true',
                        help='append ~0.7%% clean (no-dirt) samples, following Tab. I of the paper')
    parser.add_argument('--alpha_a_range', type=float, nargs=2, default=[0.5, 2.0],
                        help='sampling range of the attenuation factor (default: 0.5 2.0, Sec. III-B)')
    parser.add_argument('--alpha_s_range', type=float, nargs=2, default=[0.5, 2.0],
                        help='sampling range of the scattering factor (default: 0.5 2.0, Sec. III-B)')
    args = parser.parse_args()

    sample_alphas = lambda: (rng.uniform(*args.alpha_a_range), rng.uniform(*args.alpha_s_range))

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.output_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'dirty'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'mask'), exist_ok=True)

    clean_files = natsorted(f for f in os.listdir(args.clean_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    pattern_files = natsorted(f for f in os.listdir(args.pattern_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    assert clean_files and pattern_files, 'empty clean or pattern folder'

    n = 0
    i = 0
    while n < args.n:
        clean = cv2.cvtColor(cv2.imread(os.path.join(args.clean_dir, clean_files[i % len(clean_files)])),
                             cv2.COLOR_BGR2RGB)
        no_dirt = args.include_no_dirt and rng.uniform() < 0.007
        if no_dirt:
            dirty = clean.astype(np.float32) / clean.max()
            mask = np.zeros(clean.shape[:2], np.float32)
        else:
            p = pattern_files[int(rng.integers(0, len(pattern_files)))]
            mask = cv2.imread(os.path.join(args.pattern_dir, p), cv2.IMREAD_GRAYSCALE)
            a1_ran, a2_ran = sample_alphas()
            dirty = dusty_syn(clean, mask, a1_ran=a1_ran, a2_ran=a2_ran, a0=1.0, rng=rng)
            mask = mask.astype(np.float32) / 255.
        dirty = np.clip(dirty, 0, 1)

        cv2.imwrite(os.path.join(args.output_dir, 'gt', f'{n}.png'),
                    cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.output_dir, 'dirty', f'{n}.png'),
                    cv2.cvtColor((dirty * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.output_dir, 'mask', f'{n}.png'),
                    (mask * 255).round().astype(np.uint8))
        n += 1
        i += 1
    print(f'Generated {n} triplets in {args.output_dir}/{{gt,dirty,mask}}')


if __name__ == '__main__':
    main()
