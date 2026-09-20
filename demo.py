"""ReDNet demo: restore images taken through dirty windows.

Works on a single image or a folder. Large images (e.g., 12 MP) can be
processed with block-wise restoration via --blockwise.

Usage:
    python demo.py --weights pretrained/rednet_dirt.pth --input dirty.jpg --output_dir results
    python demo.py --weights pretrained/rednet_dirt.pth --input ./my_photos --output_dir results --blockwise
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rednet.archs.rednet_arch import ReDNet
from rednet.utils import load_img, save_img, restore_image, restore_image_blockwise


def main():
    parser = argparse.ArgumentParser(description='ReDNet demo')
    parser.add_argument('--weights', default='pretrained/rednet_dirt.pth', type=str)
    parser.add_argument('--input', required=True, type=str, help='an image or a folder of images')
    parser.add_argument('--output_dir', default='results', type=str)
    parser.add_argument('--blockwise', action='store_true',
                        help='block-wise restoration for very large images (>= several MP)')
    parser.add_argument('--block', default=512, type=int, help='block size for blockwise mode')
    parser.add_argument('--overlap', default=32, type=int, help='block overlap for blending')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ReDNet(in_ch=3, out_ch=3, num_blocks=[4, 6, 6, 8, 16], num_refinement_blocks=4,
                   ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt.get('params', ckpt))
    model.to(device).eval()
    print(f'==> Loaded weights: {args.weights}')

    if os.path.isdir(args.input):
        from natsort import natsorted
        from glob import glob
        files = natsorted(glob(os.path.join(args.input, '*.png')) +
                          glob(os.path.join(args.input, '*.jpg')) +
                          glob(os.path.join(args.input, '*.JPG')))
    else:
        files = [args.input]
    os.makedirs(args.output_dir, exist_ok=True)

    for f in files:
        img = load_img(f).astype('float32') / 255.
        t0 = time.time()
        if args.blockwise:
            out = restore_image_blockwise(model, img, device=device,
                                          block=args.block, overlap=args.overlap)
        else:
            out, _ = restore_image(model, img, device=device)
        dt = time.time() - t0
        name = os.path.splitext(os.path.basename(f))[0] + '.png'
        save_img(os.path.join(args.output_dir, name), out)
        print(f'{os.path.basename(f)}: {img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]} '
              f'({dt:.2f}s)')


if __name__ == '__main__':
    main()
