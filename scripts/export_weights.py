"""Export a clean release checkpoint (model params only, no optimizer state)
and print its SHA256 checksum.

Usage:
    python scripts/export_weights.py --input experiments/ReDNet/models/net_g_208000.pth \
        --output pretrained/rednet_dirt.pth
"""
import argparse
import hashlib
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location='cpu')
    params = ckpt.get('params', ckpt)
    n_params = sum(v.numel() for v in params.values())
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({'params': params}, args.output)
    print(f'Saved {args.output}: {n_params / 1e6:.3f}M parameters, '
          f'{os.path.getsize(args.output) / 1024 / 1024:.2f} MB')

    sha = hashlib.sha256()
    with open(args.output, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            sha.update(chunk)
    print(f'SHA256: {sha.hexdigest()}')


if __name__ == '__main__':
    main()
