"""Upload release weights and the LPIPS calibration file to Hugging Face Hub.

Prerequisites:
    pip install huggingface_hub
    export HF_TOKEN=<your token with write access>

Usage:
    python scripts/upload_hf.py --repo_id Zongliang-Wu/ReDNet \
        --files pretrained/rednet_dirt.pth pretrained/lpips_vgg.pth
"""
import argparse
import os

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', default='Zongliang-Wu/ReDNet', type=str)
    parser.add_argument('--files', nargs='+', required=True, type=str)
    parser.add_argument('--private', action='store_true')
    args = parser.parse_args()

    token = os.environ.get('HF_TOKEN')
    assert token, 'Please set the HF_TOKEN environment variable.'

    api = HfApi(token=token)
    create_repo(repo_id=args.repo_id, repo_type='model', private=args.private, exist_ok=True)
    for path in args.files:
        path_in_repo = os.path.relpath(path)
        print(f'Uploading {path} -> {args.repo_id}/{path_in_repo}')
        api.upload_file(path_or_fileobj=path, path_in_repo=path_in_repo,
                        repo_id=args.repo_id, repo_type='model')
    print('Done.')


if __name__ == '__main__':
    main()
