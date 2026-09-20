"""Common utilities: image IO, padding, metrics."""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_img(path, img):
    """Save a uint8 or float [0,1] RGB image."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def pad_to_multiple(tensor, factor=8):
    """Reflect-pad a (B, C, H, W) tensor so H and W are multiples of `factor`."""
    h, w = tensor.shape[-2], tensor.shape[-1]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    return F.pad(tensor, (0, padw, 0, padh), 'reflect')


def psnr(img1, img2):
    """PSNR for float images in [0, 1] (RGB, HWC), data_range=1."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def ssim(img1, img2, data_range=2.0):
    """SSIM for float images in [0, 1] (RGB, HWC).

    ``data_range=2.0`` reproduces the metric convention of the paper
    (scikit-image <= 0.19 assumed a range of 2 for float inputs, which reads
    ~0.02-0.15 higher than the modern default). Pass ``data_range=1.0`` for
    the modern standard convention.
    """
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=2, data_range=data_range)


def compute_lpips(img1, img2, net='alex'):
    """LPIPS for uint8 image paths or RGB uint8 arrays, matching the
    evaluation protocol of the paper (perceptual metrics on saved images)."""
    import lpips as lpips_pkg
    import torch as _torch
    loss_fn = lpips_pkg.LPIPS(net=net)
    if isinstance(img1, (str, os.PathLike)):
        img1 = load_img(str(img1))
    if isinstance(img2, (str, os.PathLike)):
        img2 = load_img(str(img2))
    t1 = _torch.from_numpy(img1.astype(np.float32) / 255. * 2 - 1).permute(2, 0, 1)[None]
    t2 = _torch.from_numpy(img2.astype(np.float32) / 255. * 2 - 1).permute(2, 0, 1)[None]
    with _torch.no_grad():
        return float(loss_fn(t1, t2))


@torch.no_grad()
def restore_image(model, img_rgb_float, device='cuda', max_size=None):
    """Run restoration on a float [0,1] RGB image (HWC).

    Args:
        model: ReDNet model (returns (out, dirt_pattern)).
        img_rgb_float: HWC float32 in [0, 1].
        max_size: optionally downscale so min(H, W) == max_size before inference.
    Returns:
        restored HWC float32 in [0, 1], and the estimated dirt pattern.
    """
    img = img_rgb_float
    if max_size is not None and min(img.shape[0], img.shape[1]) > max_size:
        h, w = img.shape[:2]
        scale = max_size / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    inp = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()[None].to(device)
    inp = pad_to_multiple(inp, factor=8)
    out, dirt_pattern = model(inp)
    out = out[:, :, :img.shape[0], :img.shape[1]]
    dirt_pattern = dirt_pattern[:, :, :img.shape[0], :img.shape[1]]
    out = torch.clamp(out, 0, 1)
    return (out[0].permute(1, 2, 0).cpu().numpy(),
            dirt_pattern[0].permute(1, 2, 0).cpu().numpy())


@torch.no_grad()
def restore_image_blockwise(model, img_rgb_float, device='cuda', block=512, overlap=32):
    """Block-wise restoration for very large images (e.g., 12 MP).

    The image is padded so it can be divided into overlapping blocks; each
    block is restored independently and blended back with linear weighting in
    the overlap regions.
    """
    h, w = img_rgb_float.shape[:2]
    stride = block - overlap
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img_rgb_float, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        img = img_rgb_float
    H, W = img.shape[:2]

    out = np.zeros_like(img)
    weight = np.zeros((H, W, 1), dtype=np.float32)
    # linear blending weights inside a block (ramp in the overlap border)
    win = np.ones((block, block, 1), dtype=np.float32)
    ramp = overlap
    if ramp > 0:
        ramp_arr = np.linspace(0, 1, ramp + 1)[1:]
        win[:ramp, :, 0] *= ramp_arr[:, None]
        win[-ramp:, :, 0] *= ramp_arr[::-1][:, None]
        win[:, :ramp, 0] *= ramp_arr[None, :]
        win[:, -ramp:, 0] *= ramp_arr[::-1][None, :]

    n_blocks = 0
    t0_sum = 0.0
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2, x2 = min(y + block, H), min(x + block, W)
            y1, x1 = y2 - block, x2 - block
            patch = img[y1:y2, x1:x2]
            inp = torch.from_numpy(np.ascontiguousarray(np.transpose(patch, (2, 0, 1)))).float()[None].to(device)
            inp = pad_to_multiple(inp, factor=8)
            out_patch, _ = model(inp)
            out_patch = torch.clamp(out_patch, 0, 1)
            out_patch = out_patch[0].permute(1, 2, 0).cpu().numpy()
            out[y1:y2, x1:x2] += out_patch * win
            weight[y1:y2, x1:x2] += win
            n_blocks += 1
    out = out / np.maximum(weight, 1e-8)
    return out[:h, :w]
