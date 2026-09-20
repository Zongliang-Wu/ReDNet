"""Dirty-window image synthesis model (Sec. III of the paper).

Implements the imaging model
    Id = a0 * Io - alpha_a * (Z . P) + alpha_s * P
where the inside background illumination Z is modulated by the scene content
(mean of the clean image), which produces the background-dependent dirt
color observed in real dirty-window photographs (dirt looks dark against
bright backgrounds and bright against dark backgrounds).

This is a faithful, vectorized port of the original research implementation
(`main_dust_syn.py::dusty_syn`).
"""
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def dusty_syn(ori, dust, a1_ran=None, a2_ran=None, a0=1, rng=None):
    """Synthesize a dirty image from a clean image and a dirt pattern.

    Args:
        ori (ndarray): HxWx3 uint8 clean image.
        dust (ndarray): HxW or HxWx3 uint8 dirt pattern (white = dirt).
        a1_ran / a2_ran (float): attenuation / scattering factors. If None,
            sampled as ``1.8 + U/3`` and ``0.9 + U/5`` (the paper setting).
        a0 (float): global transmission scale; set to ``1 + a2_ran/2`` by the
            default path.
        rng (np.random.Generator): optional RNG for reproducibility.

    Returns:
        ndarray: HxWx3 float in [0, 1]-ish range (clip before saving).
    """
    rng = rng or np.random
    W, H, C = ori.shape
    if W > H:
        ori = np.transpose(ori, [1, 0, 2])

    if a1_ran is None:
        a1 = 1.8
        a2 = 0.9
        a1_ran = a1 + rng.uniform() / 3
        a2_ran = a2 + rng.uniform() / 5
        a0 = 1 + a2_ran / 2

    if len(dust.shape) == 3:
        rows_d, cols_d, depth_d = dust.shape
    else:
        rows_d, cols_d = dust.shape
        depth_d = 1

    rows_o, cols_o, depth_o = ori.shape
    if rows_d > rows_o and cols_d > cols_o:
        xmax = round(rows_d - rows_o) - 1
        ymax = round(cols_d - cols_o) - 1
        x = rng.integers(0, xmax) if hasattr(rng, 'integers') else np.random.randint(0, xmax)
        y = rng.integers(0, ymax) if hasattr(rng, 'integers') else np.random.randint(0, ymax)
        dust_part = dust[x:x + rows_o - 1, y:y + cols_o - 1, 0:depth_d]
        dust_part = dust_part / 255
    else:
        dust_part = dust / 255

    ori = ori / ori.max()

    z = np.zeros((rows_o, cols_o, depth_o))
    r1 = 0.9 + rng.uniform() / 5
    r2 = 0.9 + rng.uniform() / 5
    r3 = 0.9 + rng.uniform() / 5

    if depth_d == 3:
        z1 = np.sum(ori, axis=2)
        z2 = np.sum(dust_part, axis=2)
        z2 = z2 / 3
    else:
        z1 = np.sum(ori, axis=2)
        z2 = dust_part
    z1 = z1 / 3
    z[:, :, 0] = z1 * z2 * r1
    z[:, :, 1] = z1 * z2 * r2
    z[:, :, 2] = z1 * z2 * r3

    if len(dust_part.shape) < 3:
        dust_part = np.expand_dims(dust_part, axis=2)
    dusty_pic = a0 * ori * 1 - z * a1_ran + a2_ran * dust_part
    return dusty_pic
