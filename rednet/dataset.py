"""Datasets for ReDNet training / evaluation."""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils import data as data


def imread_float(path):
    """Read an image as RGB float32 in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.


def _list_images(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.JPG', '.PNG')
    files = [f for f in sorted(os.listdir(folder)) if f.endswith(exts)]
    return [os.path.join(folder, f) for f in files]


def padding(img_gt, img_lq, img_mask, gt_size):
    """Reflect-pad images smaller than the target patch size."""
    h, w, _ = img_gt.shape
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    if h_pad == 0 and w_pad == 0:
        return img_gt, img_lq, img_mask
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_mask = cv2.copyMakeBorder(img_mask, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    return img_gt, img_lq, img_mask


def paired_random_crop(img_gt, img_lq, img_mask, gt_size):
    h, w, _ = img_gt.shape
    top = random.randint(0, h - gt_size) if h > gt_size else 0
    left = random.randint(0, w - gt_size) if w > gt_size else 0
    img_gt = img_gt[top:top + gt_size, left:left + gt_size, :]
    img_lq = img_lq[top:top + gt_size, left:left + gt_size, :]
    img_mask = img_mask[top:top + gt_size, left:left + gt_size, :]
    return img_gt, img_lq, img_mask


def random_augmentation(img_gt, img_lq, img_mask):
    """Random horizontal / vertical flips and 90-degree rotation."""
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    def _aug(img):
        if hflip:
            img = np.ascontiguousarray(img[:, ::-1, :])
        if vflip:
            img = np.ascontiguousarray(img[::-1, :, :])
        if rot90:
            img = np.ascontiguousarray(np.rot90(img))
        return img

    return _aug(img_gt), _aug(img_lq), _aug(img_mask)


class DirtTrainDataset(data.Dataset):
    """Paired (dirty, clean, dirt-mask) training dataset.

    Folder layout:
        dataroot_gt/gt images (clean, 256x256 or larger)
        dataroot_lq/dirty images
        dataroot_mask/dirt masks (optional; pass None to disable)
    All images are matched by sorted order of filenames.
    """

    def __init__(self, dataroot_gt, dataroot_lq, dataroot_mask=None, gt_size=384,
                 geometric_augs=True, use_mask=True):
        super(DirtTrainDataset, self).__init__()
        self.gt_paths = _list_images(dataroot_gt)
        self.lq_paths = _list_images(dataroot_lq)
        self.mask_paths = _list_images(dataroot_mask) if (dataroot_mask is not None and use_mask) else None
        assert len(self.gt_paths) == len(self.lq_paths), \
            f'gt/lq mismatch: {len(self.gt_paths)} vs {len(self.lq_paths)}'
        if self.mask_paths is not None:
            assert len(self.mask_paths) == len(self.gt_paths)
        self.gt_size = gt_size
        self.geometric_augs = geometric_augs
        self.use_mask = use_mask and self.mask_paths is not None

    def __getitem__(self, index):
        index = index % len(self.gt_paths)
        img_gt = imread_float(self.gt_paths[index])
        img_lq = imread_float(self.lq_paths[index])
        if self.use_mask:
            img_mask = imread_float(self.mask_paths[index])
        else:
            img_mask = np.zeros_like(img_gt)

        img_gt, img_lq, img_mask = padding(img_gt, img_lq, img_mask, self.gt_size)
        img_gt, img_lq, img_mask = paired_random_crop(img_gt, img_lq, img_mask, self.gt_size)
        if self.geometric_augs:
            img_gt, img_lq, img_mask = random_augmentation(img_gt, img_lq, img_mask)

        to_tensor = lambda img: torch.from_numpy(
            np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

        return {
            'lq': to_tensor(img_lq),
            'gt': to_tensor(img_gt),
            'mask': to_tensor(img_mask),
            'lq_path': self.lq_paths[index],
            'gt_path': self.gt_paths[index],
        }

    def __len__(self):
        return len(self.gt_paths)


class PairedValDataset(data.Dataset):
    """Paired (dirty, clean) evaluation dataset, matched by sorted order."""

    def __init__(self, dataroot_lq, dataroot_gt):
        super(PairedValDataset, self).__init__()
        self.lq_paths = _list_images(dataroot_lq)
        self.gt_paths = _list_images(dataroot_gt)
        assert len(self.lq_paths) == len(self.gt_paths), \
            f'lq/gt mismatch: {len(self.lq_paths)} vs {len(self.gt_paths)}'

    def __getitem__(self, index):
        img_lq = imread_float(self.lq_paths[index])
        img_gt = imread_float(self.gt_paths[index])
        to_tensor = lambda img: torch.from_numpy(
            np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        return {'lq': to_tensor(img_lq), 'gt': to_tensor(img_gt),
                'lq_path': self.lq_paths[index], 'gt_path': self.gt_paths[index]}

    def __len__(self):
        return len(self.lq_paths)


class ImageFolderDataset(data.Dataset):
    """A plain folder of images for inference."""

    def __init__(self, dataroot):
        super(ImageFolderDataset, self).__init__()
        self.paths = _list_images(dataroot)

    def __getitem__(self, index):
        img = imread_float(self.paths[index])
        return {'lq': torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float(),
                'lq_path': self.paths[index]}

    def __len__(self):
        return len(self.paths)
