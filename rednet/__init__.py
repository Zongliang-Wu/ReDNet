"""ReDNet: Removing Dirt Network.

PyTorch implementation of the paper "Restoration of Images Taken Through a
Dirty Window Using Optics-guided Transformer" (IEEE TIP 2025).
"""
from .archs.rednet_arch import ReDNet

__all__ = ['ReDNet']
