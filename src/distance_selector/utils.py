# src/distance/utils.py
# Shared image->tensor helpers.

from __future__ import annotations
import numpy as np
import torch
import cv2


def bgr_to_tensor_minus1_1(bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 to normalized tensor in [-1, 1], shape (1, 3, H, W)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return ten


def bgr_to_tensor_0_1(bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 to normalized tensor in [0, 1], shape (1, 3, H, W)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return ten
