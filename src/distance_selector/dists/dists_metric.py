# src/distance/dists_metric.py
# DISTS metric implementation (Ding et al.). Expects inputs in [0, 1].

from __future__ import annotations
from typing import Any
import torch
import numpy as np
import cv2

from ..interface import DistanceMetric
from ..registry import register_metric
from .. import utils as U

# pip install dists-pytorch
from DISTS_pytorch import DISTS


@register_metric("dists")
class DISTSMetric(DistanceMetric):
    """
    DISTS distance. Expects inputs in [0, 1].
    Params:
      - device: 'cuda' | 'cpu' (auto if not provided)
      - as_distance: bool  # DISTS outputs lower is more similar; often used as distance already.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        dev = self.params.get("device")
        if dev not in ("cpu", "cuda"):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = dev

        self._dists = DISTS().to(self._device).eval()
        self._as_distance = bool(self.params.get("as_distance", True))

    def device(self) -> str:
        return self._device

    def preprocess_bgr(self, bgr: np.ndarray) -> torch.Tensor:
        """Preprocess BGR image with resizing for memory efficiency."""
        x = bgr
        assert x.ndim == 3 and x.shape[2] in (1, 3), f"Expect HxWxC image, got {x.shape}"
        # Ensure 3 channels
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        
        # Resize to max 256x256 to prevent OOM in batch processing
        H, W = x.shape[:2]
        max_side = 256
        if max(H, W) > max_side:
            scale = max_side / float(max(H, W))
            newH, newW = int(round(H * scale)), int(round(W * scale))
            x = cv2.resize(x, (newW, newH), interpolation=cv2.INTER_AREA)
        
        # Ensure minimum size for DISTS
        H, W = x.shape[:2]
        if min(H, W) < 64:
            scale = 64.0 / min(H, W)
            newH, newW = int(round(H * scale)), int(round(W * scale))
            x = cv2.resize(x, (newW, newH), interpolation=cv2.INTER_LINEAR)
        
        # BGR -> RGB, [0, 255] -> [0, 1]
        x = x.astype(np.float32)
        if x.max() > 1.0:
            x /= 255.0
        x = x[..., ::-1].copy()  # BGR -> RGB, .copy() fixes negative stride issue
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).contiguous().to(self._device)
        return t

    @torch.no_grad()
    def pair_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        # DISTS returns a similarity-like measure (lower ~ better).
        val = float(self._dists(t1, t2).item())
        return val if self._as_distance else -val
