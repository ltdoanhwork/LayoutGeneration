# src/distance/lpips_metric.py
# LPIPS metric implementation.

from __future__ import annotations
from typing import Optional, Any
import torch
from src.distance_selector.lpips.src.lpips import lpips
import numpy as np
import cv2, numpy as np, torch

from ..interface import DistanceMetric
from ..registry import register_metric
from .. import utils as U


@register_metric("lpips")
class LPIPSMetric(DistanceMetric):
    """
    LPIPS distance (Alex/VGG/Squeeze). Expects inputs in [-1, 1].
    Params:
      - net: str = 'alex' | 'vgg' | 'squeeze'
      - device: Optional[str]  # auto: 'cuda' if available, else 'cpu'
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        dev = self.params.get("device")
        if dev not in ("cpu", "cuda"):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = dev

        net = self.params.get("net", "alex")
        self._lpips = lpips.LPIPS(net=net).to(self._device).eval()

    def device(self) -> str:
        return self._device

    # src/distance_selector/lpips_metric.py
    def preprocess_bgr(self, bgr: np.ndarray) -> torch.Tensor:
        x = bgr
        assert x.ndim == 3 and x.shape[2] in (1,3), f"Expect HxWxC image, got {x.shape}"
        # ensure 3 channels
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)

        # --- resize: make min side >= 64, cap max side to avoid huge memory ---
        H, W = x.shape[:2]
        min_side = min(H, W)
        if min_side < 64:
            scale = 64.0 / max(1, float(min_side))
            newH, newW = int(round(H*scale)), int(round(W*scale))
            newH = max(newH, 64); newW = max(newW, 64)
            x = cv2.resize(x, (newW, newH), interpolation=cv2.INTER_AREA)
            H, W = x.shape[:2]
        if max(H, W) > 1024:
            scale = 1024.0 / float(max(H, W))
            x = cv2.resize(x, (int(round(W*scale)), int(round(H*scale))), interpolation=cv2.INTER_AREA)

        # BGR->RGB, [0,1] -> [-1,1]
        x = x.astype(np.float32)
        if x.max() > 1.0: x /= 255.0
        x = x[..., ::-1]
        x = (x * 2.0) - 1.0
        t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).contiguous().to(self._device)
        return t


    @torch.no_grad()
    def pair_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        return float(self._lpips(t1, t2).item())
