# src/distance/lpips_metric.py
# LPIPS metric implementation.

from __future__ import annotations
from typing import Optional, Any
import torch
from src.distance_selector.lpips.src.lpips import lpips
import numpy as np

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

    def preprocess_bgr(self, bgr: np.ndarray) -> torch.Tensor:
        return U.bgr_to_tensor_minus1_1(bgr).to(self._device)

    @torch.no_grad()
    def pair_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        return float(self._lpips(t1, t2).item())
