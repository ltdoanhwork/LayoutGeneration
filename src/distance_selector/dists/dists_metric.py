# src/distance/dists_metric.py
# DISTS metric implementation (Ding et al.). Expects inputs in [0, 1].

from __future__ import annotations
from typing import Any
import torch
import numpy as np

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
        return U.bgr_to_tensor_0_1(bgr).to(self._device)

    @torch.no_grad()
    def pair_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        # DISTS returns a similarity-like measure (lower ~ better).
        val = float(self._dists(t1, t2).item())
        return val if self._as_distance else -val
