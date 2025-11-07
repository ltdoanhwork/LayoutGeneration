# src/distance/interface.py
# All comments are in English.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np
import torch


class DistanceMetric(ABC):
    """
    Abstract distance metric between two images.
    Implementations may own a torch.Module and a target device.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.params = dict(kwargs)

    @abstractmethod
    def device(self) -> str:
        """Return the device string used by the metric ('cpu' or 'cuda')."""
        raise NotImplementedError

    @abstractmethod
    def preprocess_bgr(self, bgr: np.ndarray) -> torch.Tensor:
        """
        Convert a BGR uint8 image (H, W, C) to a metric-specific tensor,
        usually shape (1, 3, H, W) with the correct normalization.
        """
        raise NotImplementedError

    @abstractmethod
    def pair_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        """
        Compute the distance between two preprocessed tensors and return a Python float.
        """
        raise NotImplementedError

    def pairwise_matrix(self, tensors: Sequence[torch.Tensor], batch_pairs: int = 16) -> np.ndarray:
        """
        Default O(N^2) pairwise distance computation (upper triangle fill).
        Implementations can override for vectorized speed-ups if desired.
        """
        import numpy as np
        n = len(tensors)
        D = np.zeros((n, n), dtype=np.float32)
        with torch.no_grad():
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))

            # Small chunks to limit memory
            for k in range(0, len(pairs), batch_pairs):
                chunk = pairs[k : k + batch_pairs]
                for (i, j) in chunk:
                    d = self.pair_distance(tensors[i], tensors[j])
                    D[i, j] = D[j, i] = float(d)
        return D
