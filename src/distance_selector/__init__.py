# src/distance/__init__.py
from .interface import DistanceMetric
from .registry import register_metric, create_metric, available_metrics

# Register built-ins
from .lpips.lpips_metric import LPIPSMetric   # noqa: F401
from .dists.dists_metric import DISTSMetric   # noqa: F401

__all__ = [
    "DistanceMetric",
    "register_metric",
    "create_metric",
    "available_metrics",
]
