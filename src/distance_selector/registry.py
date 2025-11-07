# src/distance/registry.py
from typing import Dict, Type, Any
from .interface import DistanceMetric

_METRIC_REGISTRY: Dict[str, Type[DistanceMetric]] = {}

def register_metric(name: str):
    def _wrap(cls: Type[DistanceMetric]) -> Type[DistanceMetric]:
        key = name.strip().lower()
        if key in _METRIC_REGISTRY:
            raise ValueError(f"Metric '{key}' already registered.")
        _METRIC_REGISTRY[key] = cls
        return cls
    return _wrap

def available_metrics():
    return sorted(_METRIC_REGISTRY.keys())

def create_metric(name: str, **kwargs: Any) -> DistanceMetric:
    key = name.strip().lower()
    if key not in _METRIC_REGISTRY:
        raise KeyError(f"Unknown metric '{key}'. Available: {available_metrics()}")
    return _METRIC_REGISTRY[key](**kwargs)
