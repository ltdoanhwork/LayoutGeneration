# src/scene_detection/registry.py
from typing import Dict, Type, Any
from .interface import SceneDetector

_DETECTOR_REGISTRY: Dict[str, Type[SceneDetector]] = {}

def register_detector(name: str):
    """Decorator để đăng ký backend mới."""
    def _wrap(cls: Type[SceneDetector]) -> Type[SceneDetector]:
        key = name.strip().lower()
        if key in _DETECTOR_REGISTRY:
            raise ValueError(f"Detector '{key}' already registered.")
        cls.name = key
        _DETECTOR_REGISTRY[key] = cls
        return cls
    return _wrap

def available_detectors():
    return sorted(_DETECTOR_REGISTRY.keys())

def create_detector(name: str, **kwargs: Any) -> SceneDetector:
    key = name.strip().lower()
    if key not in _DETECTOR_REGISTRY:
        raise KeyError(f"Unknown detector '{key}'. Available: {available_detectors()}")
    return _DETECTOR_REGISTRY[key](**kwargs)
