# src/scene_detection/__init__.py
from .interface import Scene, Scenes, SceneDetector
from .registry import register_detector, create_detector, available_detectors

# Import các backend mặc định để auto-register
from .pyscenedetect_backend import PySceneDetectDetector   # noqa: F401
from .transnetv2_backend import TransNetV2Detector         # noqa: F401

__all__ = [
    "Scene", "Scenes", "SceneDetector",
    "register_detector", "create_detector", "available_detectors",
]
