import os
import torch
from typing import Any, Dict, List
from .interface import Scene, Scenes, SceneDetector
from .registry import register_detector

@register_detector("random")
class RandomSceneDetector(SceneDetector):
    """Random scene detection backend."""
    name: str = "random"

    def __init__(self, **kwargs: Any) -> None:
        """
        Backend receives free parameters through kwargs (e.g., threshold, model_dir, batch_size, ...).
        Helps CLI/pipeline inject configuration without code modification.
        """
        self.params: Dict[str, Any] = kwargs

    def detect(self, video_path: str) -> Scenes:
        """Returns a list of Scenes (start_frame, end_frame) by frame index (inclusive)."""
        # Randomly generate scene boundaries
        num_frames = self.params.get("num_frames", 100)
        scene_length = self.params.get("scene_length", 10)
        scenes = []
        for start in range(0, num_frames, scene_length):
            end = min(start + scene_length - 1, num_frames - 1)
            scenes.append((start, end))
        return scenes

    
