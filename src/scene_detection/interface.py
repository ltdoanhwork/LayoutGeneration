# src/scene_detection/interface.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass(frozen=True)
class Scene:
    """Scene based on frame index (inclusive)."""
    start_frame: int
    end_frame: int

Scenes = List[Scene]

class SceneDetector(ABC):
    """Standard interface for all scene detection backends."""
    name: str = "abstract"

    def __init__(self, **kwargs: Any) -> None:
        """
        Backend receives free parameters through kwargs (e.g., threshold, model_dir, batch_size, ...).
        Helps CLI/pipeline inject configuration without code modification.
        """
        self.params: Dict[str, Any] = kwargs

    @abstractmethod
    def detect(self, video_path: str) -> Scenes:
        """Returns a list of Scenes (start_frame, end_frame) by frame index (inclusive)."""
        raise NotImplementedError

    def close(self) -> None:
        """If backend needs to release resources (TF session, file handle...), override this method."""
        pass
