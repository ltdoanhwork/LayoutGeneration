from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import json
import cv2

class SceneSample:
    """One scene: frames (optional for MS-SWD), feats (T,D), optional flow (T,)."""
    def __init__(self, frames: Optional[List], feats: np.ndarray, flow: Optional[np.ndarray]):
        self.frames = frames
        self.feats = feats
        self.flow = flow

def list_scene_dirs(dataset_root: str) -> List[Path]:
    root = Path(dataset_root)
    # Expect structure: <root>/<video_stem>/scene_xxxx/
    return sorted([p for p in root.glob("*/*") if (p / "feats.npy").exists()])

def load_scene_dir(scene_dir: Path, load_frames: bool = True) -> SceneSample:
    feats = np.load(scene_dir / "feats.npy")  # (T,D) L2-normalized
    flow_path = scene_dir / "flow.npy"
    flow = np.load(flow_path) if flow_path.exists() else None
    frames = None
    if load_frames:
        frame_files = sorted((scene_dir / "frames").glob("*.jpg"))
        frames = [cv2.imread(str(fp)) for fp in frame_files]
    return SceneSample(frames=frames, feats=feats, flow=flow)

def build_epoch_index(dataset_root: str) -> List[Path]:
    return list_scene_dirs(dataset_root)
