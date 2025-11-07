# src/keyframe/medoid_selector.py
# Generic medoid-based selector that works with any DistanceMetric.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Any

import os
import cv2
import numpy as np
import torch

from src.distance_selector.registry import create_metric
from src.distance_selector.interface import DistanceMetric


@dataclass(frozen=True)
class Keyframe:
    frame_idx: int
    score: float
    scene_id: int


class MedoidSelector:
    """
    Generic keyframe selector that uses an arbitrary DistanceMetric backend.
    """
    def __init__(self, metric: DistanceMetric) -> None:
        self.metric = metric

    @staticmethod
    def _sample_indices(s: int, e: int, stride: int, cap: Optional[int]) -> List[int]:
        idxs = list(range(s, e + 1, max(1, stride)))
        if cap is not None and cap > 0 and len(idxs) > cap:
            sel = np.linspace(0, len(idxs) - 1, cap, dtype=int)
            idxs = [idxs[i] for i in sel]
        return idxs

    @staticmethod
    def _greedy_medoid(D: np.ndarray, k: int, nms_radius: int) -> List[int]:
        if D.size == 0:
            return []
        sums = D.sum(axis=1)
        candidates = list(range(D.shape[0]))
        picked: List[int] = []
        while len(picked) < k and candidates:
            best = min(candidates, key=lambda i: sums[i])
            picked.append(best)
            candidates = [i for i in candidates if abs(i - best) > nms_radius]
        return sorted(picked)

    def select_for_scene(
        self,
        video_path: str,
        scene_range: Tuple[int, int],
        sample_stride: int = 10,
        max_frames_per_scene: int = 30,
        keyframes_per_scene: int = 1,
        nms_radius: int = 2,
        resize_to: Optional[Tuple[int, int]] = (320, 180),
        scene_id: int = -1,
        batch_pairs: int = 16,
    ) -> List[Keyframe]:
        s, e = scene_range
        idxs = self._sample_indices(s, e, sample_stride, max_frames_per_scene)
        if not idxs:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames = []
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()

        if not frames:
            return []

        # Preprocess using metric-specific normalization
        tensors = [self.metric.preprocess_bgr(f) for f in frames]

        # Compute pairwise distances
        D = self.metric.pairwise_matrix(tensors, batch_pairs=batch_pairs)

        sums = D.sum(axis=1)
        picks = self._greedy_medoid(D, k=keyframes_per_scene, nms_radius=nms_radius)

        return [Keyframe(frame_idx=idxs[i], score=float(sums[i]), scene_id=scene_id) for i in picks]


# Convenience function to build + run selector
def select_keyframes_for_scenes(
    video_path: str,
    scenes: Sequence[Tuple[int, int]],
    distance_backend: str = "lpips",
    distance_kwargs: Optional[dict] = None,
    sample_stride: int = 10,
    max_frames_per_scene: int = 30,
    keyframes_per_scene: int = 1,
    nms_radius: int = 2,
    resize_to: Optional[Tuple[int, int]] = (320, 180),
    batch_pairs: int = 16,
) -> List[Keyframe]:
    metric = create_metric(distance_backend, **(distance_kwargs or {}))
    selector = MedoidSelector(metric=metric)

    all_keys: List[Keyframe] = []
    for sid, (s, e) in enumerate(scenes):
        ks = selector.select_for_scene(
            video_path=video_path,
            scene_range=(s, e),
            sample_stride=sample_stride,
            max_frames_per_scene=max_frames_per_scene,
            keyframes_per_scene=keyframes_per_scene,
            nms_radius=nms_radius,
            resize_to=resize_to,
            scene_id=sid,
            batch_pairs=batch_pairs,
        )
        all_keys.extend(ks)
    return all_keys
