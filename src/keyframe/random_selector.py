# src/keyframe/random_selector.py
# Random keyframe selector for comparison/testing purposes.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Any

import os
import cv2
import numpy as np
import random


@dataclass(frozen=True)
class Keyframe:
    frame_idx: int
    score: float
    scene_id: int


class RandomSelector:
    """
    Random keyframe selector.
    - Samples indices within a scene (stride + optional cap).
    - Randomly picks up to k indices with simple NMS on the *sampled index positions*.
    - Does NOT load frames or compute distances (fast & lightweight).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _sample_indices(s: int, e: int, stride: int, cap: Optional[int]) -> List[int]:
        """Create a subsampled index list [s, s+stride, ..., e].
        If cap is set and the list is longer than cap, thin it evenly via linspace."""
        idxs = list(range(s, e + 1, max(1, stride)))
        if cap is not None and cap > 0 and len(idxs) > cap:
            sel = np.linspace(0, len(idxs) - 1, cap, dtype=int)
            idxs = [idxs[i] for i in sel]
        return idxs

    def _random_pick_with_nms(self, n: int, k: int, nms_radius: int) -> List[int]:
        """Pick up to k positions in [0..n-1] at random, applying NMS on positions.
        NMS is enforced on *sample list positions* (not absolute frame numbers),
        consistent with MedoidSelector's behavior."""
        if n <= 0 or k <= 0:
            return []
        positions = list(range(n))
        self.rng.shuffle(positions)

        picked: List[int] = []
        alive = np.ones(n, dtype=bool)  # alive[j] = True if still selectable

        for p in positions:
            if not alive[p]:
                continue
            picked.append(p)
            if len(picked) == k:
                break
            # suppress neighbors within radius on sampled positions
            lo = max(0, p - nms_radius)
            hi = min(n - 1, p + nms_radius)
            alive[lo : hi + 1] = False

        # Return positions sorted so output is in temporal order
        return sorted(picked)

    def select_for_scene(
        self,
        video_path: str,
        scene_range: Tuple[int, int],
        sample_stride: int = 10,
        max_frames_per_scene: int = 30,
        keyframes_per_scene: int = 1,
        nms_radius: int = 2,
        resize_to: Optional[Tuple[int, int]] = (320, 180),  # kept for API parity; not used
        scene_id: int = -1,
        batch_pairs: int = 16,  # kept for API parity; not used
    ) -> List[Keyframe]:
        """Select random keyframes for a single scene.

        Notes:
        - We don't read frames or compute distances (score will be 0.0).
        - If NMS prunes too many neighbors, fewer than k keyframes may be returned.
        """
        s, e = scene_range
        idxs = self._sample_indices(s, e, sample_stride, max_frames_per_scene)
        if not idxs:
            return []

        pos = self._random_pick_with_nms(n=len(idxs), k=keyframes_per_scene, nms_radius=nms_radius)
        return [Keyframe(frame_idx=idxs[p], score=0.0, scene_id=scene_id) for p in pos]


# Convenience function mirroring the MedoidSelector helper
def select_random_keyframes_for_scenes(
    video_path: str,
    scenes: Sequence[Tuple[int, int]],
    sample_stride: int = 10,
    max_frames_per_scene: int = 30,
    keyframes_per_scene: int = 1,
    nms_radius: int = 2,
    seed: Optional[int] = None,
) -> List[Keyframe]:
    """Randomly select keyframes for multiple scenes."""
    selector = RandomSelector(seed=seed)
    all_keys: List[Keyframe] = []
    for sid, (s, e) in enumerate(scenes):
        ks = selector.select_for_scene(
            video_path=video_path,
            scene_range=(s, e),
            sample_stride=sample_stride,
            max_frames_per_scene=max_frames_per_scene,
            keyframes_per_scene=keyframes_per_scene,
            nms_radius=nms_radius,
            scene_id=sid,
        )
        all_keys.extend(ks)
    return all_keys
