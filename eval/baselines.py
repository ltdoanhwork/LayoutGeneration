from __future__ import annotations
from typing import List, Tuple
import numpy as np
import cv2


def baseline_uniform(total_frames: int, m: int) -> List[int]:
    """Uniformly pick m frames over the whole video frame range."""
    if m <= 0 or total_frames <= 0:
        return []
    idx = np.linspace(0, total_frames - 1, m, dtype=int).tolist()
    return idx


def baseline_middle_of_scene(scenes: List[Tuple[int, int]], m: int) -> List[int]:
    """Pick middle frame of scenes; if more than needed, downsample; if fewer, fill by interpolation."""
    mids = [ (s + e) // 2 for (s, e) in scenes ]
    if len(mids) >= m:
        sel = np.linspace(0, len(mids) - 1, m, dtype=int).tolist()
        return [mids[i] for i in sel]
    # Fill the rest by uniform across entire range of scenes
    if not scenes:
        return mids
    all_start = scenes[0][0]
    all_end   = scenes[-1][1]
    need = m - len(mids)
    extra = np.linspace(all_start, all_end, need, dtype=int).tolist()
    return sorted(mids + extra)


def baseline_motion_peaks(video_path: str, m: int, stride: int = 1) -> List[int]:
    """
    Very simple motion-based baseline: pick frames with top |Δframe| energy.
    """
    if m <= 0:
        return []
    cap = cv2.VideoCapture(video_path)
    prev = None
    diffs = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride != 0:
            i += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            d = np.mean(np.abs(gray.astype(np.int16) - prev.astype(np.int16)))
            diffs.append((i, float(d)))
        prev = gray
        i += 1
    cap.release()
    diffs.sort(key=lambda x: x[1], reverse=True)
    return [idx for (idx, _) in diffs[:m]]
