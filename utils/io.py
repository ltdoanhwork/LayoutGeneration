import os
import cv2
import json
import csv
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from src.scene_detection import (
    Scene,
)
from src.keyframe.medoid_selector import (
    Keyframe as KF,
)
# ------------------------------
# Basic I/O helpers
# ------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_video_basic_info(video_path: str) -> Tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return total_frames, fps

def frames_to_timecode(frame_idx: int, fps: float) -> str:
    t = frame_idx / max(1.0, fps)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def export_scene_previews(
    video_path: str,
    scenes: List[Scene],
    out_dir: str,
    which: str = "mid",      # "start" | "mid" | "end"
    jpeg_quality: int = 95,
) -> None:
    """Export one preview per scene at start/mid/end index."""
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    for i, sc in enumerate(tqdm(scenes, desc="Export scene previews")):
        if which == "start":
            frame_idx = sc.start_frame
        elif which == "end":
            frame_idx = sc.end_frame
        else:
            frame_idx = (sc.start_frame + sc.end_frame) // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Try previous frame as a fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
            ok, frame = cap.read()
        if not ok or frame is None:
            continue

        fn = f"scene_{i:04d}_{which}_f{frame_idx:08d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fn), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

    cap.release()


def export_keyframe_images(
    video_path: str,
    keyframes: List[KF],
    out_dir: str,
    jpeg_quality: int = 95,
) -> None:
    """Export JPG image for each selected keyframe."""
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    for kf in tqdm(keyframes, desc="Export keyframe images"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf.frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        fn = f"scene_{kf.scene_id:04d}_frame_{kf.frame_idx:08d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fn), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

    cap.release()