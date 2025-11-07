#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scene Detection Pipeline (CLI)
- Uses registry to select backend: pyscenedetect / transnetv2 / ...
- Exports scenes.json & scenes.c               hel                            help="[transnetv2] Boundary probability threshold (default 0.5 if not provided).") help="[transnetv2] SavedModel path for TransNetV2.") help="[pyscenedetect] ContentDetector threshold (default 27.0 if not provided).")elp="Minimum number of frames for a scene after detection (0 = disabled).")="Choose frame for preview export.")help="Export preview image (mid frame) for each scene.")v
- Optional export of preview image for each scene (mid-frame)
"""

from __future__ import annotations
import os
import csv
import json
import argparse
from dataclasses import asdict
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Import modules standardized according to the available interface/registry
# Requires running from project root for Python to find the src/ package
from src.scene_detection import create_detector, available_detectors, Scene


# ------------------------------
# Helper I/O
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


def save_scenes_csv(scenes: List[Dict[str, Any]], path: str) -> None:
    if not scenes:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")  # empty
        return
    fieldnames = list(scenes[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scenes:
            writer.writerow(row)


def export_scene_preview_images(
    video_path: str,
    scenes: List[Scene],
    out_dir: str,
    which: str = "mid",  # "mid" | "start" | "end"
    jpeg_quality: int = 95,
) -> None:
    """
    Export preview image for each scene. which:
      - "start": start_frame
      - "end":   end_frame
      - "mid":   middle frame ( (s+e)//2 )
    """
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    for i, sc in enumerate(tqdm(scenes, desc="Export previews")):
        if which == "start":
            frame_idx = sc.start_frame
        elif which == "end":
            frame_idx = sc.end_frame
        else:
            frame_idx = (sc.start_frame + sc.end_frame) // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Try to go back 1 frame if failed
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
            ok, frame = cap.read()
        if not ok or frame is None:
            continue

        out_name = f"scene_{i:04d}_{which}_f{frame_idx:08d}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

    cap.release()


# ------------------------------
# Postprocess Scenes (optional)
# ------------------------------
def normalize_and_merge_scenes(
    scenes: List[Scene],
    min_len_frames: int = 0,
) -> List[Scene]:
    """
    - Ensures (start <= end), sorts by start.
    - Optional merging of short scenes: if min_len_frames > 0,
      will merge short scenes into the previous adjacent scene (if exists), or next.
    """
    if not scenes:
        return []

    # chuẩn hoá
    norm = []
    for s in scenes:
        a, b = int(s.start_frame), int(s.end_frame)
        if b < a:
            a, b = b, a
        norm.append(Scene(a, b))
    norm.sort(key=lambda x: (x.start_frame, x.end_frame))

    if min_len_frames <= 0:
        return norm

    merged: List[Scene] = []
    for sc in norm:
        if not merged:
            merged.append(sc)
            continue
        cur_len = sc.end_frame - sc.start_frame + 1
        if cur_len >= min_len_frames:
            merged.append(sc)
        else:
            # too short: merge into previous if touching/overlapping/nearby,
            # otherwise merge into the previous element itself.
            prev = merged[-1]
            if sc.start_frame <= prev.end_frame + 1:
                # merge contiguous scenes
                merged[-1] = Scene(prev.start_frame, max(prev.end_frame, sc.end_frame))
            else:
                # not adjacent, still force merge into prev (extend end)
                merged[-1] = Scene(prev.start_frame, sc.end_frame)
    return merged


# ------------------------------
# Main
# ------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Scene Detection Pipeline using pluggable backends."
    )
    ap.add_argument("--video", type=str, required=True, help="Input video path.")
    ap.add_argument(
        "--backend",
        type=str,
        default="pyscenedetect",
        choices=available_detectors(),
        help="Backend scene detection.",
    )

    # Common parameters for export
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument(
        "--export_preview",
        action="store_true",
        help="Xuất ảnh preview (mid frame) cho mỗi scene.",
    )
    ap.add_argument(
        "--preview_which",
        type=str,
        default="mid",
        choices=["start", "mid", "end"],
        help="Chọn frame để export preview.",
    )
    ap.add_argument(
        "--jpeg_quality", type=int, default=95, help="JPEG quality cho preview export."
    )

    # Options for scene length post-processing
    ap.add_argument(
        "--min_scene_len",
        type=int,
        default=0,
        help="Số frame tối thiểu cho 1 scene sau detect (0 = không dùng).",
    )

    # Backend-specific parameters (passed via kwargs — backend will use what it needs)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="[pyscenedetect] ContentDetector threshold (mặc định 27.0 nếu không truyền).",
    )
    ap.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="[transnetv2] Đường dẫn SavedModel cho TransNetV2.",
    )
    ap.add_argument(
        "--prob_threshold",
        type=float,
        default=None,
        help="[transnetv2] Ngưỡng boundary probability (mặc định 0.5 nếu không truyền).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None
    )
    return ap


def main():
    args = build_argparser().parse_args()

    # Prepare output
    ensure_dir(args.out_dir)
    previews_dir = os.path.join(args.out_dir, "scene_previews")
    if args.export_preview:
        ensure_dir(previews_dir)

    # Read video info
    total_frames, fps = read_video_basic_info(args.video)

    # Create detector from registry, only pass kwargs with actual values
    detector_kwargs: Dict[str, Any] = {
        "threshold": args.threshold,
        "model_dir": args.model_dir,
        "prob_threshold": args.prob_threshold,
    }
    detector_kwargs = {k: v for k, v in detector_kwargs.items() if v not in (None, "", [])}

    detector = create_detector(args.backend, **detector_kwargs)

    # Detect scenes
    scenes_raw: List[Scene] = detector.detect(args.video)
    detector.close()

    # Fallback if detector returns nothing
    if not scenes_raw:
        print("[WARN] No scenes detected. Fallback to whole video.")
        scenes_raw = [Scene(0, max(0, total_frames - 1))]

    # Length post-processing (optional)
    scenes_final = normalize_and_merge_scenes(scenes_raw, min_len_frames=args.min_scene_len)

    # Prepare data for saving
    scene_rows: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenes_final):
        s, e = int(sc.start_frame), int(sc.end_frame)
        dur_frames = max(0, e - s + 1)
        dur_seconds = dur_frames / fps if fps > 0 else 0.0
        scene_rows.append(
            {
                "scene_id": i,
                "start_frame": s,
                "end_frame": e,
                "start_time": frames_to_timecode(s, fps),
                "end_time": frames_to_timecode(e, fps),
                "duration_frames": dur_frames,
                "duration_seconds": round(dur_seconds, 3),
            }
        )

    # Lưu JSON & CSV
    save_json(scene_rows, os.path.join(args.out_dir, "scenes.json"))
    save_scenes_csv(scene_rows, os.path.join(args.out_dir, "scenes.csv"))

    # Export preview if needed
    if args.export_preview:
        export_scene_preview_images(
            video_path=args.video,
            scenes=scenes_final,
            out_dir=previews_dir,
            which=args.preview_which,
            jpeg_quality=args.jpeg_quality,
        )

    # Summary log
    print(f"[DONE] Detected {len(scenes_final)} scenes.")
    print(f"  • JSON: {os.path.join(args.out_dir, 'scenes.json')}")
    print(f"  • CSV : {os.path.join(args.out_dir, 'scenes.csv')}")
    if args.export_preview:
        print(f"  • Previews: {previews_dir}")


if __name__ == "__main__":
    main()


"""
Example usage:
# PySceneDetect (backend default threshold is 27.0 if not provided)
python scene_detection_pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend pyscenedetect \
  --out_dir outputs/run_psd \
  --export_preview --preview_which mid

# TransNetV2 (requires SavedModel)
python scene_detection_pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend transnetv2 \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --out_dir outputs/run_tv2 \
  --export_preview --preview_which mid
"""