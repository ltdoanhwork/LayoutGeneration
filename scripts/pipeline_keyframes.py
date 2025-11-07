#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Scene→Keyframe Pipeline
- Scene detection via pluggable backends (registry): pyscenedetect, transnetv2, ...
- Keyframe selection via pluggable distance metrics (registry): lpips, dists, ...
- Outputs:
    * scenes.json / scenes.csv
    * keyframes.csv
    * keyframes/ (exported JPGs)
    * scene_previews/ (optional mid/start/end frame of each scene)
All code comments are in English (per user requirement).
"""

from __future__ import annotations
import os
import csv
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# --- Registries (auto-register built-ins via package __init__) ---
from src.scene_detection import (
    create_detector,
    available_detectors,
    Scene,
)
from src.distance_selector import (
    create_metric,
    available_metrics,
)
from src.keyframe.medoid_selector import (
    MedoidSelector,
    Keyframe as KF,
)

from src.keyframe.random_selector import RandomSelector

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


# ------------------------------
# Scenes post-processing
# ------------------------------
def normalize_and_merge_scenes(
    scenes: List[Scene],
    min_len_frames: int = 0,
) -> List[Scene]:
    """
    Normalize (ensure start<=end), sort by start, and optionally merge short scenes
    into the previous one if below `min_len_frames`.
    """
    if not scenes:
        return []

    # Normalize and sort
    norm: List[Scene] = []
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
            prev = merged[-1]
            if sc.start_frame <= prev.end_frame + 1:
                # Contiguous → extend previous
                merged[-1] = Scene(prev.start_frame, max(prev.end_frame, sc.end_frame))
            else:
                # Non-contiguous but still merge into previous by extending end
                merged[-1] = Scene(prev.start_frame, sc.end_frame)
    return merged


# ------------------------------
# Argparse
# ------------------------------
def build_argparser() -> argparse.ArgumentParser:
    # Query available backends dynamically (packages import will auto-register)
    scene_choices = available_detectors()
    metric_choices = available_metrics()

    ap = argparse.ArgumentParser(
        description="Scene→Keyframe pipeline using pluggable scene detectors and distance metrics."
    )
    ap.add_argument("--video", type=str, required=True, help="Input video path.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")

    # Scene detection backend + params
    ap.add_argument("--backend", type=str, default="pyscenedetect", choices=scene_choices,
                    help="Scene detection backend.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="[pyscenedetect] ContentDetector threshold (default 27.0).")
    ap.add_argument("--model_dir", type=str, default=None,
                    help="[transnetv2] Directory containing weights/, or pass --weights_path.")
    ap.add_argument("--weights_path", type=str, default=None,
                    help="[transnetv2] Direct path to .pth weights (overrides model_dir).")
    ap.add_argument("--prob_threshold", type=float, default=None,
                    help="[transnetv2] Boundary probability threshold (default 0.5).")
    ap.add_argument("--scene_device", type=str, default=None,
                    help="[transnetv2] Device for model ('cuda'/'cpu').")

    # Scenes post-process & preview
    ap.add_argument("--min_scene_len", type=int, default=0,
                    help="Minimum scene length in frames for post-merge (0 = disabled).")
    ap.add_argument("--export_preview", action="store_true",
                    help="Export one preview image per scene.")
    ap.add_argument("--preview_which", type=str, default="mid",
                    choices=["start", "mid", "end"], help="Which frame to export as preview.")
    ap.add_argument("--preview_jpeg_quality", type=int, default=95)

    # Distance metric + selection params
    ap.add_argument("--distance_backend", type=str, default="lpips", choices=metric_choices,
                    help="Distance metric backend.")
    ap.add_argument("--distance_device", type=str, default=None,
                    help="Device for metric ('cuda'/'cpu').")
    ap.add_argument("--lpips_net", type=str, default="alex",
                    help="[lpips] Backbone: 'alex'|'vgg'|'squeeze'.")
    ap.add_argument("--dists_as_distance", type=int, default=1,
                    help="[dists] Use raw DISTS as distance (1) or negate as similarity (0).")

    ap.add_argument("--sample_stride", type=int, default=10,
                    help="Sample every N frames within a scene.")
    ap.add_argument("--max_frames_per_scene", type=int, default=30,
                    help="Cap sampled frames per scene (controls O(N^2) cost).")
    ap.add_argument("--keyframes_per_scene", type=int, default=1,
                    help="How many keyframes to pick per scene.")
    ap.add_argument("--nms_radius", type=int, default=3,
                    help="Greedy index-NMS radius when selecting multiple keyframes per scene.")
    ap.add_argument("--resize_w", type=int, default=320,
                    help="Resize width for distance computation (<=0 to disable).")
    ap.add_argument("--resize_h", type=int, default=180,
                    help="Resize height for distance computation (<=0 to disable).")
    ap.add_argument("--batch_pairs", type=int, default=16,
                    help="Mini-batch size of (i,j) pairs when computing pairwise distances.")

    # Keyframe selection
    ap.add_argument("--keyframe_selector", type=str, default="medoid", choices=["medoid", "random"],
                    help="Keyframe selection strategy.")
    ap.add_argument("--random_seed", type=int, default=None,
                    help="Random seed for reproducibility (only used with random selector).")

    # Keyframe export
    ap.add_argument("--key_jpeg_quality", type=int, default=95,
                    help="JPEG quality for exported keyframe images.")
    return ap


# ------------------------------
# Main
# ------------------------------
def main():
    args = build_argparser().parse_args()
    args.out_dir = args.out_dir + f"_{args.video.split('/')[-1].split('.')[0]}"

    # Prepare output folders
    ensure_dir(args.out_dir)
    key_dir = os.path.join(args.out_dir, "keyframes")
    ensure_dir(key_dir)
    preview_dir = os.path.join(args.out_dir, "scene_previews")

    # Read basic video info
    total_frames, fps = read_video_basic_info(args.video)

    # Build scene-detector kwargs (only pass values that are actually set)
    det_kwargs: Dict[str, Any] = {
        "threshold": args.threshold,
        "model_dir": args.model_dir,
        "weights_path": args.weights_path,
        "prob_threshold": args.prob_threshold,
        "device": args.scene_device,
    }
    det_kwargs = {k: v for k, v in det_kwargs.items() if v not in (None, "", [])}

    # Run scene detection
    detector = create_detector(args.backend, **det_kwargs)
    scenes_raw: List[Scene] = detector.detect(args.video)
    detector.close()

    if not scenes_raw:
        print("[WARN] No scenes detected by backend. Fallback to the whole video as one scene.")
        scenes_raw = [Scene(0, max(0, total_frames - 1))]

    # Post-process scenes (optional)
    scenes = normalize_and_merge_scenes(scenes_raw, min_len_frames=args.min_scene_len)

    # Save scenes to JSON/CSV
    scene_rows: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenes):
        s, e = int(sc.start_frame), int(sc.end_frame)
        dur_frames = max(0, e - s + 1)
        scene_rows.append({
            "scene_id": i,
            "start_frame": s,
            "end_frame": e,
            "start_time": frames_to_timecode(s, fps),
            "end_time": frames_to_timecode(e, fps),
            "duration_frames": dur_frames,
            "duration_seconds": round(dur_frames / fps, 3) if fps > 0 else 0.0,
        })

    save_json(scene_rows, os.path.join(args.out_dir, "scenes.json"))
    save_csv(scene_rows, os.path.join(args.out_dir, "scenes.csv"))

    if args.export_preview:
        export_scene_previews(
            video_path=args.video,
            scenes=scenes,
            out_dir=preview_dir,
            which=args.preview_which,
            jpeg_quality=args.preview_jpeg_quality,
        )

    # Build distance metric
    dist_kwargs: Dict[str, Any] = {"device": args.distance_device}
    if args.distance_backend == "lpips":
        dist_kwargs.update({"net": args.lpips_net})
    elif args.distance_backend == "dists":
        dist_kwargs.update({"as_distance": bool(args.dists_as_distance)})

    metric = create_metric(args.distance_backend, **dist_kwargs)

    if args.keyframe_selector == "random":
        selector = RandomSelector(seed=args.random_seed)
    else:
        selector = MedoidSelector(metric=metric)

    # Prepare resize
    resize_to: Optional[Tuple[int, int]]
    if args.resize_w > 0 and args.resize_h > 0:
        resize_to = (args.resize_w, args.resize_h)
    else:
        resize_to = None

    # Select keyframes per scene
    keyframes: List[KF] = []
    for sid, sc in enumerate(tqdm(scenes, desc="Selecting keyframes")):
        kfs = selector.select_for_scene(
            video_path=args.video,
            scene_range=(sc.start_frame, sc.end_frame),
            sample_stride=args.sample_stride,
            max_frames_per_scene=args.max_frames_per_scene,
            keyframes_per_scene=args.keyframes_per_scene,
            nms_radius=args.nms_radius,
            resize_to=resize_to,
            scene_id=sid,
            batch_pairs=args.batch_pairs,
        )
        keyframes.extend(kfs)

    # Save keyframes CSV
    key_rows: List[Dict[str, Any]] = []
    for kf in keyframes:
        key_rows.append({
            "scene_id": kf.scene_id,
            "frame_idx": kf.frame_idx,
            "time": frames_to_timecode(kf.frame_idx, fps),
            "score": round(kf.score, 6),
            "distance_backend": args.distance_backend,
        })
    save_csv(key_rows, os.path.join(args.out_dir, "keyframes.csv"))

    # Export keyframe images
    export_keyframe_images(
        video_path=args.video,
        keyframes=keyframes,
        out_dir=key_dir,
        jpeg_quality=args.key_jpeg_quality,
    )

    # Summary
    print(f"[DONE] Scenes: {len(scenes)} | Keyframes: {len(keyframes)}")
    print(f"  • Scenes JSON : {os.path.join(args.out_dir, 'scenes.json')}")
    print(f"  • Scenes CSV  : {os.path.join(args.out_dir, 'scenes.csv')}")
    print(f"  • Keyframes CSV: {os.path.join(args.out_dir, 'keyframes.csv')}")
    if args.export_preview:
        print(f"  • Scene previews: {preview_dir}")
    print(f"  • Keyframe images: {key_dir}")


if __name__ == "__main__":
    main()
    
"""
# 1) PySceneDetect + LPIPS(Alex)
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 3 --max_frames_per_scene 100 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_psd_lpips \
  --export_preview

# 1) PySceneDetect + DISTS(Alex)
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend dists --lpips_net alex \
  --sample_stride 3 --max_frames_per_scene 100 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_psd_dists \
  --export_preview

# 2) TransNetV2 (PyTorch) + DISTS
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend transnetv2  \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --distance_backend dists --dists_as_distance 1 \
  --sample_stride 8 --max_frames_per_scene 40 \
  --keyframes_per_scene 2 --nms_radius 4 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_tv2_dists

python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 3 --max_frames_per_scene 100 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_psd_lpips \
  --export_preview \
  --keyframe_selector random --random_seed 42
"""

## transnet v2 + lpips 
## transnet v2 + dtits
##
""" 
# 1) TransNetV2 (PyTorch) + LPIPS(Alex)
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend transnetv2  \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 8 --max_frames_per_scene 40 \
  --keyframes_per_scene 2 --nms_radius 4 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_tv2_lpips

# 2) TransNetV2 (PyTorch) + DISTS
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend transnetv2  \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --distance_backend dists --dists_as_distance 1 \
  --sample_stride 8 --max_frames_per_scene 40 \
  --keyframes_per_scene 2 --nms_radius 4 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_tv2_dists """
