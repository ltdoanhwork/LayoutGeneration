#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset_v2.py
Builds a scene-level dataset from a single video or a directory of videos,
using YOUR scene detection backend (create_detector/normalize_and_merge_scenes/...).

Per scene it saves:
  <out_root>/<video_stem>/scene_xxxx/
      frames/%06d.jpg
      feats.npy        # (T, D) L2-normalized features
      flow.npy         # (T,) optional TV-L1 forward flow magnitude
      meta.json        # fps/stride, resize, scene start/end, etc.

It also writes:
  <out_root>/<video_stem>/scenes.json
  <out_root>/<video_stem>/scenes.csv
and (optional) scene preview JPEGs.

Dependencies:
  - OpenCV (cv2)    : decoding, HOG, flow, hist, pyrDown
  - numpy
  - (optional) torch + clip (if you want CLIP features)

NOTE: This script expects your utility functions to be importable:
  create_detector, normalize_and_merge_scenes, frames_to_timecode,
  save_json, save_csv, export_scene_previews,
  create_metric (only if you actually need to build a perceptual metric here).
Adjust the import paths to your project layout.
"""

import os
import json
import glob
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from src.scene_detection import (
    create_detector,
    available_detectors,
    Scene,
)
from utils.io import *

# ======================
# Generic helper routines
# ======================
def log(msg: str):
    print(f"[prepare] {msg}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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
# ====================
# Video → frame sampler
# ====================
def decode_video_frames(video_path: str,
                        fps: Optional[float],
                        stride: int,
                        resize_to: Optional[Tuple[int, int]] = None
                        ) -> Tuple[List[np.ndarray], float, int]:
    """
    Decode frames from a video.
      - If fps > 0: sample ~that fps based on original fps (by keeping every Kth frame).
      - Else: sample by stride directly on frames.
      - Optional resize (W,H) after decode.

    Returns (frames, orig_fps, total_frames_read_before_subsample)
    """
    assert cv2 is not None, "OpenCV is required."
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    keep_every = 1
    if fps is not None and fps > 0 and orig_fps > 0:
        keep_every = max(1, int(round(orig_fps / fps)))
    keep_every = max(keep_every, stride)

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % keep_every == 0:
            if resize_to is not None and resize_to[0] > 0 and resize_to[1] > 0:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, float(orig_fps), total

# ==================
# Feature extraction
# ==================
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

class FeatureExtractor:
    """
    Pluggable extractor:
      - "auto": try CLIP (ViT-B/32), else fallback to "classic"
      - "clip": force CLIP
      - "classic": HSV hist + pooled HOG (OpenCV-only)
    """
    def __init__(self, kind: str = "auto", device: str = "cpu"):
        self.kind = kind
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        if kind in ("auto", "clip"):
            try:
                import torch
                import clip  # openai/CLIP
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
                self.clip_model.eval()
                self.kind = "clip"
                print("[feat] Using CLIP ViT-B/32")
            except Exception as e:
                print(f"[feat] CLIP not available ({e}); fallback to classic")
                self.kind = "classic"

    def _extract_clip(self, frames: List[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image
        pil_list = []
        for img in frames:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_list.append(Image.fromarray(rgb))
        batch = torch.stack([self.clip_preprocess(p) for p in pil_list], dim=0).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_image(batch).float().cpu().numpy()
        return l2_normalize(feats, axis=1)

    def _extract_classic(self, frames: List[np.ndarray]) -> np.ndarray:
        hog = cv2.HOGDescriptor()
        feats = []
        for img in frames:
            # HSV histogram (H,S) 32x32
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            h = cv2.normalize(h, None).flatten()

            # HOG on downscaled gray
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (128, 128))
            hvec = hog.compute(g).reshape(-1)
            bins = 64
            pool = np.mean(np.array_split(hvec, bins), axis=1)

            feat = np.concatenate([h, pool], axis=0).astype(np.float32)
            feats.append(feat)
        feats = np.stack(feats, axis=0)
        return l2_normalize(feats, axis=1)

    def __call__(self, frames: List[np.ndarray]) -> np.ndarray:
        if self.kind == "clip":
            return self._extract_clip(frames)
        else:
            return self._extract_classic(frames)

# ================
# Optical Flow - DEPRECATED
# ================
# NOTE: Optical flow computation has been moved to RAFT precomputation.
# Use scripts/precompute_raft_motion.py instead.
# This function is kept for backward compatibility but not used in the pipeline.

# =========
# Save scene
# =========
def save_scene(out_root: Path,
               video_stem: str,
               scene_id: int,
               frames: List[np.ndarray],
               feats: np.ndarray,
               meta: Dict[str, Any],
               jpeg_quality: int = 85):
    """Save scene data (frames, features, metadata). Flow is deprecated."""
    scene_dir = out_root / video_stem / f"scene_{scene_id:04d}"
    ensure_dir(scene_dir / "frames")
    # Save frames
    for i, im in enumerate(frames):
        cv2.imwrite(str(scene_dir / "frames" / f"{i:06d}.jpg"),
                    im, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    # Save feats
    np.save(scene_dir / "feats.npy", feats)
    # Save meta
    with open(scene_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

# =====
# Main
# =====
def main():
    scene_choices = available_detectors()
    ap = argparse.ArgumentParser()
    # Inputs
    ap.add_argument("--video_dir", type=str, required=True,
                    help="Directory of input videos (we will process all).")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output dataset root.")
    # Frame sampling
    ap.add_argument("--fps", type=float, default=20.0,
                    help="Target FPS; if <=0, we use stride only.")
    ap.add_argument("--stride", type=int, default=1)
    # Resize
    ap.add_argument("--resize_w", type=int, default=0)
    ap.add_argument("--resize_h", type=int, default=0)
    # Scene detection (YOUR backend)
    # ap.add_argument("--backend", type=str, default = "pyscenedetect", required=True,
    #                 help="Your scene detection backend name (for create_detector).")
    ap.add_argument("--min_scene_len", type=int, default=48,
                    help="Min scene length after normalize/merge (in frames of the SAMPLED stream).")
    
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
    # Export preview
    ap.add_argument("--export_preview", action="store_true")
    ap.add_argument("--preview_which", type=str, default="middle",
                    choices=["first","middle","last","all"])
    ap.add_argument("--preview_jpeg_quality", type=int, default=85)
    # Feature extractor
    ap.add_argument("--extractor", type=str, default="auto",
                    choices=["auto","clip","classic"])
    ap.add_argument("--device", type=str, default="cuda:0")
    # JPEG saving
    ap.add_argument("--save_jpeg_quality", type=int, default=85)
    # NOTE: --do_flow has been removed. Use scripts/precompute_raft_motion.py for motion features.
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    # Prepare resize tuple
    resize_to: Optional[Tuple[int, int]] = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize_to = (args.resize_w, args.resize_h)

    # Feature extractor
    extractor = FeatureExtractor(kind=args.extractor, device=args.device)

    # Detector kwargs
    det_kwargs: Dict[str, Any] = {
        "threshold": args.threshold,
        "model_dir": args.model_dir,
        "weights_path": args.weights_path,
        "prob_threshold": args.prob_threshold,
        "device": args.scene_device,
    }

    # Process each video
    video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.*")))
    if len(video_paths) == 0:
        log("No videos found.")
        return

    for vp in video_paths:
        video_path = Path(vp)
        video_stem = video_path.stem
        log(f"Processing video: {video_stem}")

        # Decode frames once (sampled)
        frames, orig_fps, total_frames_read = decode_video_frames(
            str(video_path),
            fps=(args.fps if args.fps and args.fps > 0 else None),
            stride=args.stride,
            resize_to=resize_to
        )
        if len(frames) == 0:
            log("  -> No frames after sampling. Skipping.")
            continue

        # Use your scene detector on the ORIGINAL VIDEO PATH
        # (Most detectors want the file path; yours returns Scene objects with frame indices.)
        detector = create_detector(args.backend, **det_kwargs)
        try:
            scenes_raw = detector.detect(str(video_path))
        finally:
            detector.close()

        if not scenes_raw:
            print("[WARN] No scenes detected by backend. Fallback to the whole video as one scene.")
            # In your interface Scene(start_frame, end_frame) is inclusive,
            # and you're saving start/end relative to the *original* frame space.
            # Because we already sub-sampled frames, we treat the whole sampled stream as one scene:
            # start=0, end=len(frames)-1 (sampled index space).
            # If your downstream expects original indices, adapt here accordingly.
            class _Scene:
                def __init__(self, s, e): self.start_frame, self.end_frame = s, e
            scenes_raw = [_Scene(0, max(0, len(frames) - 1))]

        # Post-process scenes using your normalization/merging util.
        # IMPORTANT: If your normalize function expects original-frame indices,
        # but we operate on the sampled index space, ensure consistency.
        scenes = normalize_and_merge_scenes(scenes_raw, min_len_frames=args.min_scene_len)

        # Save scenes.json/csv at video level
        video_out_dir = out_root / video_stem
        ensure_dir(video_out_dir)
        fps_used = args.fps if args.fps and args.fps > 0 else orig_fps / max(1, args.stride)

        scene_rows: List[Dict[str, Any]] = []
        for i, sc in enumerate(scenes):
            s = int(sc.start_frame)
            e = int(sc.end_frame)
            dur = max(0, e - s + 1)
            scene_rows.append({
                "scene_id": i,
                "start_frame": s,
                "end_frame": e,
                "start_time": frames_to_timecode(s, fps_used),
                "end_time": frames_to_timecode(e, fps_used),
                "duration_frames": dur,
                "duration_seconds": round(dur / fps_used, 3) if fps_used > 0 else 0.0,
            })
        save_json(scene_rows, str(video_out_dir / "scenes.json"))
        save_csv(scene_rows, str(video_out_dir / "scenes.csv"))

        # Optional preview export with your helper
        if args.export_preview:
            preview_dir = str(video_out_dir / "scene_previews")
            export_scene_previews(
                video_path=str(video_path),
                scenes=scenes,
                out_dir=preview_dir,
                which=args.preview_which,
                jpeg_quality=args.preview_jpeg_quality,
            )

        # For each scene, slice frames by the *sampled* indices and write per-scene package
        for sid, sc in enumerate(scenes):
            s = int(sc.start_frame)
            e = int(sc.end_frame)
            # Clamp to sampled stream bounds
            s = max(0, min(s, len(frames) - 1))
            e = max(0, min(e, len(frames) - 1))
            if e < s:
                continue
            sub = frames[s:e + 1]
            if len(sub) < 2:
                continue

            # Features
            feats = extractor(sub)

            # NOTE: Flow computation removed. Use scripts/precompute_raft_motion.py instead.

            meta = dict(
                video=str(video_path),
                video_stem=video_stem,
                scene_id=int(sid),
                sampled_start=int(s),
                sampled_end=int(e),
                T=len(sub),
                D=int(feats.shape[1]),
                fps_used=float(fps_used),
                fps_arg=float(args.fps),
                stride=int(args.stride),
                resize_w=int(args.resize_w),
                resize_h=int(args.resize_h),
                extractor=args.extractor,
            )
            save_scene(out_root, video_stem, sid, sub, feats, meta, jpeg_quality=args.save_jpeg_quality)
            log(f"  scene {sid:04d}: T={len(sub)} D={feats.shape[1]} saved.")

    log("All done.")

if __name__ == "__main__":
    main()

"""
python -m src.pipeline.prepare_rl_dataset \
  --video_dir /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga \
  --out_dir data/sakuga_dataset_100_samples \
  --backend transnetv2  \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --fps 6 --stride 1 \
  --resize_w 0 --resize_h 0 \
  --min_scene_len 48 \
  --extractor auto --device cuda \
  --export_preview --preview_which middle \
  --threshold 27.0

NOTE: For motion features, use scripts/precompute_raft_motion.py after running this script.
"""