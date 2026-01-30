#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 Dataset Preparation Script

Key improvements over V9/V10:
1. Diverse scene lengths (no forced splitting)
2. Relative position encoding for each frame
3. Adaptive stride based on scene length
4. Saves to separate directory to preserve original data
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.scene_detection import create_detector, available_detectors, Scene
from src.models.anime_clipiqa_v3 import create_anime_clipiqa
from utils.io import save_json, save_csv, frames_to_timecode, export_scene_previews


def log(msg: str):
    print(f"[V11 Prep] {msg}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def normalize_and_merge_scenes_v11(
    scenes: List[Scene],
    min_len_frames: int = 30,
    max_len_frames: int = 500,
    force_split: bool = False,
) -> List[Scene]:
    """
    V11: Allow diverse scene lengths.
    - min_len_frames: Merge very short scenes (< 30 frames)
    - max_len_frames: If force_split=True, split scenes longer than this.
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
    
    # Merge short scenes
    merged: List[Scene] = []
    for sc in norm:
        cur_len = sc.end_frame - sc.start_frame + 1
        if not merged:
            merged.append(sc)
            continue
        
        if cur_len < min_len_frames:
            # Merge into previous
            prev = merged[-1]
            merged[-1] = Scene(prev.start_frame, sc.end_frame)
        else:
            merged.append(sc)
    
    if not force_split:
        # Just log long scenes
        for sc in merged:
            length = sc.end_frame - sc.start_frame + 1
            if length > max_len_frames:
                log(f"  Long scene: {length} frames (not splitting)")
        return merged
        
    # Force split long scenes
    final_scenes: List[Scene] = []
    for sc in merged:
        length = sc.end_frame - sc.start_frame + 1
        if length > max_len_frames:
            # Split into chunks
            cur_start = sc.start_frame
            while cur_start <= sc.end_frame:
                # Ensure the last chunk isn't too tiny if possible, 
                # but simplistic splitting is usually requested for "fixed length".
                # We'll just split greedily by max_len_frames.
                cur_end = min(cur_start + max_len_frames - 1, sc.end_frame)
                
                # Check if this split results in a very small remainder?
                # For this ablation, strictly adhering to max_len is key.
                final_scenes.append(Scene(cur_start, cur_end))
                cur_start = cur_end + 1
        else:
            final_scenes.append(sc)
            
    return final_scenes


def adaptive_stride(scene_len: int) -> int:
    """Adaptive stride based on scene length for diverse sampling."""
    if scene_len < 100:
        return 3
    elif scene_len < 300:
        return 5
    else:
        return 7


def decode_scene_frames(
    video_path: str,
    start_frame: int,
    end_frame: int,
    stride: int,
    resize_to: Optional[Tuple[int, int]] = None
) -> Tuple[List[np.ndarray], List[int]]:
    """Decode frames from a specific scene with given stride."""
    assert cv2 is not None, "OpenCV required"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    
    frames = []
    frame_indices = []
    
    for fidx in range(start_frame, end_frame + 1, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        if resize_to:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        frames.append(frame)
        frame_indices.append(fidx)
    
    cap.release()
    return frames, frame_indices


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


class CLIPExtractor:
    """CLIP feature extractor with caching."""
    
    def __init__(self, device: str = "cuda"):
        import torch
        import clip
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        log("CLIP ViT-B/32 loaded")
    
    def __call__(self, frames: List[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image
        
        pil_list = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        batch = torch.stack([self.preprocess(p) for p in pil_list]).to(self.device)
        
        with torch.no_grad():
            feats = self.model.encode_image(batch).float().cpu().numpy()
        
        return l2_normalize(feats, axis=1)


def save_scene_v11(
    out_root: Path,
    video_stem: str,
    scene_id: int,
    frames: List[np.ndarray],
    feats: np.ndarray,
    rel_positions: np.ndarray,  # V11: relative positions
    anime_attrs: np.ndarray,    # V11: Anime-CLIP-IQA scores
    meta: Dict[str, Any],
    jpeg_quality: int = 85
):
    """Save scene with V11 format (includes relative positions)."""
    scene_dir = out_root / video_stem / f"scene_{scene_id:04d}"
    ensure_dir(scene_dir / "frames")
    
    # Save frames
    for i, im in enumerate(frames):
        cv2.imwrite(
            str(scene_dir / "frames" / f"{i:06d}.jpg"),
            im, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
    
    # Save features
    np.save(scene_dir / "feats.npy", feats)
    
    # V11: Save relative positions
    np.save(scene_dir / "rel_positions.npy", rel_positions)

    # V11: Save Anime-CLIP-IQA attributes
    np.save(scene_dir / "anime_attrs.npy", anime_attrs)
    
    # Save meta
    meta["version"] = "v11"
    with open(scene_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="V11 Dataset Preparation")
    
    # Inputs/outputs
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    
    # Scene detection
    parser.add_argument("--backend", type=str, default="transnetv2",
                        choices=available_detectors())
    parser.add_argument("--model_dir", type=str, default="./src/models/TransNetV2")
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument("--min_scene_len", type=int, default=30)
    parser.add_argument("--max_scene_len", type=int, default=500)
    parser.add_argument("--force_split", action="store_true", help="Force split scenes larger than max_scene_len")
    
    # Frame processing
    parser.add_argument("--resize_w", type=int, default=0)
    parser.add_argument("--resize_h", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    
    args = parser.parse_args()
    
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    
    resize_to = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize_to = (args.resize_w, args.resize_h)
    
    extractor = CLIPExtractor(device=args.device)
    
    # Initialize Anime-CLIP-IQA model
    log("Initializing Anime-CLIP-IQA (Standard/V3)...")
    iqa_model = create_anime_clipiqa(device=args.device)
    
    det_kwargs = {
        "model_dir": args.model_dir,
        "prob_threshold": args.prob_threshold,
        "device": args.device,
    }
    
    video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.*")))
    video_paths = [v for v in video_paths if v.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    log(f"Found {len(video_paths)} videos")
    
    total_scenes = 0
    
    for vp in video_paths:
        video_path = Path(vp)
        video_stem = video_path.stem
        log(f"Processing: {video_stem}")
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        # Detect scenes
        detector = create_detector(args.backend, **det_kwargs)
        try:
            scenes_raw = detector.detect(str(video_path))
        finally:
            detector.close()
        
        if not scenes_raw:
            log(f"  No scenes detected, using whole video")
            scenes_raw = [Scene(0, total_frames - 1)]
        
        # V11: Diverse scene handling
        scenes = normalize_and_merge_scenes_v11(
            scenes_raw,
            min_len_frames=args.min_scene_len,
            max_len_frames=args.max_scene_len,
            force_split=args.force_split
        )
        
        log(f"  {len(scenes)} scenes after processing")
        
        # Save scenes list
        video_out_dir = out_root / video_stem
        ensure_dir(video_out_dir)
        
        scene_rows = []
        for i, sc in enumerate(scenes):
            s, e = int(sc.start_frame), int(sc.end_frame)
            scene_rows.append({
                "scene_id": i,
                "start_frame": s,
                "end_frame": e,
                "duration_frames": e - s + 1,
            })
        save_json(scene_rows, str(video_out_dir / "scenes.json"))
        
        # Process each scene
        for sid, sc in enumerate(scenes):
            s, e = int(sc.start_frame), int(sc.end_frame)
            scene_len = e - s + 1
            
            # V11: Adaptive stride
            stride = adaptive_stride(scene_len)
            
            # Decode frames
            frames, frame_indices = decode_scene_frames(
                str(video_path), s, e, stride, resize_to
            )
            
            if len(frames) < 2:
                continue
            
            # Extract features
            feats = extractor(frames)
            
            # V11: Compute relative positions (0.0 to 1.0)
            rel_positions = np.array([
                (fidx - s) / max(1, e - s)
                for fidx in frame_indices
            ], dtype=np.float32)

            # V11: Compute Anime-CLIP-IQA scores
            # Use legacy format (N, 6) for compatibility with training
            anime_attrs = iqa_model.get_legacy_format_scores(frames)
            
            meta = {
                "video": str(video_path),
                "video_stem": video_stem,
                "scene_id": sid,
                "start_frame": s,
                "end_frame": e,
                "scene_length": scene_len,
                "T": len(frames),
                "D": feats.shape[1],
                "stride": stride,
                "fps": fps,
                "frame_indices": frame_indices,
            }
            
            save_scene_v11(
                out_root, video_stem, sid, frames, feats, rel_positions, anime_attrs, meta,
                jpeg_quality=args.jpeg_quality
            )
            
            log(f"  scene {sid:04d}: T={len(frames)}, len={scene_len}, stride={stride}")
            total_scenes += 1
    
    log(f"Done! Total {total_scenes} scenes saved to {out_root}")


if __name__ == "__main__":
    main()


"""
Usage:
python -m scripts.prepare_rl_dataset_v11 \
    --video_dir data/samples/Sakuga \
    --out_dir data/sakuga_dataset_v11 \
    --backend transnetv2 \
    --model_dir src/models/TransNetV2 \
    --min_scene_len 30 \
    --max_scene_len 500 \
    --device cuda

python -m scripts.precompute_script.prepare_rl_dataset_v11 --video_dir data/samples/Sakuga --out_dir data/sakuga_dataset_v11_new --backend transnetv2 --model_dir src/models/TransNetV2 --min_scene_len 30 --max_scene_len 500 --device cuda

python -m scripts.precompute_script.prepare_rl_dataset_v11 --video_dir data/samples/Sakuga_test --out_dir data/sakuga_dataset_v11_new_test --backend transnetv2 --model_dir src/models/TransNetV2 --min_scene_len 30 --max_scene_len 500 --device cuda
"""
