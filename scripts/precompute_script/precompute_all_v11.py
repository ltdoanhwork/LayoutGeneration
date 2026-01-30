#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 All-in-One Precomputation Script

Runs all preprocessing steps in sequence:
1. Scene detection + CLIP features + relative positions
2. Anime attributes (multi-prompt CLIP)
3. (Optional) VLM quality scores

Usage:
    python -m scripts.precompute_all_v11 \
        --video_dir data/samples/Sakuga \
        --out_dir data/sakuga_dataset_v11 \
        --device cuda
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

import torch
import clip
from PIL import Image

from src.scene_detection import create_detector, available_detectors, Scene
from utils.io import save_json, save_csv


def log(msg: str):
    print(f"[V11 Precompute] {msg}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STEP 1: Scene Detection + CLIP Features + Relative Positions
# ============================================================================

def normalize_and_merge_scenes(
    scenes: List[Scene],
    min_len_frames: int = 30,
) -> List[Scene]:
    """Normalize and merge short scenes."""
    if not scenes:
        return []
    
    norm = []
    for s in scenes:
        a, b = int(s.start_frame), int(s.end_frame)
        if b < a:
            a, b = b, a
        norm.append(Scene(a, b))
    norm.sort(key=lambda x: x.start_frame)
    
    merged = []
    for sc in norm:
        cur_len = sc.end_frame - sc.start_frame + 1
        if not merged:
            merged.append(sc)
        elif cur_len < min_len_frames:
            prev = merged[-1]
            merged[-1] = Scene(prev.start_frame, sc.end_frame)
        else:
            merged.append(sc)
    
    # Post-process: Check if the first scene is still too short (it was never merged into a previous one)
    # If so, merge it forward into the second scene.
    if len(merged) > 1:
        first = merged[0]
        if (first.end_frame - first.start_frame + 1) < min_len_frames:
            second = merged[1]
            merged[1] = Scene(first.start_frame, second.end_frame)
            merged.pop(0)
    
    return merged


def adaptive_stride(scene_len: int) -> int:
    """Adaptive stride based on scene length."""
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
    """Decode frames from a specific scene."""
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
    """CLIP feature extractor."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        log("CLIP ViT-B/32 loaded")
    
    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        pil_list = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        batch = torch.stack([self.preprocess(p) for p in pil_list]).to(self.device)
        
        with torch.no_grad():
            feats = self.model.encode_image(batch).float().cpu().numpy()
        
        return l2_normalize(feats, axis=1)


# ============================================================================
# STEP 2: Multi-Prompt Anime Attributes
# ============================================================================

MULTI_PROMPTS = {
    "sharpness": [
        ("Sharp anime frame.", "Blurry anime frame."),
        ("Crisp anime artwork.", "Fuzzy anime artwork."),
        ("Clear anime image.", "Unclear anime image."),
    ],
    "colorfulness": [
        ("Vibrant anime colors.", "Dull anime colors."),
        ("Colorful anime scene.", "Desaturated anime scene."),
        ("Rich anime palette.", "Muted anime palette."),
    ],
    "brightness": [
        ("Well-lit anime scene.", "Dark anime scene."),
        ("Bright anime frame.", "Dim anime frame."),
        ("Good exposure anime.", "Underexposed anime."),
    ],
    "sakuga": [
        ("High sakuga animation frame.", "Low sakuga animation frame."),
        ("Key animation frame.", "In-between animation frame."),
        ("Fluid motion anime.", "Static anime frame."),
    ],
    "cinematic": [
        ("Cinematic anime shot.", "Plain anime shot."),
        ("Well-composed anime.", "Poorly-composed anime."),
        ("Professional anime framing.", "Amateur anime framing."),
    ],
    "expression": [
        ("Expressive anime face.", "Bland anime face."),
        ("Emotional anime character.", "Neutral anime character."),
        ("Dynamic anime expression.", "Static anime expression."),
    ],
}

ATTR_NAMES = list(MULTI_PROMPTS.keys())


class MultiPromptScorer:
    """Multi-prompt CLIP quality scorer."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        
        self.pos_embeds = {}
        self.neg_embeds = {}
        
        with torch.no_grad():
            for attr, prompts in MULTI_PROMPTS.items():
                pos_list, neg_list = [], []
                for pos_text, neg_text in prompts:
                    pos_tok = clip.tokenize([pos_text]).to(device)
                    neg_tok = clip.tokenize([neg_text]).to(device)
                    pos_list.append(self.model.encode_text(pos_tok).float())
                    neg_list.append(self.model.encode_text(neg_tok).float())
                self.pos_embeds[attr] = torch.cat(pos_list, dim=0)
                self.neg_embeds[attr] = torch.cat(neg_list, dim=0)
        
        log(f"Multi-prompt scorer ready ({len(ATTR_NAMES)} attrs)")
    
    def score_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Score frames. Returns (T, 6) array."""
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        batch = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        
        with torch.no_grad():
            img_feats = self.model.encode_image(batch).float()
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        
        results = {}
        for attr in ATTR_NAMES:
            pos_emb = self.pos_embeds[attr] / self.pos_embeds[attr].norm(dim=-1, keepdim=True)
            neg_emb = self.neg_embeds[attr] / self.neg_embeds[attr].norm(dim=-1, keepdim=True)
            
            pos_sim = img_feats @ pos_emb.T
            neg_sim = img_feats @ neg_emb.T
            
            logits = torch.stack([pos_sim, neg_sim], dim=-1)
            probs = torch.softmax(logits * 100, dim=-1)
            scores = probs[:, :, 0].mean(dim=1).cpu().numpy()
            results[attr] = scores
        
        return np.stack([results[attr] for attr in ATTR_NAMES], axis=1)


# ============================================================================
# MAIN PRECOMPUTE
# ============================================================================

def process_video(
    video_path: Path,
    out_root: Path,
    clip_extractor: CLIPExtractor,
    anime_scorer: MultiPromptScorer,
    detector_kwargs: Dict[str, Any],
    min_scene_len: int = 30,
    jpeg_quality: int = 85,
) -> int:
    """Process a single video, return number of scenes saved."""
    video_stem = video_path.stem
    log(f"Processing: {video_stem}")
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    # Detect scenes
    detector = create_detector("transnetv2", **detector_kwargs)
    try:
        scenes_raw = detector.detect(str(video_path))
    finally:
        detector.close()
    
    if not scenes_raw:
        log(f"  No scenes detected, using whole video")
        scenes_raw = [Scene(0, total_frames - 1)]
    
    scenes = normalize_and_merge_scenes(scenes_raw, min_len_frames=min_scene_len)
    log(f"  {len(scenes)} scenes")
    
    # Save scenes.json
    video_out_dir = out_root / video_stem
    ensure_dir(video_out_dir)
    
    scene_rows = []
    for i, sc in enumerate(scenes):
        scene_rows.append({
            "scene_id": i,
            "start_frame": int(sc.start_frame),
            "end_frame": int(sc.end_frame),
            "duration_frames": int(sc.end_frame - sc.start_frame + 1),
        })
    save_json(scene_rows, str(video_out_dir / "scenes.json"))
    
    # Process each scene
    n_saved = 0
    for sid, sc in enumerate(scenes):
        s, e = int(sc.start_frame), int(sc.end_frame)
        scene_len = e - s + 1
        stride = adaptive_stride(scene_len)
        
        # Decode frames
        frames, frame_indices = decode_scene_frames(str(video_path), s, e, stride)
        
        if len(frames) < 2:
            continue
        
        # Create scene directory
        scene_dir = video_out_dir / f"scene_{sid:04d}"
        ensure_dir(scene_dir / "frames")
        
        # Save frames
        for i, im in enumerate(frames):
            cv2.imwrite(
                str(scene_dir / "frames" / f"{i:06d}.jpg"),
                im, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
        
        # STEP 1: CLIP features
        feats = clip_extractor.extract(frames)
        np.save(scene_dir / "feats.npy", feats.astype(np.float32))
        
        # STEP 1: Relative positions
        rel_positions = np.array([
            (fidx - s) / max(1, e - s)
            for fidx in frame_indices
        ], dtype=np.float32)
        np.save(scene_dir / "rel_positions.npy", rel_positions)
        
        # STEP 2: Anime attributes (multi-prompt)
        anime_attrs = anime_scorer.score_frames(frames)
        np.save(scene_dir / "anime_attrs.npy", anime_attrs.astype(np.float32))
        
        # Save meta
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
            "version": "v11",
        }
        with open(scene_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        n_saved += 1
        log(f"  scene {sid:04d}: T={len(frames)}, stride={stride}")
    
    return n_saved


def main():
    parser = argparse.ArgumentParser(description="V11 All-in-One Precompute")
    
    # Input/Output
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output dataset directory")
    
    # Scene detection
    parser.add_argument("--model_dir", type=str, default="src/models/TransNetV2",
                        help="TransNetV2 model directory")
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument("--min_scene_len", type=int, default=30)
    
    # Processing
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    
    args = parser.parse_args()
    
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    
    # Initialize extractors
    log("Initializing CLIP extractor...")
    clip_extractor = CLIPExtractor(device=args.device)
    
    log("Initializing multi-prompt scorer...")
    anime_scorer = MultiPromptScorer(device=args.device)
    
    detector_kwargs = {
        "model_dir": args.model_dir,
        "prob_threshold": args.prob_threshold,
        "device": args.device,
    }
    
    # Find videos
    video_paths = []
    for ext in ['*.mp4', '*.avi', '*.mkv', '*.mov']:
        video_paths.extend(glob.glob(os.path.join(args.video_dir, ext)))
    video_paths = sorted(video_paths)
    
    log(f"Found {len(video_paths)} videos")
    
    total_scenes = 0
    for vp in tqdm(video_paths, desc="Videos"):
        n = process_video(
            Path(vp), out_root, clip_extractor, anime_scorer,
            detector_kwargs, args.min_scene_len, args.jpeg_quality
        )
        total_scenes += n
    
    log(f"✅ Done! {total_scenes} scenes saved to {out_root}")
    log(f"Each scene has: feats.npy, rel_positions.npy, anime_attrs.npy, frames/")


if __name__ == "__main__":
    main()
