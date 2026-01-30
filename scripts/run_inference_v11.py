#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 Inference Script

Runs the full V11 pipeline on a single video or directory of videos:
1. Scene Detection (TransNetV2)
2. Feature Extraction (CLIP + Multi-Prompt Anime Attributes)
3. Model Inference (DSN V8)
4. Keyframe Selection & Saving

Usage:
    python -m scripts.run_inference_v11 \
        --video_path /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/9046.mp4 \
        --checkpoint runs/training_v11_final_new/best.pt \
        --output_dir outputs/inference_v11 \
        --device cuda
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import torch
import clip
from PIL import Image
from tqdm import tqdm

from src.scene_detection import create_detector, Scene
from src.models.dsn_v8 import create_dsn_v8
from scripts.precompute_script.precompute_all_v11 import (
    CLIPExtractor, 
    MultiPromptScorer, 
    normalize_and_merge_scenes, 
    adaptive_stride, 
    decode_scene_frames
)

def deduplicate_by_cosine(
    selected_indices: List[int],
    features: np.ndarray,
    probs: np.ndarray,
    similarity_threshold: float = 0.92,
    budget: int = None
) -> List[int]:
    """
    Remove duplicate frames using cosine similarity with two-pointer technique.
    
    Problem: When stride is small, consecutive frames are very similar.
    Similar CLIP features → similar high scores → selecting near-duplicate frames.
    
    Solution: After initial selection, remove frames that are too similar to 
    previously accepted frames (cosine similarity > threshold).
    
    Args:
        selected_indices: Initial selected frame indices (sorted by frame order)
        features: Feature matrix (T, D) - CLIP features
        probs: Probability scores (T,)
        similarity_threshold: If cosine > this, consider as duplicate (default 0.92)
            - 0.90: Aggressive - only keep very different frames
            - 0.92: Balanced - remove near-similar frames
            - 0.95: Conservative - only remove near-identical
        budget: Target number of frames to return
        
    Returns:
        Deduplicated list of frame indices (sorted by frame order)
    """
    if len(selected_indices) <= 1:
        return selected_indices
    
    # Normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features_norm = features / norms
    
    # Two pointers technique: keep track of accepted frames
    accepted = [selected_indices[0]]  # Always keep first frame
    
    for i in range(1, len(selected_indices)):
        current_idx = selected_indices[i]
        last_accepted_idx = accepted[-1]
        
        # Compute cosine similarity with last accepted frame
        cosine_sim = np.dot(features_norm[current_idx], features_norm[last_accepted_idx])
        
        if cosine_sim < similarity_threshold:
            # Not a duplicate, accept it
            accepted.append(current_idx)
        # else: Duplicate detected, skip this frame
    
    # If we removed too many, try to fill from remaining candidates
    if budget is not None and len(accepted) < budget:
        # Get all frames sorted by prob (descending), excluding already accepted
        all_sorted = np.argsort(probs)[::-1]
        
        for candidate in all_sorted:
            if candidate in accepted:
                continue
            if len(accepted) >= budget:
                break
                
            # Check if candidate is similar to any accepted frame
            is_duplicate = False
            for acc_idx in accepted:
                cosine_sim = np.dot(features_norm[candidate], features_norm[acc_idx])
                if cosine_sim >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                accepted.append(candidate)
        
        # Re-sort by frame order
        accepted = sorted(accepted)
    
    return accepted

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

class V11Predictor:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading V11 model from {checkpoint_path}...")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        
        # Determine feature dim from weights if possible, else default
        # Input proj weight shape: (hidden_dim, feat_dim)
        if "input_proj.0.weight" in state_dict:
            feat_dim = state_dict["input_proj.0.weight"].shape[1]
        else:
            feat_dim = 518  # Default V11 (512 CLIP + 6 Attributes)
            
        print(f"Detected feature dim: {feat_dim}")
        
        # Create model
        self.model = create_dsn_v8(feat_dim=feat_dim, use_pcgrad=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
        # Initialize extractors
        self.clip_extractor = CLIPExtractor(device=device)
        self.anime_scorer = MultiPromptScorer(device=device)
        
        # Scene detector (lazy init)
        self.detector = None
        
    def get_detector(self, model_dir="src/models/TransNetV2", threshold=0.5):
        # Always create new detector if threshold changes or not created
        # Simple fix: recreate or update params
        self.detector = create_detector("transnetv2", model_dir=model_dir, device=self.device, prob_threshold=threshold)
        return self.detector

    def process_video(
        self, 
        video_path: str, 
        output_dir: str, 
        budget_ratio: float = 0.15,
        b_min: int = 3, #3 Đăng sửa để check
        b_max: int = 15,
        stride: int = None,
        save_images: bool = False,
        scene_threshold: float = 0.5,
        min_scene_len: int = 15,
        dedup_threshold: float = 0.92  # Cosine similarity threshold for deduplication
    ):
        video_path = Path(video_path)
        out_root = Path(output_dir) / video_path.stem
        ensure_dir(out_root)
        
        # Save images directory
        if save_images:
            ensure_dir(out_root / "keyframes")
        
        # Prepare consolidated outputs
        scene_rows = []
        key_rows = []
        all_prob_rows = []
        
        # 1. Detect Scenes
        print(f"  Detecting scenes (threshold={scene_threshold}, min_len={min_scene_len})...")
        detector = self.get_detector(threshold=scene_threshold)
        scenes_raw = detector.detect(str(video_path))
        scenes = normalize_and_merge_scenes(scenes_raw, min_len_frames=min_scene_len)
        print(f"  Found {len(scenes)} scenes")
        
        # Get video FPS for timecode
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        def timecode(fidx):
            sec = fidx / fps
            return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{sec%60:06.3f}"

        # 2. Process each scene
        for i, scene in enumerate(tqdm(scenes, desc="Inferring Scenes")):
            s, e = int(scene.start_frame), int(scene.end_frame)
            scene_len = e - s + 1
            
            # Use custom stride or adaptive
            curr_stride = stride if stride is not None else adaptive_stride(scene_len)
            
            # Decode frames
            frames, frame_indices = decode_scene_frames(str(video_path), s, e, curr_stride)
            
            if len(frames) < 2:
                continue
                
            # Extract Features
            clip_feats = self.clip_extractor.extract(frames)       # (T, 512)
            anime_attrs = self.anime_scorer.score_frames(frames)   # (T, 6)
            
            # Concatenate
            feats_full = np.concatenate([clip_feats, anime_attrs], axis=1) # (T, 518)
            
            # Model Inference
            feats_t = torch.from_numpy(feats_full).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                probs, _ = self.model(feats_t, return_gating=False)
                probs = probs.squeeze(0).cpu().numpy()
            
            # Selection with Cosine Deduplication
            budget = max(b_min, min(b_max, int(len(frames) * budget_ratio)))
            
            # Step 1: Get top candidates (3x budget as replacement pool)
            candidate_pool_size = min(len(frames), budget * 3)
            top_candidates = sorted(np.argsort(probs)[-candidate_pool_size:].tolist())
            
            # Step 2: Deduplicate using cosine similarity on CLIP features
            sel_idx = deduplicate_by_cosine(
                selected_indices=top_candidates,
                features=clip_feats,  # Use CLIP features (512-dim) for similarity
                probs=probs,
                similarity_threshold=dedup_threshold,
                budget=budget
            )
            
            # Collect Scene Data
            scene_rows.append({
                "scene_id": i,
                "start_frame": s,
                "end_frame": e,
                "start_time": timecode(s),
                "end_time": timecode(e),
                "duration_frames": scene_len,
                "duration_seconds": scene_len / fps
            })
            
            # Collect Keyframes & Probs
            for local_idx, prob in enumerate(probs):
                global_idx = frame_indices[local_idx]
                is_selected = int(local_idx in sel_idx)
                
                row = {
                    "scene_id": i,
                    "frame_global": global_idx,
                    "frame_in_scene": local_idx,
                    "time": timecode(global_idx),
                    "prob": float(prob),
                }
                
                # Add to all_probs
                all_prob_rows.append({**row, "selected": is_selected})
                
                # Add to keyframes if selected
                if is_selected:
                    key_rows.append(row)
                    
                    # Save image if requested
                    if save_images:
                        img_path = out_root / "keyframes" / f"scene_{i:03d}_frame_{global_idx:06d}.jpg"
                        cv2.imwrite(str(img_path), frames[local_idx])
        
        # 3. Save Consolidated Results
        import csv
        import json
        
        # scenes.json
        with open(out_root / "scenes.json", "w") as f:
            json.dump(scene_rows, f, indent=2)
            
        # keyframes.csv
        with open(out_root / "keyframes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scene_id", "frame_global", "frame_in_scene", "time", "prob"])
            writer.writeheader()
            writer.writerows(key_rows)
            
        # all_probs.csv
        with open(out_root / "all_probs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scene_id", "frame_global", "frame_in_scene", "time", "prob", "selected"])
            writer.writeheader()
            writer.writerows(all_prob_rows)

        print(f"✅ Done! Results saved to {out_root}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--budget_ratio", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=None, help="Frame sampling stride (default: adaptive)")
    parser.add_argument("--save_images", action="store_true", help="Save selected keyframe images")
    parser.add_argument("--scene_threshold", type=float, default=0.5, help="TransNetV2 boundary threshold")
    parser.add_argument("--min_scene_len", type=int, default=15, help="Minimum scene length (frames) to merge")
    parser.add_argument("--dedup_threshold", type=float, default=0.92, 
                        help="Cosine similarity threshold for deduplication (0.90=aggressive, 0.95=conservative)")
    
    args = parser.parse_args()
    
    predictor = V11Predictor(args.checkpoint, args.device)
    
    # Handle directory or single file
    inp = Path(args.video_path)
    if inp.is_dir():
        videos = sorted(inp.glob("*.mp4")) + sorted(inp.glob("*.mkv"))
        for v in videos:
            predictor.process_video(
                str(v), args.output_dir, args.budget_ratio, 
                stride=args.stride, save_images=args.save_images, 
                scene_threshold=args.scene_threshold, min_scene_len=args.min_scene_len,
                dedup_threshold=args.dedup_threshold
            )
    else:
        predictor.process_video(
            str(inp), args.output_dir, args.budget_ratio, 
            stride=args.stride, save_images=args.save_images, 
            scene_threshold=args.scene_threshold, min_scene_len=args.min_scene_len,
            dedup_threshold=args.dedup_threshold
        )

if __name__ == "__main__":
    main()


"""
python3 -m scripts.run_inference_v11 \
  --video_path /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test/70025.mp4 \
  --checkpoint runs/training_v11_final_new/best.pt \
  --output_dir outputs/inference_v11_70025 \
  --budget_ratio 0.1 \
  --stride 5 \
  --save_images

python3 -m scripts.run_inference_v11 \
  --video_path /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/9046.mp4 \
  --checkpoint runs/training_v11_final_new/best.pt \
  --output_dir outputs/inference_v11_9046 \
  --budget_ratio 0.1 \
  --stride 5 \
  --save_images
  
python3 -m scripts.run_inference_v11 \
  --video_path /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/115042.mp4 \
  --checkpoint runs/training_v11_final_new/best.pt \
  --output_dir outputs/inference_v11_115042 \
  --budget_ratio 0.1 \
  --stride 5 \
  --save_images \
  --scene_threshold 0.8 \
  --min_scene_len 100

python3 -m scripts.run_inference_v11 \
  --video_path /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/9046.mp4 \
  --checkpoint runs/training_v11_final_new/best.pt \
  --output_dir outputs/inference_v11_9046 \
  --budget_ratio 0.2 \
  --stride 5 \
  --save_images \
  --scene_threshold 0.8 \
  --min_scene_len 100

python3 -m scripts.run_inference_v11 \
  --video_path /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/32030.mp4 \
  --checkpoint runs/training_v11_final_new/best.pt \
  --output_dir outputs/inference_v11_32030 \
  --budget_ratio 0.1 \
  --stride 8 \
  --save_images \
  --scene_threshold 0.8 \
  --min_scene_len 50
"""


