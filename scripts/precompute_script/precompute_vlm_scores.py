#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute VLM Quality Scores

Iterates through the dataset and generates VLM pseudo-labels for distillation.
Optimized for A100 with batch processing support.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm
from PIL import Image
import torch

from src.models.vlm_iqa_teacher import VLMIQATeacher

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VLM Quality Scores")
    parser.add_argument("--dataset_dir", type=str, default="data/sakuga_dataset_100_samples", help="Path to dataset")
    parser.add_argument("--vlm_model", type=str, default="internvl2-8b", help="VLM model to use (internvl2-8b, qwen-vl-chat)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for VLM inference")
    parser.add_argument("--scene_limit", type=int, default=None, help="Limit number of scenes to process (for testing)")
    parser.add_argument("--force", action="store_true", help="Force recomputation if file exists")
    return parser.parse_args()

def load_scene_frames(scene_dir: Path) -> List[Image.Image]:
    """Load all frames from a scene directory."""
    frame_dir = scene_dir / "frames"
    if not frame_dir.exists():
        return []
    
    # Get all .jpg or .png files, sorted by name
    frame_paths = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))
    
    frames = []
    for p in frame_paths:
        try:
            frames.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"Error loading {p}: {e}")
            
    return frames

def main():
    args = parse_args()
    dataset_path = Path(args.dataset_dir)
    
    if not dataset_path.exists():
        print(f"Dataset directory not found: {args.dataset_dir}")
        return

    # Initialize VLM Teacher
    # Since we are on A100, we can use bfloat16 for speed and memory efficiency
    teacher = VLMIQATeacher(model_name=args.vlm_model, device=args.device)
    
    # 1. Collect all scene directories
    scene_dirs = []
    video_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    for v_dir in sorted(video_dirs):
        # Only include folders named 'scene_' followed by digits
        v_scenes = [d for d in v_dir.iterdir() if d.is_dir() and d.name.startswith("scene_") and d.name[6:].isdigit()]
        scene_dirs.extend(sorted(v_scenes))
    
    print(f"Found {len(scene_dirs)} actual scenes across {len(video_dirs)} videos.")
    
    if args.scene_limit:
        scene_dirs = scene_dirs[:args.scene_limit]
        print(f"Limited to {len(scene_dirs)} scenes for this run.")

    # 2. Process each scene
    for scene_dir in tqdm(scene_dirs, desc="Processing Scenes"):
        out_path = scene_dir / "vlm_quality.npy"
        
        if out_path.exists() and not args.force:
            continue
            
        # Load frames
        frames = load_scene_frames(scene_dir)
        if not frames:
            continue
            
        # Batch inference
        all_scores = []
        for i in range(0, len(frames), args.batch_size):
            batch = frames[i : i + args.batch_size]
            batch_scores = teacher.batch_annotate(batch)
            all_scores.append(batch_scores)
            
        if all_scores:
            final_scores = np.concatenate(all_scores, axis=0)
            # Save as (N, 8) numpy array
            np.save(out_path, final_scores)
            
    print("\nAnnotation complete! VLM scores saved to vlm_quality.npy in each scene folder.")

if __name__ == "__main__":
    main()
