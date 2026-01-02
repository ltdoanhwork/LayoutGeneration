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
        --video_path path/to/video.mp4 \
        --checkpoint runs/dsn_v11_simple/best.pt \
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
from scripts.precompute_all_v11 import (
    CLIPExtractor, 
    MultiPromptScorer, 
    normalize_and_merge_scenes, 
    adaptive_stride, 
    decode_scene_frames
)

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
        
    def get_detector(self, model_dir="src/models/TransNetV2"):
        if self.detector is None:
            self.detector = create_detector("transnetv2", model_dir=model_dir, device=self.device)
        return self.detector

    def process_video(
        self, 
        video_path: str, 
        output_dir: str, 
        budget_ratio: float = 0.15,
        b_min: int = 3,
        b_max: int = 15
    ):
        video_path = Path(video_path)
        out_root = Path(output_dir) / video_path.stem
        ensure_dir(out_root)
        
        # Prepare consolidated outputs
        scene_rows = []
        key_rows = []
        all_prob_rows = []
        
        # 1. Detect Scenes
        print("  Detecting scenes...")
        detector = self.get_detector()
        scenes_raw = detector.detect(str(video_path))
        scenes = normalize_and_merge_scenes(scenes_raw)
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
            stride = adaptive_stride(scene_len)
            
            # Decode frames
            frames, frame_indices = decode_scene_frames(str(video_path), s, e, stride)
            
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
            
            # Selection
            budget = max(b_min, min(b_max, int(len(frames) * budget_ratio)))
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
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
    parser.add_argument("--budget_ratio", type=float, default=0.15)
    
    args = parser.parse_args()
    
    predictor = V11Predictor(args.checkpoint, args.device)
    
    # Handle directory or single file
    inp = Path(args.video_path)
    if inp.is_dir():
        videos = sorted(inp.glob("*.mp4")) + sorted(inp.glob("*.mkv"))
        for v in videos:
            predictor.process_video(str(v), args.output_dir, args.budget_ratio)
    else:
        predictor.process_video(str(inp), args.output_dir, args.budget_ratio)

if __name__ == "__main__":
    main()
