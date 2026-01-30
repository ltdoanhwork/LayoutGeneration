#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Data Preparation: Multi-Prompt Levels (1-5 Pairs)
Generates anime_attrs_{N}pair.npy for ablation studies.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

import torch
import clip
from PIL import Image

# ============================================================================
# EXPANDED PROMPT DEFINITIONS (5 PAIRS PER ATTRIBUTE)
# Hierarchy: Pair 1 (Best/Simplest) -> Pair 5 (More specific/Abstract)
# ============================================================================
FULL_PROMPTS = {
    "sharpness": [
        ("Sharp anime frame.", "Blurry anime frame."),               # 1
        ("Crisp anime artwork.", "Fuzzy anime artwork."),            # 2
        ("Clear anime image.", "Unclear anime image."),              # 3 (Orig Ends)
        ("High resolution anime.", "Pixelated anime frame."),        # 4
        ("Detailed lineart.", "Indistinct lines."),                  # 5
    ],
    "colorfulness": [
        ("Vibrant anime colors.", "Dull anime colors."),             # 1
        ("Colorful anime scene.", "Desaturated anime scene."),       # 2
        ("Rich anime palette.", "Muted anime palette."),             # 3 (Orig Ends)
        ("Vivid anime art.", "Grayish anime art."),                  # 4
        ("Saturated colors.", "Washed out colors."),                 # 5
    ],
    "brightness": [
        ("Well-lit anime scene.", "Dark anime scene."),              # 1
        ("Bright anime frame.", "Dim anime frame."),                 # 2
        ("Good exposure anime.", "Underexposed anime."),             # 3 (Orig Ends)
        ("Balanced lighting.", "Poor lighting shadow."),             # 4
        ("Clear visibility.", "Obscured visibility."),               # 5
    ],
    "sakuga": [
        ("High sakuga animation frame.", "Low sakuga animation frame."), # 1
        ("Key animation frame.", "In-between animation frame."),         # 2
        ("Fluid motion anime.", "Static anime frame."),                  # 3 (Orig Ends)
        ("Dynamic action shot.", "Still talking head."),                 # 4
        ("Intense animation quality.", "Average animation quality."),    # 5
    ],
    "cinematic": [
        ("Cinematic anime shot.", "Plain anime shot."),                  # 1
        ("Well-composed anime.", "Poorly-composed anime."),              # 2
        ("Professional anime framing.", "Amateur anime framing."),       # 3 (Orig Ends)
        ("Movie-quality layout.", "TV-quality layout."),                 # 4
        ("Dramatic camera angle.", "Flat camera angle."),                # 5
    ],
    "expression": [
        ("Expressive anime face.", "Bland anime face."),                 # 1
        ("Emotional anime character.", "Neutral anime character."),      # 2
        ("Dynamic anime expression.", "Static anime expression."),       # 3 (Orig Ends)
        ("Character acting.", "Blank stare."),                           # 4
        ("Vivid facial emotion.", "Expressionless face."),               # 5
    ],
}

ATTR_NAMES = list(FULL_PROMPTS.keys())


class AblationCLIPScorer:
    def __init__(self, device: str = "cuda", num_pairs: int = 3):
        self.device = device
        self.num_pairs = num_pairs
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        
        # Pre-encode text prompts based on requested num_pairs
        self.pos_embeds = {}
        self.neg_embeds = {}
        
        print(f"[AblationCLIP] Loading {num_pairs} prompt pairs per attribute...")
        
        with torch.no_grad():
            for attr, all_prompts in FULL_PROMPTS.items():
                # Select first N pairs
                selected_prompts = all_prompts[:num_pairs]
                
                pos_list = []
                neg_list = []
                for pos_text, neg_text in selected_prompts:
                    pos_tok = clip.tokenize([pos_text]).to(device)
                    neg_tok = clip.tokenize([neg_text]).to(device)
                    pos_list.append(self.model.encode_text(pos_tok).float())
                    neg_list.append(self.model.encode_text(neg_tok).float())
                
                self.pos_embeds[attr] = torch.cat(pos_list, dim=0)  # (num_pairs, 512)
                self.neg_embeds[attr] = torch.cat(neg_list, dim=0)
    
    def score_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        # Preprocess
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        batch = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        
        with torch.no_grad():
            img_feats = self.model.encode_image(batch).float()
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
            results = []
            for attr in ATTR_NAMES:
                pos_emb = self.pos_embeds[attr]
                neg_emb = self.neg_embeds[attr]
                
                pos_emb = pos_emb / pos_emb.norm(dim=-1, keepdim=True)
                neg_emb = neg_emb / neg_emb.norm(dim=-1, keepdim=True)
                
                pos_sim = img_feats @ pos_emb.T
                neg_sim = img_feats @ neg_emb.T
                
                logits = torch.stack([pos_sim, neg_sim], dim=-1)
                probs = torch.softmax(logits * 100, dim=-1)
                scores = probs[:, :, 0]  # (B, num_pairs)
                
                # Average across pairs
                avg_score = scores.mean(dim=1).cpu().numpy()
                results.append(avg_score)
            
            # Stack: (B, 6)
            return np.stack(results, axis=1)


def process_scene(scene_dir: Path, scorer: AblationCLIPScorer, filename: str) -> bool:
    frames_dir = scene_dir / "frames"
    if not frames_dir.exists():
        return False
    
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if len(frame_files) < 2:
        return False
    
    frames = [cv2.imread(str(f)) for f in frame_files]
    attrs = scorer.score_frames(frames)
    
    # Save
    np.save(scene_dir / filename, attrs.astype(np.float32))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--num_pairs", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    scorer = AblationCLIPScorer(device=args.device, num_pairs=args.num_pairs)
    
    # Determine output filename
    outfile_name = f"anime_attrs_{args.num_pairs}pair.npy"
    print(f"Target Output: {outfile_name}")
    
    dataset_path = Path(args.dataset_dir)
    scene_dirs = []
    
    # Walk dataset
    if (dataset_path / "train").exists():
        # Handle split structure if exists, otherwise flat
        pass 
    
    # Flat search for scene_* folders
    for root, dirs, files in os.walk(dataset_path):
        for d in dirs:
            if d.startswith("scene_"):
                scene_dirs.append(Path(root) / d)
                
    print(f"Found {len(scene_dirs)} scenes to process.")
    
    success = 0
    for s_dir in tqdm(scene_dirs, desc=f"Gen {args.num_pairs}-pair"):
        if process_scene(s_dir, scorer, outfile_name):
            success += 1
            
    print(f"Done. Processed {success} scenes.")

if __name__ == "__main__":
    main()
