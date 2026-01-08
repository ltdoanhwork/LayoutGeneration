#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 Anime Attributes Preparation

Uses multi-prompt CLIP ensemble for more robust quality scoring.
Each attribute has multiple synonym prompts, averaged for final score.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

import torch
import clip
from PIL import Image


# V11: Multi-prompt ensemble for each attribute
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


class MultiPromptCLIPScorer:
    """
    Multi-prompt CLIP quality scorer.
    Averages scores across synonym prompts for robustness.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        
        # Pre-encode all text prompts
        self.pos_embeds = {}  # attr -> list of embeddings
        self.neg_embeds = {}
        
        with torch.no_grad():
            for attr, prompts in MULTI_PROMPTS.items():
                pos_list = []
                neg_list = []
                for pos_text, neg_text in prompts:
                    pos_tok = clip.tokenize([pos_text]).to(device)
                    neg_tok = clip.tokenize([neg_text]).to(device)
                    pos_list.append(self.model.encode_text(pos_tok).float())
                    neg_list.append(self.model.encode_text(neg_tok).float())
                self.pos_embeds[attr] = torch.cat(pos_list, dim=0)  # (n_prompts, 512)
                self.neg_embeds[attr] = torch.cat(neg_list, dim=0)
        
        print(f"[V11 CLIP] Loaded {len(ATTR_NAMES)} attributes with multi-prompt ensemble")
    
    def score_batch(self, images: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Score a batch of images.
        
        Args:
            images: (B, 3, 224, 224) preprocessed images
        
        Returns:
            Dict[attr] -> (B,) array of scores in [0, 1]
        """
        with torch.no_grad():
            img_feats = self.model.encode_image(images).float()
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        
        results = {}
        for attr in ATTR_NAMES:
            pos_emb = self.pos_embeds[attr]  # (n_prompts, 512)
            neg_emb = self.neg_embeds[attr]
            
            # Normalize
            pos_emb = pos_emb / pos_emb.norm(dim=-1, keepdim=True)
            neg_emb = neg_emb / neg_emb.norm(dim=-1, keepdim=True)
            
            # Compute similarity to each prompt
            pos_sim = img_feats @ pos_emb.T  # (B, n_prompts)
            neg_sim = img_feats @ neg_emb.T
            
            # Softmax per-prompt to get scores
            logits = torch.stack([pos_sim, neg_sim], dim=-1)  # (B, n_prompts, 2)
            probs = torch.softmax(logits * 100, dim=-1)  # temperature=100
            scores = probs[:, :, 0]  # (B, n_prompts) - probability of positive
            
            # Average across prompts
            avg_score = scores.mean(dim=1).cpu().numpy()  # (B,)
            results[attr] = avg_score
        
        return results
    
    def score_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Score list of BGR frames.
        
        Returns:
            (T, 6) array of attribute scores
        """
        # Preprocess
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        batch = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        
        # Score
        attr_scores = self.score_batch(batch)
        
        # Stack into array
        return np.stack([attr_scores[attr] for attr in ATTR_NAMES], axis=1)


def process_scene(scene_dir: Path, scorer: MultiPromptCLIPScorer) -> bool:
    """Process a single scene and save anime_attrs_v11.npy"""
    frames_dir = scene_dir / "frames"
    if not frames_dir.exists():
        return False
    
    # Load frames
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if len(frame_files) < 2:
        return False
    
    frames = [cv2.imread(str(f)) for f in frame_files]
    
    # Score
    attrs = scorer.score_frames(frames)
    
    # Save with v11 suffix
    np.save(scene_dir / "anime_attrs_v11.npy", attrs.astype(np.float32))
    return True


def main():
    parser = argparse.ArgumentParser(description="V11 Multi-Prompt Anime Attrs")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    scorer = MultiPromptCLIPScorer(device=args.device)
    
    dataset_path = Path(args.dataset_dir)
    
    # Find all scene directories
    scene_dirs = []
    for video_dir in dataset_path.iterdir():
        if not video_dir.is_dir():
            continue
        for scene_dir in video_dir.iterdir():
            if scene_dir.is_dir() and scene_dir.name.startswith("scene_"):
                scene_dirs.append(scene_dir)
    
    print(f"Found {len(scene_dirs)} scenes")
    
    success = 0
    for scene_dir in tqdm(scene_dirs, desc="Processing"):
        if process_scene(scene_dir, scorer):
            success += 1
    
    print(f"Processed {success}/{len(scene_dirs)} scenes")


if __name__ == "__main__":
    main()


"""
Usage:
python -m scripts.prepare_anime_attrs_v11 \
    --dataset_dir data/sakuga_dataset_v11 \
    --device cuda
"""
