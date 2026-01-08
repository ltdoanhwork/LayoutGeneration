#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
import cv2

def load_frames(scene_dir: Path):
    """Load all frames from a scene directory."""
    frame_dir = scene_dir / "frames"
    if not frame_dir.exists():
        return []
    
    frame_files = sorted(frame_dir.glob("*.jpg"))
    frames = []
    for fp in frame_files:
        # Load as BGR (cv2 default) then convert to RGB
        img = cv2.imread(str(fp))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
    return frames

def get_prompt_pairs():
    """Define the prompt pairs for Anime-CLIP-IQA."""
    return [
        ("A sharp anime frame.", "A blurry anime frame."),
        ("A colorful anime frame.", "A dull anime frame."),
        ("A bright anime frame.", "A dark anime frame."),
        ("A dynamic sakuga action frame.", "A calm talking anime frame."),
        ("A cinematic impactful anime frame.", "An unremarkable anime frame."),
        ("An anime frame with strong facial expression.", "A neutral anime frame."),
    ]

def main():
    parser = argparse.ArgumentParser(description="Precompute Anime-CLIP-IQA attributes.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CLIP encoding")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Prepare text embeddings
    prompt_pairs = get_prompt_pairs()
    text_tokens = []
    for p_pos, p_neg in prompt_pairs:
        text_tokens.append(clip.tokenize(p_pos))
        text_tokens.append(clip.tokenize(p_neg))
    
    text_tokens = torch.cat(text_tokens).to(device) # (2*K, 77)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens) # (2*K, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Reshape to (K, 2, D) for easier pair-wise comparison
        K = len(prompt_pairs)
        D = text_features.shape[-1]
        text_features_pairs = text_features.view(K, 2, D) # (K, 2, D)

    # Process scenes
    dataset_root = Path(args.dataset_root)
    # Find all scene directories (assuming structure: root/video/scene_xxx)
    scene_dirs = sorted([p for p in dataset_root.glob("*/*") if (p / "frames").exists()])
    
    print(f"Found {len(scene_dirs)} scenes.")

    for scene_dir in tqdm(scene_dirs, desc="Processing Scenes"):
        output_path = scene_dir / "anime_attrs.npy"
        if output_path.exists():
            continue

        frames = load_frames(scene_dir)
        if not frames:
            continue

        # Process frames in batches
        all_scores = []
        
        for i in range(0, len(frames), args.batch_size):
            batch_frames = frames[i : i + args.batch_size]
            
            # Preprocess images
            images = torch.stack([preprocess(img) for img in batch_frames]).to(device) # (B, 3, 224, 224)
            
            with torch.no_grad():
                image_features = model.encode_image(images) # (B, D)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity: (B, D) @ (K, 2, D)^T -> (B, K, 2)
                # We can do this by iterating over K pairs or reshaping
                
                # Let's do it per pair to be clear
                batch_scores = []
                for k in range(K):
                    # pair_feats: (2, D)
                    pair_feats = text_features_pairs[k] 
                    
                    # logits: (B, 2) = (B, D) @ (D, 2)
                    logits = (100.0 * image_features @ pair_feats.T)
                    probs = logits.softmax(dim=-1)
                    
                    # We want the probability of the positive class (index 0)
                    score_pos = probs[:, 0] # (B,)
                    batch_scores.append(score_pos.cpu().numpy())
                
                # batch_scores is list of K arrays of shape (B,)
                # Stack to (B, K)
                batch_scores = np.stack(batch_scores, axis=1)
                all_scores.append(batch_scores)

        if all_scores:
            final_scores = np.concatenate(all_scores, axis=0) # (T, K)
            np.save(output_path, final_scores.astype(np.float32))

    print("Done.")

if __name__ == "__main__":
    main()
