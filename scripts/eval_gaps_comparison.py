#!/usr/bin/env python3
"""
Evaluate LPIPS Gap and Feature Distance Gap for V11 Final checkpoints.
Compare with VSUMM baseline.
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import cv2
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dsn_v8 import create_dsn_v8
from src.distance_selector.registry import create_metric
from src.datasets import load_scene_dir, build_epoch_index


def read_all_frames_sparse(video_path: str, stride: int = 5) -> List[np.ndarray]:
    """Read frames from video with stride."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    
    cap.release()
    return frames


def sample_video_frames(video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
    """Read specific frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def lpips_gap(video_path: str, key_frames: List[int], device="cuda", net="alex") -> float:
    """
    Compute LPIPS gap: mean minimum LPIPS distance from all frames to selected keyframes.
    Lower is better (selected keyframes cover the video well).
    """
    if not key_frames:
        return float("nan")
    
    metric = create_metric("lpips", net=net, device=device)
    
    # Sample all frames sparsely
    all_frames = read_all_frames_sparse(video_path, stride=5)
    if not all_frames:
        return float("nan")
    
    # Get selected keyframes
    keys = sample_video_frames(video_path, key_frames)
    if not keys:
        return float("nan")
    
    # Preprocess to tensors
    Ts_all = [metric.preprocess_bgr(f) for f in all_frames]
    Ts_keys = [metric.preprocess_bgr(f) for f in keys]
    
    # For each frame, find minimum distance to any keyframe
    vals = []
    with torch.no_grad():
        for Ta in Ts_all:
            min_dist = 1e9
            for Tk in Ts_keys:
                d = metric.pair_distance(Ta, Tk)
                if d < min_dist:
                    min_dist = d
            vals.append(min_dist)
    
    return float(np.mean(vals)) if vals else float("nan")


def feature_distance_gap(features_all: np.ndarray, selected_indices: List[int]) -> float:
    """
    Compute feature distance gap: mean minimum cosine distance from all frames to selected frames.
    Lower is better.
    """
    if not selected_indices or len(features_all) == 0:
        return float("nan")
    
    # L2 normalize features
    feats_all = features_all / (np.linalg.norm(features_all, axis=1, keepdims=True) + 1e-12)
    feats_sel = feats_all[selected_indices]
    
    # For each frame, find minimum distance to any selected frame
    gaps = []
    for i in range(len(feats_all)):
        # Cosine distance = 1 - cosine similarity
        similarities = feats_sel @ feats_all[i]
        min_distance = 1.0 - np.max(similarities)
        gaps.append(min_distance)
    
    return float(np.mean(gaps))


def evaluate_checkpoint(
    checkpoint_path: str,
    val_scenes: List[Path],
    device: str = "cuda",
    budget_ratio: float = 0.15,
    Bmin: int = 3,
    Bmax: int = 15,
    no_anime_attrs: bool = False,
) -> Dict[str, float]:
    """Evaluate a single checkpoint on validation scenes."""
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})
    
    # Determine feature dimension
    feat_dim = config.get("feat_dim", 512)
    use_anime = not config.get("no_anime_attrs", no_anime_attrs)
    full_feat_dim = feat_dim + (6 if use_anime else 0)
    
    # Create model
    model = create_dsn_v8(
        feat_dim=full_feat_dim,
        use_pcgrad=False,
        num_attn_layers=config.get("num_attn_layers", 2),
        gating_hidden=config.get("gating_hidden", 64),
        lstm_hidden=config.get("lstm_hidden", 128),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    lpips_gaps = []
    feat_gaps = []
    
    print(f"  Evaluating {len(val_scenes)} scenes...")
    
    for scene_dir in tqdm(val_scenes, desc="Scenes"):
        try:
            sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=True)
            
            # Skip if missing data
            if sample.anime_attrs is None:
                continue
            
            # Construct features
            if no_anime_attrs or not use_anime:
                feats_input = sample.feats
            else:
                feats_input = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
            
            feats_t = torch.from_numpy(feats_input).float().unsqueeze(0).to(device)
            T = len(sample.feats)
            budget = max(Bmin, min(Bmax, int(T * budget_ratio)))
            
            # Predict
            with torch.no_grad():
                probs, _ = model(feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
            # Compute feature distance gap
            feat_gap = feature_distance_gap(sample.feats, sel_idx)
            if not np.isnan(feat_gap):
                feat_gaps.append(feat_gap)
            
            # TODO: Compute LPIPS gap (need video path from metadata)
            # For now, skip LPIPS gap computation
            
        except Exception as e:
            print(f"Error processing {scene_dir}: {e}")
            continue
    
    results = {
        "feat_gap_mean": float(np.mean(feat_gaps)) if feat_gaps else float("nan"),
        "feat_gap_std": float(np.std(feat_gaps)) if feat_gaps else float("nan"),
        "lpips_gap_mean": float(np.mean(lpips_gaps)) if lpips_gaps else float("nan"),
        "lpips_gap_std": float(np.std(lpips_gaps)) if lpips_gaps else float("nan"),
        "n_scenes": len(feat_gaps),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS and Feature Gaps")
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new",
                       help="Directory containing checkpoints")
    parser.add_argument("--val_root", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test",
                       help="Validation dataset root")
    parser.add_argument("--output", type=str, default="gap_comparison_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Find all checkpoints
    ckpt_dir = Path(args.checkpoint_dir)
    
    # Look for best*.pt files
    best_ckpts = sorted(ckpt_dir.glob("best_epoch_*.pt"))
    
    if not best_ckpts:
        # Fallback to best.pt
        if (ckpt_dir / "best.pt").exists():
            best_ckpts = [ckpt_dir / "best.pt"]
        else:
            print(f"No checkpoints found in {ckpt_dir}")
            return
    
    print(f"Found {len(best_ckpts)} checkpoints to evaluate")
    
    # Load validation scenes
    val_scenes = build_epoch_index(args.val_root)
    print(f"Found {len(val_scenes)} validation scenes")
    
    # Evaluate each checkpoint
    all_results = {}
    
    for ckpt_path in best_ckpts:
        ckpt_name = ckpt_path.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name}")
        print(f"{'='*60}")
        
        results = evaluate_checkpoint(
            str(ckpt_path),
            val_scenes,
            device=args.device,
        )
        
        all_results[ckpt_name] = results
        
        print(f"\nResults for {ckpt_name}:")
        print(f"  Feature Gap:  {results['feat_gap_mean']:.4f} ± {results['feat_gap_std']:.4f}")
        print(f"  LPIPS Gap:    {results['lpips_gap_mean']:.4f} ± {results['lpips_gap_std']:.4f}")
        print(f"  Scenes:       {results['n_scenes']}")
    
    # Save results
    output_path = ckpt_dir / args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<40} {'FeatGap':>10} {'LPIPS Gap':>12}")
    print("-" * 60)
    
    for ckpt_name, results in all_results.items():
        feat_gap = results['feat_gap_mean']
        lpips_gap = results['lpips_gap_mean']
        print(f"{ckpt_name:<40} {feat_gap:>10.4f} {lpips_gap:>12.4f}")


if __name__ == "__main__":
    main()
