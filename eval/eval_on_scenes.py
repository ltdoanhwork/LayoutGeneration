#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate model on precomputed scene features (training data format).

Usage:
    python -m eval.eval_on_scenes \
        --checkpoint runs/dsn_v10_stable/ep30.pt \
        --dataset_dir data/sakuga_dataset_100_samples \
        --output_dir runs/dsn_v10_stable/scene_eval
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm

from src.datasets import build_epoch_index, load_scene_dir


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    from src.models.dsn_v8 import create_dsn_v8
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Determine feat_dim from weights
    state_dict = ckpt.get("model_state_dict", ckpt)
    if "input_proj.0.weight" in state_dict:
        feat_dim = state_dict["input_proj.0.weight"].shape[1]
    else:
        feat_dim = 518  # default
    
    model = create_dsn_v8(feat_dim=feat_dim, use_pcgrad=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path} (feat_dim={feat_dim})")
    return model


def compute_metrics(
    quality_scores: np.ndarray,  # (T,) aggregated quality
    sel_idx: List[int],
) -> Dict[str, float]:
    """Compute quality metrics for selected frames."""
    T = len(quality_scores)
    if len(sel_idx) == 0:
        return {}
    
    # Percentile ranks
    ranks = np.argsort(np.argsort(quality_scores))
    percentiles = ranks / max(1, T - 1)
    
    sel_percentiles = percentiles[sel_idx]
    mean_percentile = float(np.mean(sel_percentiles))
    
    # Top-10% recall
    k10 = max(1, int(T * 0.1))
    top10_idx = set(np.argsort(quality_scores)[-k10:])
    top10_recall = len(set(sel_idx) & top10_idx) / k10
    
    # Above P90
    p90 = np.percentile(quality_scores, 90)
    above_p90 = sum(1 for i in sel_idx if quality_scores[i] >= p90) / len(sel_idx)
    
    # Z-score improvement
    all_mean = np.mean(quality_scores)
    all_std = np.std(quality_scores) + 1e-8
    sel_mean = np.mean(quality_scores[sel_idx])
    zscore = (sel_mean - all_mean) / all_std
    
    return {
        "mean_percentile_rank": mean_percentile,
        "top10_recall": top10_recall,
        "above_p90_ratio": above_p90,
        "zscore_improvement": zscore,
        "n_selected": len(sel_idx),
        "n_total": T,
    }


def evaluate_scene(
    model: torch.nn.Module,
    feats: np.ndarray,
    anime_attrs: np.ndarray,
    vlm_scores: Optional[np.ndarray],
    budget_ratio: float,
    Bmin: int,
    Bmax: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on a single scene."""
    T = len(feats)
    budget = max(Bmin, min(Bmax, int(T * budget_ratio)))
    
    # Prepare features
    feats_full = np.concatenate([feats, anime_attrs], axis=1)
    feats_t = torch.from_numpy(feats_full).float().unsqueeze(0).to(device)
    
    # Get probabilities
    with torch.no_grad():
        probs, _ = model(feats_t)
        probs = probs.squeeze(0).cpu().numpy()
    
    # Select top-K by probability
    sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
    
    # Compute metrics using VLM scores if available, else anime_attrs
    if vlm_scores is not None:
        quality = vlm_scores.mean(axis=1)
    else:
        quality = anime_attrs.mean(axis=1)
    
    metrics = compute_metrics(quality, sel_idx)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--budget_ratio", type=float, default=0.15)
    parser.add_argument("--Bmin", type=int, default=3)
    parser.add_argument("--Bmax", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_scenes", type=int, default=None)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = load_model(args.checkpoint, args.device)
    scene_dirs = build_epoch_index(args.dataset_dir)
    
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]
    
    print(f"\nEvaluating on {len(scene_dirs)} scenes...")
    
    all_metrics = []
    for scene_dir in tqdm(scene_dirs, desc="Evaluating"):
        sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=True, load_vlm_scores=True)
        
        if sample.anime_attrs is None:
            continue
        
        metrics = evaluate_scene(
            model, sample.feats, sample.anime_attrs, sample.vlm_scores,
            args.budget_ratio, args.Bmin, args.Bmax, args.device
        )
        metrics["scene"] = str(scene_dir)
        all_metrics.append(metrics)
    
    # Aggregate
    if all_metrics:
        avg_mpr = np.mean([m["mean_percentile_rank"] for m in all_metrics])
        avg_top10 = np.mean([m["top10_recall"] for m in all_metrics])
        avg_zscore = np.mean([m["zscore_improvement"] for m in all_metrics])
        avg_p90 = np.mean([m["above_p90_ratio"] for m in all_metrics])
        
        print(f"\n{'='*60}")
        print(f"📊 Results on {len(all_metrics)} scenes")
        print(f"{'='*60}")
        print(f"  Mean Percentile Rank: {avg_mpr:.4f}")
        print(f"  Top-10% Recall:       {avg_top10:.4f}")
        print(f"  Above P90 Ratio:      {avg_p90:.4f}")
        print(f"  Z-Score Improvement:  {avg_zscore:.4f}")
        print(f"{'='*60}")
        
        # Save results
        results = {
            "checkpoint": args.checkpoint,
            "n_scenes": len(all_metrics),
            "avg_mean_percentile_rank": avg_mpr,
            "avg_top10_recall": avg_top10,
            "avg_above_p90_ratio": avg_p90,
            "avg_zscore_improvement": avg_zscore,
        }
        
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {args.output_dir}/eval_results.json")
    else:
        print("No scenes evaluated!")


if __name__ == "__main__":
    main()
