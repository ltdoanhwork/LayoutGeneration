#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 Comprehensive Evaluation

Evaluates model on both training and test data.
Computes both local (per-scene) and global metrics.
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
from src.models.dsn_v8 import create_dsn_v8


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = ckpt.get("model_state_dict", ckpt)
    if "input_proj.0.weight" in state_dict:
        feat_dim = state_dict["input_proj.0.weight"].shape[1]
    else:
        feat_dim = 518
    
    model = create_dsn_v8(feat_dim=feat_dim, use_pcgrad=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, feat_dim


def compute_metrics(
    quality: np.ndarray,
    sel_idx: List[int],
) -> Dict[str, float]:
    """Compute quality metrics for selected frames."""
    T = len(quality)
    if len(sel_idx) == 0:
        return {"mpr": 0.5, "top10": 0.0, "above_p90": 0.0, "zscore": 0.0}
    
    # Percentile ranks
    ranks = np.argsort(np.argsort(quality))
    percentiles = ranks / max(1, T - 1)
    
    mpr = float(np.mean(percentiles[sel_idx]))
    
    # Top-10% recall
    k10 = max(1, int(T * 0.1))
    top10_idx = set(np.argsort(quality)[-k10:])
    top10 = len(set(sel_idx) & top10_idx) / k10
    
    # Above P90
    p90 = np.percentile(quality, 90)
    above_p90 = sum(1 for i in sel_idx if quality[i] >= p90) / len(sel_idx)
    
    # Z-score
    all_mean = np.mean(quality)
    all_std = np.std(quality) + 1e-8
    sel_mean = np.mean(quality[sel_idx])
    zscore = (sel_mean - all_mean) / all_std
    
    return {
        "mpr": mpr,
        "top10": top10,
        "above_p90": above_p90,
        "zscore": zscore,
    }


def compute_diversity_metrics(sel_idx: List[int], T: int) -> Dict[str, float]:
    """Compute temporal diversity metrics."""
    if len(sel_idx) < 2:
        return {"min_gap": 0.0, "mean_gap": 0.0, "coverage": 0.0}
    
    sorted_idx = sorted(sel_idx)
    gaps = np.diff(sorted_idx)
    
    min_gap = float(np.min(gaps))
    mean_gap = float(np.mean(gaps))
    coverage = (max(sorted_idx) - min(sorted_idx)) / max(1, T - 1)
    
    return {
        "min_gap": min_gap,
        "mean_gap": mean_gap,
        "coverage": coverage,
    }


def evaluate_scene(
    model: torch.nn.Module,
    feats: np.ndarray,
    anime_attrs: np.ndarray,
    budget_ratio: float,
    Bmin: int,
    Bmax: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on a single scene."""
    T = len(feats)
    budget = max(Bmin, min(Bmax, int(T * budget_ratio)))
    
    feats_full = np.concatenate([feats, anime_attrs], axis=1)
    feats_t = torch.from_numpy(feats_full).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs, _ = model(feats_t)
        probs = probs.squeeze(0).cpu().numpy()
    
    sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
    
    quality = anime_attrs.mean(axis=1)
    
    metrics = compute_metrics(quality, sel_idx)
    div_metrics = compute_diversity_metrics(sel_idx, T)
    metrics.update(div_metrics)
    metrics["n_selected"] = len(sel_idx)
    metrics["n_total"] = T
    
    return metrics


def evaluate_dataset(
    model: torch.nn.Module,
    dataset_dir: str,
    budget_ratio: float,
    Bmin: int,
    Bmax: int,
    device: str,
    max_scenes: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate on entire dataset."""
    scene_dirs = build_epoch_index(dataset_dir)
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]
    
    all_metrics = []
    for scene_dir in tqdm(scene_dirs, desc="Evaluating"):
        sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=True)
        if sample.anime_attrs is None:
            continue
        
        metrics = evaluate_scene(
            model, sample.feats, sample.anime_attrs,
            budget_ratio, Bmin, Bmax, device
        )
        all_metrics.append(metrics)
    
    if not all_metrics:
        return {}
    
    # Aggregate
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}


def main():
    parser = argparse.ArgumentParser(description="V11 Comprehensive Eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--budget_ratio", type=float, default=0.15)
    parser.add_argument("--Bmin", type=int, default=3)
    parser.add_argument("--Bmax", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_scenes", type=int, default=None)
    
    args = parser.parse_args()
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    model, feat_dim = load_model(args.checkpoint, args.device)
    print(f"Loaded model from {args.checkpoint} (feat_dim={feat_dim})")
    
    results = {"checkpoint": args.checkpoint}
    
    if args.train_dir:
        print(f"\n📊 Evaluating on TRAINING data: {args.train_dir}")
        train_metrics = evaluate_dataset(
            model, args.train_dir, args.budget_ratio, 
            args.Bmin, args.Bmax, args.device, args.max_scenes
        )
        results["train"] = train_metrics
        
        print(f"  MPR:      {train_metrics['mpr']:.4f}")
        print(f"  Top-10%:  {train_metrics['top10']:.4f}")
        print(f"  Above P90:{train_metrics['above_p90']:.4f}")
        print(f"  Min Gap:  {train_metrics['min_gap']:.1f}")
        print(f"  Coverage: {train_metrics['coverage']:.4f}")
    
    if args.test_dir:
        print(f"\n📊 Evaluating on TEST data: {args.test_dir}")
        test_metrics = evaluate_dataset(
            model, args.test_dir, args.budget_ratio,
            args.Bmin, args.Bmax, args.device, args.max_scenes
        )
        results["test"] = test_metrics
        
        print(f"  MPR:      {test_metrics['mpr']:.4f}")
        print(f"  Top-10%:  {test_metrics['top10']:.4f}")
        print(f"  Above P90:{test_metrics['above_p90']:.4f}")
        print(f"  Min Gap:  {test_metrics['min_gap']:.1f}")
        print(f"  Coverage: {test_metrics['coverage']:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    if "train" in results:
        print(f"  Train: MPR={results['train']['mpr']:.4f}, Top10={results['train']['top10']:.3f}")
    if "test" in results:
        print(f"  Test:  MPR={results['test']['mpr']:.4f}, Top10={results['test']['top10']:.3f}")
    print("="*60)
    
    if args.output_dir:
        output_file = os.path.join(args.output_dir, "eval_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()


"""
Usage:
python -m eval.eval_v11_comprehensive \
    --checkpoint runs/dsn_v11/best.pt \
    --train_dir data/sakuga_dataset_100_samples \
    --test_dir data/sakuga_test_scenes \
    --output_dir runs/dsn_v11/eval
"""
