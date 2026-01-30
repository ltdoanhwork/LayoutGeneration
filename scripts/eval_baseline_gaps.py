#!/usr/bin/env python3
"""
Quick Feature Gap evaluation for VSUMM to compare with V11 Final.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import load_scene_dir, build_epoch_index
from src.distance_selector.registry import create_metric
from scipy.spatial.distance import cdist
import torch
import scipy.linalg


def feature_distance_gap(features_all: np.ndarray, selected_indices: list) -> float:
    """Compute mean minimum distance from all frames to selected frames."""
    if not selected_indices or len(features_all) == 0:
        return float("nan")
    
    # L2 normalize
    feats_all = features_all / (np.linalg.norm(features_all, axis=1, keepdims=True) + 1e-12)
    feats_sel = feats_all[selected_indices]
    
    gaps = []
    for i in range(len(feats_all)):
        similarities = feats_sel @ feats_all[i]
        min_distance = 1.0 - np.max(similarities)
        gaps.append(min_distance)
    
    return float(np.mean(gaps))

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    try:
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
             offset = np.eye(sigma1.shape[0]) * eps
             covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    except:
        return float("nan")

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def compute_frechet(features_all, selected_indices):
    if not selected_indices or len(features_all) < 2:
        return float("nan")
    
    feats_sel = features_all[selected_indices]
    
    mu1 = np.mean(features_all, axis=0)
    sigma1 = np.cov(features_all, rowvar=False)
    
    mu2 = np.mean(feats_sel, axis=0)
    sigma2 = np.cov(feats_sel, rowvar=False)
    
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def temporal_coverage(selected_indices, T):
    """Compute temporal coverage (std dev of distances between frames)."""
    if len(selected_indices) < 2:
        return 0.0
    
    gaps = []
    sorted_idx = sorted(selected_indices)
    
    # Gap from 0 to first
    # gaps.append(sorted_idx[0]) 
    
    # Gaps between selections
    for i in range(len(sorted_idx) - 1):
        gaps.append(sorted_idx[i+1] - sorted_idx[i])
        
    # Gap from last to end? Typically we just measure uniformity of spacing
    # Ideally, for uniform coverage, gaps should be low std dev.
    # Metric: Coefficient of Variation of gaps? Or just std?
    # Common metric: Std of gaps
    return np.std(gaps) if gaps else 0.0


def perceptual_gap(metric, all_frames, selected_indices):
    """Compute perceptual gap (LPIPS/DISTS)."""
    if not selected_indices or not all_frames:
        return float("nan")
    
    Ts_all = [metric.preprocess_bgr(f) for f in all_frames]
    Ts_sel = [Ts_all[i] for i in selected_indices if i < len(Ts_all)]
    
    if not Ts_sel:
        return float("nan")
    
    gaps = []
    with torch.no_grad():
        for Ta in Ts_all:
            min_dist = 1e9
            for Ts in Ts_sel:
                d = metric.pair_distance(Ta, Ts)
                if d < min_dist:
                    min_dist = d
            gaps.append(min_dist)
            
    return float(np.mean(gaps)) if gaps else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, default="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test")
    parser.add_argument("--max_scenes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="baseline_comprehensive_metrics.json")
    args = parser.parse_args()
    
    val_scenes = build_epoch_index(args.test_root)
    test_scenes = val_scenes[:args.max_scenes]
    
    print(f"Evaluating baselines on {len(test_scenes)} scenes...")
    
    # metrics
    print("Loading LPIPS...")
    lpips_metric = create_metric("lpips", net="alex", device=args.device)
    print("Loading DISTS...")
    dists_metric = create_metric("dists", device=args.device)
    
    baselines = ["random", "uniform"]
    stats = {
        name: {
            "rec_err": [], "frechet": [], "lpips_gap": [], "dists_gap": [],
            "mpr": [], "top10": [], "temp_cov": []
        }
        for name in baselines
    }
    
    for scene_dir in tqdm(test_scenes, desc="Baseline Eval"):
        try:
            sample = load_scene_dir(scene_dir, load_frames=True, load_anime_attrs=True)
            if sample.anime_attrs is None or not sample.frames:
                continue
                
            T = len(sample.feats)
            budget = max(3, min(15, int(T * 0.15)))
            
            selections = {
                "random": sorted(np.random.choice(T, min(budget, T), replace=False).tolist()),
                "uniform": sorted(set(np.linspace(0, T - 1, budget, dtype=int).tolist()))
            }
            
            # Pre-compute quality derived metrics (MPR, Top10)
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            k10 = max(1, int(T * 0.1))
            top10_idx = set(np.argsort(quality)[-k10:])
            
            for name, sel_idx in selections.items():
                if not sel_idx: continue
                
                # 1. RecErr / Feature Gap
                stats[name]["rec_err"].append(feature_distance_gap(sample.feats, sel_idx))
                
                # 2. Frechet
                stats[name]["frechet"].append(compute_frechet(sample.feats, sel_idx))
                
                # 3. LPIPS Gap
                stats[name]["lpips_gap"].append(perceptual_gap(lpips_metric, sample.frames, sel_idx))
                
                # 4. DISTS Gap
                stats[name]["dists_gap"].append(perceptual_gap(dists_metric, sample.frames, sel_idx))
                
                # 5. MPR
                stats[name]["mpr"].append(float(np.mean(percentiles[sel_idx])))
                
                # 6. Top10
                stats[name]["top10"].append(len(set(sel_idx) & top10_idx) / k10)
                
                # 7. Temporal Coverage
                stats[name]["temp_cov"].append(temporal_coverage(sel_idx, T))
                
        except Exception as e:
            # print(f"Error: {e}")
            continue
    
    # Aggregation
    results = {}
    for name in baselines:
        results[name] = {}
        for metric, values in stats[name].items():
            valid = [v for v in values if not np.isnan(v)]
            results[name][f"{metric}_mean"] = float(np.mean(valid)) if valid else float("nan")
            results[name][f"{metric}_std"] = float(np.std(valid)) if valid else float("nan")
            
    # Print Table
    print("\n" + "="*100)
    print(f"{'Metric':<15} | {'Random':<35} | {'Uniform':<35}")
    print("-" * 100)
    
    metrics_map = [
        ("rec_err", "RecErr/FeatGap"),
        ("frechet", "Frechet"),
        ("mpr", "MPR (Aesthetic)"),
        ("top10", "Top10 (Aesthetic)"),
        ("lpips_gap", "LPIPS Gap"),
        ("dists_gap", "DISTS Gap"),
        ("temp_cov", "Temp Coverage")
    ]
    
    for key, label in metrics_map:
        r_mean = results["random"].get(f"{key}_mean", float("nan"))
        r_std = results["random"].get(f"{key}_std", float("nan"))
        u_mean = results["uniform"].get(f"{key}_mean", float("nan"))
        u_std = results["uniform"].get(f"{key}_std", float("nan"))
        
        print(f"{label:<15} | {r_mean:8.4f} ± {r_std:6.4f}             | {u_mean:8.4f} ± {u_std:6.4f}")
        
    # Save
    out_path = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new") / args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
