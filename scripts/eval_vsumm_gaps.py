#!/usr/bin/env python3
"""
Evaluate VSUMM model and compute gap metrics.
Uses correct DSN model from ablation/pytorch-vsumm-reinforce
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import json
import pandas as pd
import scipy.linalg
from tqdm import tqdm

# Add ablation path
sys.path.insert(0, str(Path(__file__).parent.parent / "ablation" / "pytorch-vsumm-reinforce"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DSN
from src.datasets import load_scene_dir, build_epoch_index
from src.distance_selector.registry import create_metric
import eval.metrics as M
from src.rl.distribution_metrics import DistributionAwareMetrics, ATTR_NAMES


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


def temporal_coverage(selected_indices):
    """Compute temporal coverage (std dev of distances between frames)."""
    if len(selected_indices) < 2:
        return 0.0
    
    gaps = []
    sorted_idx = sorted(selected_indices)
    
    # Gaps between selections
    for i in range(len(sorted_idx) - 1):
        gaps.append(sorted_idx[i+1] - sorted_idx[i])
        
    return np.std(gaps) if gaps else 0.0


def dists_gap(all_frames, key_frames, device="cuda"):
    """Compute DISTS gap."""
    if not key_frames or not all_frames:
        return float("nan")
    
    try:
        metric = create_metric("dists", device=device)
        Ts_all = [metric.preprocess_bgr(f) for f in all_frames]
        Ts_keys = [metric.preprocess_bgr(f) for f in key_frames]
        
        gaps = []
        with torch.no_grad():
            for Ta in Ts_all:
                min_dist = 1e9
                for Tk in Ts_keys:
                    d = metric.pair_distance(Ta, Tk)
                    if d < min_dist:
                        min_dist = d
                gaps.append(min_dist)
        return float(np.mean(gaps)) if gaps else float("nan")
    except:
        return float("nan")


def eval_vsumm_comprehensive(model, scenes, device="cuda", budget_ratio=0.15, 
                             Bmin=3, Bmax=15, max_scenes=20, fast=False):
    """Evaluate VSUMM model."""
    
    model.eval()
    results = []
    
    # Init per-attribute
    metrics_computer = DistributionAwareMetrics()
    
    print(f"\nEvaluating VSUMM on {min(len(scenes), max_scenes)} scenes (Fast={fast})...")
    
    for idx, scene_dir in enumerate(tqdm(scenes[:max_scenes], desc="VSUMM")):
        try:
            sample = load_scene_dir(scene_dir, load_frames=True, load_anime_attrs=True)
            
            if sample.anime_attrs is None or not sample.frames:
                continue
            
            # VSUMM uses only CLIP features (512d)
            feats = torch.from_numpy(sample.feats).float().unsqueeze(0).to(device)
            T = len(sample.feats)
            budget = max(Bmin, min(Bmax, int(T * budget_ratio)))
            
            # Predict
            with torch.no_grad():
                probs = model(feats)  # (1, T, 1)
                probs = probs.squeeze().cpu().numpy()  # (T,)
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            key_frames = [sample.frames[i] for i in sel_idx if i < len(sample.frames)]
            
            # Sample all frames
            all_frames_sparse = sample.frames[::5]
            
            # LPIPS Gap
            lpips_gap_val = float("nan")
            if not fast:
                try:
                    lpips_metric = create_metric("lpips", net="alex", device=device)
                    Ts_all = [lpips_metric.preprocess_bgr(f) for f in all_frames_sparse]
                    Ts_keys = [lpips_metric.preprocess_bgr(f) for f in key_frames]
                    
                    gaps_lpips = []
                    with torch.no_grad():
                        for Ta in Ts_all:
                            min_dist = 1e9
                            for Tk in Ts_keys:
                                d = lpips_metric.pair_distance(Ta, Tk)
                                if d < min_dist:
                                    min_dist = d
                            gaps_lpips.append(min_dist)
                    lpips_gap_val = float(np.mean(gaps_lpips)) if gaps_lpips else float("nan")
                except Exception as e:
                    print(f"  LPIPS error: {e}")
            
            # DISTS Gap
            dists_gap_val = float("nan")
            if not fast:
                dists_gap_val = dists_gap(all_frames_sparse, key_frames, device=device)
            
            # Feature distance gap
            feat_gap = M.reconstruction_error(sample.feats, sample.feats[sel_idx])
            
            # Quality metrics
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            mpr = float(np.mean(percentiles[sel_idx]))
            
            # Per-Attribute
            per_attr = metrics_computer.compute_per_attribute_percentile(sample.anime_attrs, sel_idx)
            
            k10 = max(1, int(T * 0.1))
            top10_idx = set(np.argsort(quality)[-k10:])
            top10 = len(set(sel_idx) & top10_idx) / k10

            # Frechet
            frechet = compute_frechet(sample.feats, sel_idx)

            # Temp Coverage
            temp_cov = temporal_coverage(sel_idx)
            
            result = {
                "scene_id": idx + 1,
                "scene_name": scene_dir.name,
                "total_frames": T,
                "budget": budget,
                "lpips_gap": lpips_gap_val,
                "dists_gap": dists_gap_val,
                "feat_gap": feat_gap,
                "mpr": mpr,
                "top10": top10,
                "frechet": frechet,
                "temp_cov": temp_cov,
            }
            # Add per-attr to result dict
            for name in ATTR_NAMES:
                result[f"percentile_{name}"] = per_attr.get(f"percentile_{name}", 0.5)
                
            results.append(result)
            
        except Exception as e:
            print(f"Error on {scene_dir.name}: {e}")
            continue
    
    # Aggregate
    if not results:
        print("\n⚠️  No scenes successfully evaluated!")
        return None, None
    
    df = pd.DataFrame(results)
    
    summary = {
        "model": "VSUMM",
        "n_scenes": len(results),
        "lpips_gap_mean": float(df["lpips_gap"].mean()),
        "lpips_gap_std": float(df["lpips_gap"].std()),
        "dists_gap_mean": float(df["dists_gap"].mean()),
        "dists_gap_std": float(df["dists_gap"].std()),
        "feat_gap_mean": float(df["feat_gap"].mean()),
        "feat_gap_std": float(df["feat_gap"].std()),
        "mpr_mean": float(df["mpr"].mean()),
        "mpr_std": float(df["mpr"].std()),
        "top10_mean": float(df["top10"].mean()),
        "top10_std": float(df["top10"].std()),
        "frechet_mean": float(df["frechet"].mean()),
        "frechet_std": float(df["frechet"].std()),
        "temp_cov_mean": float(df["temp_cov"].mean()),
        "temp_cov_std": float(df["temp_cov"].std()),
    }
    
    for name in ATTR_NAMES:
        col = f"percentile_{name}"
        if col in df.columns:
            summary[col] = float(df[col].mean())
            summary[f"{col}_std"] = float(df[col].std())
    
    return summary, df


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsumm_checkpoint", type=str, required=True)
    parser.add_argument("--test_root", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test")
    parser.add_argument("--output", type=str, default="vsumm_gaps.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_scenes", type=int, default=20)
    parser.add_argument("--input_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--fast", action="store_true", help="Skip expensive metrics")
    
    args = parser.parse_args()
    
    # Load scenes
    all_scenes = build_epoch_index(args.test_root)
    print(f"Total scenes: {len(all_scenes)}")
    
    # Load VSUMM model
    print(f"\nLoading VSUMM from {args.vsumm_checkpoint}")
    model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=1, cell='lstm')
    
    checkpoint = torch.load(args.vsumm_checkpoint, map_location="cpu", weights_only=False)
    
    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Strip 'module.' prefix if present  
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(args.device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Evaluate
    summary, df = eval_vsumm_comprehensive(
        model, all_scenes, args.device, max_scenes=args.max_scenes, fast=args.fast
    )
    
    if summary is None:
        print("\n❌ Evaluation failed - no valid results")
        return
    
    # Save
    output_path = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new") / args.output
    
    # Merge if exists
    if args.fast and output_path.exists():
        print(f"Merging fast metrics into existing: {output_path}")
        try:
            with open(output_path, "r") as f:
                old_res = json.load(f)
            for k in ["lpips_gap_mean", "lpips_gap_std", "dists_gap_mean", "dists_gap_std"]:
                if k in old_res and (k not in summary or np.isnan(summary[k])):
                    summary[k] = old_res[k]
        except Exception as e:
            print(f"Merge error: {e}")
            
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "="*70)
    print("VSUMM EVALUATION RESULTS")
    print("="*70)
    print(f"Scenes:      {summary['n_scenes']}")
    print(f"LPIPS Gap:   {summary['lpips_gap_mean']:.4f} ± {summary['lpips_gap_std']:.4f}")
    print(f"DISTS Gap:   {summary['dists_gap_mean']:.4f} ± {summary['dists_gap_std']:.4f}")
    print(f"Feat Gap:    {summary['feat_gap_mean']:.4f} ± {summary['feat_gap_std']:.4f}")
    print(f"Frechet:     {summary['frechet_mean']:.4f} ± {summary['frechet_std']:.4f}")
    print(f"MPR:         {summary['mpr_mean']:.4f} ± {summary['mpr_std']:.4f}")
    print(f"Top10:       {summary['top10_mean']:.4f} ± {summary['top10_std']:.4f}")
    print(f"Temp Cov:    {summary['temp_cov_mean']:.4f} ± {summary['temp_cov_std']:.4f}")
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

"""
python3 scripts/eval_vsumm_gaps.py --vsumm_checkpoint /home/serverai/ltdoanh/LayoutGeneration/runs/ablation_vsumm/sakuga_train/model_epoch60.pth.tar --max_scenes 50
"""