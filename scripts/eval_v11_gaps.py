#! /usr/bin/env python3
"""
Single-purpose script to evaluate Gaps (Feature, LPIPS, DISTS) and MPR/Top10 for V11 Final.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import scipy.linalg
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dsn_v8 import create_dsn_v8
from src.datasets import load_scene_dir, build_epoch_index
from src.distance_selector.registry import create_metric
from src.rl.distribution_metrics import DistributionAwareMetrics, ATTR_NAMES


def feature_distance_gap(features_all, selected_indices):
    """Compute mean minimum cosine distance."""
    if not selected_indices or len(features_all) == 0:
        return float("nan")
    
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
    
    # Gaps between selections
    for i in range(len(sorted_idx) - 1):
        gaps.append(sorted_idx[i+1] - sorted_idx[i])
        
    return np.std(gaps) if gaps else 0.0


def perceptual_gap_from_frames(metric, all_frames, selected_indices):
    """
    Compute perceptual gap (LPIPS or DISTS): mean minimum distance.
    """
    if not selected_indices or not all_frames:
        return float("nan")
    
    # Preprocess frames
    Ts_all = [metric.preprocess_bgr(f) for f in all_frames]
    Ts_sel = [Ts_all[i] for i in selected_indices if i < len(Ts_all)]
    
    if not Ts_sel:
        return float("nan")
    
    # Compute min distance for each frame
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
    parser.add_argument("--v11_checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_recerr_w0.2/ep59/checkpoint.pt")
    parser.add_argument("--test_root", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test")
    parser.add_argument("--output", type=str, default="v11_gaps.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_scenes", type=int, default=474, help="Max scenes for LPIPS/DISTS (expensive)")
    parser.add_argument("--fast", action="store_true", help="Skip expensive metrics (LPIPS/DISTS)")
    
    args = parser.parse_args()
    
    # Load scenes
    all_scenes = build_epoch_index(args.test_root)
    # Check if we should shuffle or just take first N
    # For fair comparison with VSUMM script, we hopefully use same set or randomize same seed.
    # The eval_vsumm_gaps.py likely used build_epoch_index order.
    test_scenes = all_scenes[:args.max_scenes]
    
    print(f"Evaluating V11 on {len(test_scenes)} scenes...")
    
    # Load Model
    print(f"Loading checkpoint: {args.v11_checkpoint}")
    ckpt = torch.load(args.v11_checkpoint, map_location="cpu")
    config = ckpt.get("config", {})
    
    feat_dim = config.get("feat_dim", 512)
    # V11 typically uses generic/anime attributes
    # Check config to be sure
    # In training_v11_final_new, we used use_anime_attrs=True (so +6)
    use_anime = True 
    # But let's verify from config if possible, or assume based on training script
    if "no_anime_attrs" in config and config["no_anime_attrs"]:
        use_anime = False
    
    full_feat_dim = feat_dim + (6 if use_anime else 0)
    
    model = create_dsn_v8(
        feat_dim=full_feat_dim,
        use_pcgrad=False,
        num_attn_layers=config.get("num_attn_layers", 2),
        gating_hidden=config.get("gating_hidden", 64),
        lstm_hidden=config.get("lstm_hidden", 128),
    ).to(args.device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Metrics
    # Metrics
    metrics_computer = DistributionAwareMetrics()
    lpips_metric = None
    dists_metric = None
    
    if not args.fast:
        print("Setting up LPIPS...")
        lpips_metric = create_metric("lpips", net="alex", device=args.device)
        print("Setting up DISTS...")
        dists_metric = create_metric("dists", device=args.device)
    
    feat_gaps = []
    lpips_gaps = []
    dists_gaps = []
    mprs = []
    top10s = []
    frechets = []
    temp_covs = []
    per_attr_lists = {name: [] for name in ATTR_NAMES}
    
    for scene_dir in tqdm(test_scenes, desc="V11 Eval"):
        try:
            # Load
            sample = load_scene_dir(scene_dir, load_frames=not args.fast, load_anime_attrs=True)
            if sample.anime_attrs is None:
                continue
            # If frames not loaded, sample.frames is empty or None
            if not args.fast and (not sample.frames or len(sample.frames) == 0):
                continue
            
            # Predict
            if use_anime:
                feats_input = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
            else:
                feats_input = sample.feats
            
            feats_t = torch.from_numpy(feats_input).float().unsqueeze(0).to(args.device)
            T = len(sample.feats)
            budget = max(3, min(15, int(T * 0.15)))
            
            with torch.no_grad():
                probs, _ = model(feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
            # 1. Feature Gap
            fg = feature_distance_gap(sample.feats, sel_idx)
            if not np.isnan(fg): feat_gaps.append(fg)
            
            # 2. LPIPS Gap
            if not args.fast and lpips_metric:
                lg = perceptual_gap_from_frames(lpips_metric, sample.frames, sel_idx)
                if not np.isnan(lg): lpips_gaps.append(lg)
            
            # 3. DISTS Gap
            if not args.fast and dists_metric:
                dg = perceptual_gap_from_frames(dists_metric, sample.frames, sel_idx)
                if not np.isnan(dg): dists_gaps.append(dg)
            
            # 4. MPR & Top10
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            
            mpr = float(np.mean(percentiles[sel_idx]))
            mprs.append(mpr)
            
            # Per-Attribute percentiles
            per_attr = metrics_computer.compute_per_attribute_percentile(sample.anime_attrs, sel_idx)
            for name in ATTR_NAMES:
                per_attr_lists[name].append(per_attr.get(f"percentile_{name}", 0.5))
            
            k10 = max(1, int(T * 0.1))
            top10_idx = set(np.argsort(quality)[-k10:])
            top10 = len(set(sel_idx) & top10_idx) / k10
            top10s.append(top10)

            # 5. Frechet Distance
            frechet = compute_frechet(sample.feats, sel_idx)
            if not np.isnan(frechet): frechets.append(frechet)

            # 6. Temporal Coverage
            temp_cov = temporal_coverage(sel_idx, T)
            temp_covs.append(temp_cov)
            
        except Exception as e:
            # print(f"Error: {e}")
            continue
    
    results = {
        "n_scenes": len(feat_gaps),
        "feat_gap_mean": float(np.mean(feat_gaps)),
        "feat_gap_std": float(np.std(feat_gaps)),
        "lpips_gap_mean": float(np.mean(lpips_gaps)),
        "lpips_gap_std": float(np.std(lpips_gaps)),
        "dists_gap_mean": float(np.mean(dists_gaps)),
        "dists_gap_std": float(np.std(dists_gaps)),
        "mpr_mean": float(np.mean(mprs)),
        "mpr_std": float(np.std(mprs)),
        "top10_mean": float(np.mean(top10s)),
        "top10_std": float(np.std(top10s)),
        "frechet_mean": float(np.mean(frechets)),
        "frechet_std": float(np.std(frechets)),
        "temp_cov_mean": float(np.mean(temp_covs)),
        "temp_cov_std": float(np.std(temp_covs)),
    }
    
    # Add per-attribute results
    for name in ATTR_NAMES:
        vals = per_attr_lists[name]
        results[f"percentile_{name}"] = float(np.mean(vals)) if vals else 0.5
        results[f"percentile_{name}_std"] = float(np.std(vals)) if vals else 0.0

    # Save logic
    ckpt_path = Path(args.v11_checkpoint)
    if ckpt_path.parent.name.startswith("ep"):
        save_dir = ckpt_path.parent.parent
    else:
        save_dir = ckpt_path.parent
    out_path = save_dir / args.output
    
    # Merge if exists
    if args.fast and out_path.exists():
        print(f"Merging fast metrics into existing: {out_path}")
        try:
            with open(out_path, "r") as f:
                old_res = json.load(f)
            # Retain LPIPS/DISTS from old if missing in new
            for k in ["lpips_gap_mean", "lpips_gap_std", "dists_gap_mean", "dists_gap_std"]:
                if k in old_res and (k not in results or np.isnan(results[k]) or results[k]==0):
                    results[k] = old_res[k]
        except Exception as e:
            print(f"Merge failed: {e}")
    
    print("\n" + "="*70)
    print("V11 EVALUATION RESULTS")
    print("="*70)
    print(f"Scenes:      {results['n_scenes']}")
    print(f"LPIPS Gap:   {results['lpips_gap_mean']:.4f} ± {results['lpips_gap_std']:.4f}")
    print(f"DISTS Gap:   {results['dists_gap_mean']:.4f} ± {results['dists_gap_std']:.4f}")
    print(f"Feat Gap:    {results['feat_gap_mean']:.4f} ± {results['feat_gap_std']:.4f}")
    print(f"Frechet:     {results['frechet_mean']:.4f} ± {results['frechet_std']:.4f}")
    print(f"MPR:         {results['mpr_mean']:.4f} ± {results['mpr_std']:.4f}")
    print(f"Top10:       {results['top10_mean']:.4f} ± {results['top10_std']:.4f}")
    print(f"Temp Cov:    {results['temp_cov_mean']:.4f} ± {results['temp_cov_std']:.4f}")
    
    # Save
    # Save to checkpoint directory (grandparent of epXX/checkpoint.pt)
    ckpt_path = Path(args.v11_checkpoint)
    # Check if checkpoint is in "epXX" folder
    if ckpt_path.parent.name.startswith("ep"):
        save_dir = ckpt_path.parent.parent
    else:
        save_dir = ckpt_path.parent
        
    out_path = save_dir / args.output
    
    # If using default hardcoded in argparse, args.v11_checkpoint might be "w0.2" but we might want to respect that.
    # But strictly, saving next to checkpoint is safest.
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {out_path}")

if __name__ == "__main__":
    main()
