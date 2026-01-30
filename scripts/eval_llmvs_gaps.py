#!/usr/bin/env python3
"""
Single-purpose script to evaluate Gaps (Feature, LPIPS, DISTS) and MPR/Top10 for LLMVS.
Matches the format of eval_v11_gaps.py and eval_vsumm_gaps.py.
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import scipy.linalg
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablation" / "LLMVS"))

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
    parser.add_argument("--llmvs_checkpoint", type=str, required=True,
                       help="Path to LLMVS checkpoint (for Option B: LLMVSVisual)")
    parser.add_argument("--test_root", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test")
    parser.add_argument("--output", type=str, default="llmvs_gaps.json")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_llmvs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_scenes", type=int, default=50, help="Max scenes for LPIPS/DISTS (expensive)")
    parser.add_argument("--input_dim", type=int, default=512)
    parser.add_argument("--reduced_dim", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fast", action="store_true", help="Skip expensive metrics")
    
    args = parser.parse_args()
    
    # Import LLMVS modules
    from networks.model_visual import LLMVSVisual
    from llmvs_utils.configs import Config
    
    # Load scenes
    all_scenes = build_epoch_index(args.test_root)
    test_scenes = all_scenes[:args.max_scenes]
    
    print(f"Evaluating LLMVS on {len(test_scenes)} scenes...")
    
    # Load Model (Option B: LLMVSVisual)
    print(f"Loading checkpoint: {args.llmvs_checkpoint}")
    
    config = Config(
        reduced_dim=args.reduced_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dataset='sakuga',
        input_dim=args.input_dim,
        lr=1e-4,
        model='LLMVSVisual',
        tag='eval_gaps'
    )
    
    try:
        print(f"Attempting to load as PL checkpoint...")
        model = LLMVSVisual.load_from_checkpoint(args.llmvs_checkpoint, config=config, strict=False)
    except Exception as e:
        print(f"PL load failed ({e}), attempting manual state_dict load...")
        model = LLMVSVisual(config)
        checkpoint = torch.load(args.llmvs_checkpoint, map_location="cpu")
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Remove module. prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        # Handle PL prefix if we are loading into fresh model
        # PL usually stores keys as is, but if trained with DDP might have others.
        # But here we are loading INTO a PL module manually.
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model = model.to(args.device)
    model.eval()
    
    print("Model loaded successfully!")
    
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
    
    for scene_dir in tqdm(test_scenes, desc="LLMVS Eval"):
        try:
            # Load
            sample = load_scene_dir(scene_dir, load_frames=True, load_anime_attrs=True)
            if sample.anime_attrs is None or len(sample.frames) == 0:
                continue
            
            # Prepare features for LLMVS (uses CLIP features 512d)
            feats_input = sample.feats
            feats_t = torch.from_numpy(feats_input).float().unsqueeze(0).to(args.device)
            T = len(sample.feats)
            budget = max(3, min(15, int(T * 0.15)))
            
            # Predict
            with torch.no_grad():
                scores = model(feats_t)  # (1, T, 1) or (1, T)
                scores = scores.squeeze().cpu().numpy()
            
            sel_idx = sorted(np.argsort(scores)[-budget:].tolist())
            
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
        "model": "LLMVS",
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
    
    print("\n" + "="*70)
    print("LLMVS EVALUATION RESULTS")
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
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / args.output
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

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {out_path}")

if __name__ == "__main__":
    main()

"""
Example usage:
python3 scripts/eval_llmvs_gaps.py --llmvs_checkpoint /home/serverai/ltdoanh/LayoutGeneration/runs/ablation_llmvs/optionB_visual/best_model.pth --max_scenes 50
"""
