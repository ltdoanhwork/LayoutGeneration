#!/usr/bin/env python3
"""
Comprehensive gap evaluation with LPIPS for V11 and VSUMM.
Computes both feature distance gap and LPIPS gap.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dsn_v8 import create_dsn_v8
from src.datasets import load_scene_dir, build_epoch_index
from src.distance_selector.registry import create_metric


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


def lpips_gap_from_frames(all_frames, selected_indices, device="cuda", net="alex"):
    """
    Compute LPIPS gap: mean minimum LPIPS distance from all frames to selected keyframes.
    """
    if not selected_indices or not all_frames:
        return float("nan")
    
    try:
        lpips_metric = create_metric("lpips", net=net, device=device)
    except:
        return float("nan")
    
    # Preprocess frames
    Ts_all = [lpips_metric.preprocess_bgr(f) for f in all_frames]
    Ts_sel = [Ts_all[i] for i in selected_indices if i < len(Ts_all)]
    
    if not Ts_sel:
        return float("nan")
    
    # Compute min distance for each frame
    gaps = []
    with torch.no_grad():
        for Ta in Ts_all:
            min_dist = 1e9
            for Ts in Ts_sel:
                d = lpips_metric.pair_distance(Ta, Ts)
                if d < min_dist:
                    min_dist = d
            gaps.append(min_dist)
    
    return float(np.mean(gaps)) if gaps else float("nan")


def evaluate_model(model, scenes, device="cuda", budget_ratio=0.15, 
                   Bmin=3, Bmax=15, use_anime=True, compute_lpips=True):
    """Evaluate model on scenes."""
    
    model.eval()
    
    feat_gaps = []
    lpips_gaps = []
    mprs = []
    top10s = []
    
    for scene_dir in tqdm(scenes, desc="Evaluating"):
        try:
            # Load with frames for LPIPS
            sample = load_scene_dir(scene_dir, load_frames=compute_lpips, load_anime_attrs=True)
            
            if sample.anime_attrs is None:
                continue
            
            # Construct features
            if use_anime:
                feats_input = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
            else:
                feats_input = sample.feats
            
            feats_t = torch.from_numpy(feats_input).float().unsqueeze(0).to(device)
            T = len(sample.feats)
            budget = max(Bmin, min(Bmax, int(T * budget_ratio)))
            
            # Predict
            with torch.no_grad():
                probs, _ = model(feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
            # Feature distance gap
            feat_gap = feature_distance_gap(sample.feats, sel_idx)
            if not np.isnan(feat_gap):
                feat_gaps.append(feat_gap)
            
            # LPIPS gap (if frames available)
            if compute_lpips and sample.frames:
                lpips_gap = lpips_gap_from_frames(sample.frames, sel_idx, device=device)
                if not np.isnan(lpips_gap):
                    lpips_gaps.append(lpips_gap)
            
            # Quality metrics
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            mpr = float(np.mean(percentiles[sel_idx]))
            mprs.append(mpr)
            
            k10 = max(1, int(T * 0.1))
            top10_idx = set(np.argsort(quality)[-k10:])
            top10 = len(set(sel_idx) & top10_idx) / k10
            top10s.append(top10)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    results = {
        "feat_gap_mean": float(np.mean(feat_gaps)) if feat_gaps else float("nan"),
        "feat_gap_std": float(np.std(feat_gaps)) if feat_gaps else float("nan"),
        "lpips_gap_mean": float(np.mean(lpips_gaps)) if lpips_gaps else float("nan"),
        "lpips_gap_std": float(np.std(lpips_gaps)) if lpips_gaps else float("nan"),
        "mpr": float(np.mean(mprs)) if mprs else float("nan"),
        "top10": float(np.mean(top10s)) if top10s else float("nan"),
        "n_scenes": len(feat_gaps),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v11_checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new/best.pt")
    parser.add_argument("--vsumm_checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_vsumm/sakuga_train/best.pt")
    parser.add_argument("--test_root", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test")
    parser.add_argument("--output", type=str, default="comprehensive_gaps.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_scenes", type=int, default=50, help="Max scenes for LPIPS (expensive)")
    
    args = parser.parse_args()
    
    # Load scenes
    all_scenes = build_epoch_index(args.test_root)
    test_scenes = all_scenes[:args.max_scenes]  # Limit for LPIPS
    
    print(f"Evaluating on {len(test_scenes)} scenes (LPIPS enabled)")
    
    all_results = {}
    
    # Evaluate V11
    if Path(args.v11_checkpoint).exists():
        print("\n" + "="*60)
        print("Evaluating V11")
        print("="*60)
        
        ckpt = torch.load(args.v11_checkpoint, map_location="cpu")
        config = ckpt.get("config", {})
        
        feat_dim = config.get("feat_dim", 512)
        use_anime = not config.get("no_anime_attrs", False)
        full_feat_dim = feat_dim + (6 if use_anime else 0)
        
        model = create_dsn_v8(
            feat_dim=full_feat_dim,
            use_pcgrad=False,
            num_attn_layers=config.get("num_attn_layers", 2),
            gating_hidden=config.get("gating_hidden", 64),
            lstm_hidden=config.get("lstm_hidden", 128),
        ).to(args.device)
        
        model.load_state_dict(ckpt["model_state_dict"])
        
        v11_results = evaluate_model(model, test_scenes, args.device, 
                                     use_anime=use_anime, compute_lpips=True)
        all_results["V11"] = v11_results
        
        print(f"\nV11 Results:")
        print(f"  Feat Gap:  {v11_results['feat_gap_mean']:.4f} ± {v11_results['feat_gap_std']:.4f}")
        print(f"  LPIPS Gap: {v11_results['lpips_gap_mean']:.4f} ± {v11_results['lpips_gap_std']:.4f}")
        print(f"  MPR:       {v11_results['mpr']:.4f}")
        print(f"  Top10:     {v11_results['top10']:.4f}")
    
    # Evaluate VSUMM
    if Path(args.vsumm_checkpoint).exists():
        print("\n" + "="*60)
        print("Evaluating VSUMM")
        print("="*60)
        
        ckpt = torch.load(args.vsumm_checkpoint, map_location="cpu")
        config = ckpt.get("config", {})
        
        feat_dim = config.get("feat_dim", 512)
        full_feat_dim = feat_dim  # VSUMM doesn't use anime attrs
        
        model = create_dsn_v8(
            feat_dim=full_feat_dim,
            use_pcgrad=False,
            num_attn_layers=config.get("num_attn_layers", 2),
            gating_hidden=config.get("gating_hidden", 64),
            lstm_hidden=config.get("lstm_hidden", 128),
        ).to(args.device)
        
        model.load_state_dict(ckpt["model_state_dict"])
        
        vsumm_results = evaluate_model(model, test_scenes, args.device,
                                       use_anime=False, compute_lpips=True)
        all_results["VSUMM"] = vsumm_results
        
        print(f"\nVSUMM Results:")
        print(f"  Feat Gap:  {vsumm_results['feat_gap_mean']:.4f} ± {vsumm_results['feat_gap_std']:.4f}")
        print(f"  LPIPS Gap: {vsumm_results['lpips_gap_mean']:.4f} ± {vsumm_results['lpips_gap_std']:.4f}")
        print(f"  MPR:       {vsumm_results['mpr']:.4f}")
        print(f"  Top10:     {vsumm_results['top10']:.4f}")
    
    # Save
    output_path = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new") / args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<15} {'FeatGap':>12} {'LPIPS Gap':>12} {'MPR':>10} {'Top10':>10}")
    print("-"*70)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<15} {results['feat_gap_mean']:>12.4f} "
              f"{results['lpips_gap_mean']:>12.4f} {results['mpr']:>10.4f} "
              f"{results['top10']:>10.4f}")


if __name__ == "__main__":
    main()
