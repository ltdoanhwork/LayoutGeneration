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
from tqdm import tqdm

# Add ablation path
sys.path.insert(0, str(Path(__file__).parent.parent / "ablation" / "pytorch-vsumm-reinforce"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DSN
from src.datasets import load_scene_dir, build_epoch_index
from src.distance_selector.registry import create_metric
import eval.metrics as M


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
                             Bmin=3, Bmax=15, max_scenes=20):
    """Evaluate VSUMM model."""
    
    model.eval()
    results = []
    
    print(f"\nEvaluating VSUMM on {min(len(scenes), max_scenes)} scenes...")
    
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
            dists_gap_val = dists_gap(all_frames_sparse, key_frames, device=device)
            
            # Feature distance gap
            feat_gap = M.reconstruction_error(sample.feats, sample.feats[sel_idx])
            
            # Quality metrics
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            mpr = float(np.mean(percentiles[sel_idx]))
            
            k10 = max(1, int(T * 0.1))
            top10_idx = set(np.argsort(quality)[-k10:])
            top10 = len(set(sel_idx) & top10_idx) / k10
            
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
            }
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
    }
    
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
        model, all_scenes, args.device, max_scenes=args.max_scenes
    )
    
    if summary is None:
        print("\n❌ Evaluation failed - no valid results")
        return
    
    # Save
    output_path = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new") / args.output
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
    print(f"MPR:         {summary['mpr_mean']:.4f} ± {summary['mpr_std']:.4f}")
    print(f"Top10:       {summary['top10_mean']:.4f} ± {summary['top10_std']:.4f}")
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
