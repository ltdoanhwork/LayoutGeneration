#!/usr/bin/env python3
"""
Comprehensive comparison V11 vs VSUMM using eval/metrics.py
Computes LPIPS Gap, DISTS Gap, and quality metrics.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dsn_v8 import create_dsn_v8
from src.datasets import load_scene_dir, build_epoch_index
from src.distance_selector.registry import create_metric
import eval.metrics as M


def dists_gap(all_frames, key_frames, device="cuda"):
    """Compute DISTS gap using DISTS metric."""
    if not key_frames or not all_frames:
        return float("nan")
    
    try:
        metric = create_metric("dists", device=device)
    except:
        return float("nan")
    
    # Preprocess
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


def eval_model_comprehensive(model, scenes, model_name, device="cuda", 
                             budget_ratio=0.15, Bmin=3, Bmax=15, 
                             use_anime=True, max_scenes=20):
    """
    Comprehensive evaluation using eval/metrics.py functions.
    Computes LPIPS gap, DISTS gap, MPR, Top10.
    """
    
    model.eval()
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {min(len(scenes), max_scenes)} scenes")
    print(f"{'='*60}\n")
    
    for idx, scene_dir in enumerate(tqdm(scenes[:max_scenes], desc=model_name)):
        try:
            sample = load_scene_dir(scene_dir, load_frames=True, load_anime_attrs=True)
            
            if sample.anime_attrs is None or not sample.frames:
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
                sel_probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(sel_probs)[-budget:].tolist())
            
            # Selected frames
            key_frames = [sample.frames[i] for i in sel_idx if i < len(sample.frames)]
            
            # === Compute gaps using eval/metrics.py approach ===
            
            # Sample all frames sparsely (stride=5)
            all_frames_sparse = sample.frames[::5]  # Simple stride
            
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
            
            # Feature distance gap (cosine)
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
    df = pd.DataFrame(results)
    
    summary = {
        "model": model_name,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--v11_checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new/best.pt")
    parser.add_argument("--vsumm_checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_vsumm/sakuga_train/best.pt")
    parser.add_argument("--test_root", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_scenes", type=int, default=474, 
                       help="Max scenes to evaluate (LPIPS is slow)")
    parser.add_argument("--budget_ratio", type=float, default=0.15)
    
    args = parser.parse_args()
    
    # Load scenes
    all_scenes = build_epoch_index(args.test_root)
    print(f"Total scenes: {len(all_scenes)}")
    print(f"Evaluating on: {args.max_scenes} scenes")
    
    all_summaries = {}
    all_details = {}
    
    # === V11 ===
    if Path(args.v11_checkpoint).exists():
        print("\n" + "="*70)
        print("LOADING V11")
        print("="*70)
        
        ckpt = torch.load(args.v11_checkpoint, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        
        model_v11 = create_dsn_v8(
            feat_dim=512 + 6,
            use_pcgrad=False,
            num_attn_layers=config.get("num_attn_layers", 2),
            gating_hidden=config.get("gating_hidden", 64),
            lstm_hidden=config.get("lstm_hidden", 128),
        ).to(args.device)
        
        model_v11.load_state_dict(ckpt["model_state_dict"])
        model_v11.eval()
        
        summary_v11, df_v11 = eval_model_comprehensive(
            model_v11, all_scenes, "V11", args.device,
            budget_ratio=args.budget_ratio, use_anime=True,
            max_scenes=args.max_scenes
        )
        all_summaries["V11"] = summary_v11
        all_details["V11"] = df_v11.to_dict(orient="records")
    
    # === VSUMM ===
    if Path(args.vsumm_checkpoint).exists():
        print("\n" + "="*70)
        print("LOADING VSUMM")
        print("="*70)
        
        ckpt = torch.load(args.vsumm_checkpoint, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        
        model_vsumm = create_dsn_v8(
            feat_dim=512,
            use_pcgrad=False,
            num_attn_layers=config.get("num_attn_layers", 2),
            gating_hidden=config.get("gating_hidden", 64),
            lstm_hidden=config.get("lstm_hidden", 128),
        ).to(args.device)
        
        # Handle different checkpoint formats
        if "model_state_dict" in ckpt:
            model_vsumm.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            model_vsumm.load_state_dict(ckpt["model"])
        elif "state_dict" in ckpt:
            # VSUMM format from ablation
            model_vsumm.load_state_dict(ckpt["state_dict"])
        else:
            # Direct state dict
            model_vsumm.load_state_dict(ckpt)
        
        model_vsumm.eval()
        
        summary_vsumm, df_vsumm = eval_model_comprehensive(
            model_vsumm, all_scenes, "VSUMM", args.device,
            budget_ratio=args.budget_ratio, use_anime=False,
            max_scenes=args.max_scenes
        )
        all_summaries["VSUMM"] = summary_vsumm
        all_details["VSUMM"] = df_vsumm.to_dict(orient="records")
    
    # === SAVE RESULTS ===
    output_dir = Path(args.output_dir)
    
    # Summary JSON
    summary_path = output_dir / "comprehensive_gaps_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    # Details JSON
    details_path = output_dir / "comprehensive_gaps_details.json"
    with open(details_path, "w") as f:
        json.dump(all_details, f, indent=2)
    
    # === PRINT COMPARISON TABLE ===
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON: V11 vs VSUMM")
    print("="*100)
    print(f"{'Model':<10} {'LPIPS Gap':>15} {'DISTS Gap':>15} {'Feat Gap':>12} {'MPR':>10} {'Top10':>10}")
    print("-"*100)
    
    for model_name, summary in all_summaries.items():
        print(f"{model_name:<10} "
              f"{summary['lpips_gap_mean']:>8.4f}±{summary['lpips_gap_std']:<5.4f} "
              f"{summary['dists_gap_mean']:>8.4f}±{summary['dists_gap_std']:<5.4f} "
              f"{summary['feat_gap_mean']:>6.4f}±{summary['feat_gap_std']:<4.4f} "
              f"{summary['mpr_mean']:>6.3f}±{summary['mpr_std']:<3.3f} "
              f"{summary['top10_mean']:>6.3f}±{summary['top10_std']:<3.3f}")
    
    print("\n✅ Results saved:")
    print(f"   Summary: {summary_path}")
    print(f"   Details: {details_path}")


if __name__ == "__main__":
    main()
