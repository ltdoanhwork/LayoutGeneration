#!/usr/bin/env python3
"""
Demo comparison: V11 vs VSUMM on 10 videos
Budget ratio = 0.1, save results in comparison table with visualizations
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dsn_v8 import create_dsn_v8
from src.datasets import load_scene_dir, build_epoch_index


def create_keyframe_grid(frames, selected_indices, title="", max_frames=10):
    """Create visualization grid of keyframes."""
    if not frames or not selected_indices:
        return None
    
    # Select frames to display
    display_indices = selected_indices[:max_frames]
    key_frames = [frames[i] for i in display_indices if i < len(frames)]
    
    if not key_frames:
        return None
    
    # Resize frames
    target_h, target_w = 180, 320
    resized = [cv2.resize(f, (target_w, target_h)) for f in key_frames]
    
    # Create grid
    n = len(resized)
    cols = 5
    rows = (n + cols - 1) // cols
    
    grid = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
    
    for idx, frame in enumerate(resized):
        r = idx // cols
        c = idx % cols
        grid[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = frame
    
    # Add title
    if title:
        cv2.putText(grid, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return grid


def run_demo_comparison(v11_model, vsumm_model, scenes, output_dir, 
                       budget_ratio=0.1, device="cuda", num_videos=10):
    """Run demo on N videos, comparing V11 and VSUMM."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for idx, scene_dir in enumerate(scenes[:num_videos]):
        print(f"\n{'='*60}")
        print(f"Video {idx+1}/{num_videos}: {scene_dir.name}")
        print(f"{'='*60}")
        
        try:
            sample = load_scene_dir(scene_dir, load_frames=True, load_anime_attrs=True)
            
            if sample.anime_attrs is None or not sample.frames:
                continue
            
            T = len(sample.feats)
            budget = max(1, int(T * budget_ratio))
            
            print(f"Total frames: {T}, Budget: {budget}")
            
            # V11 prediction
            feats_v11 = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
            feats_t_v11 = torch.from_numpy(feats_v11).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs_v11, _ = v11_model(feats_t_v11)
                probs_v11 = probs_v11.squeeze(0).cpu().numpy()
            
            sel_idx_v11 = sorted(np.argsort(probs_v11)[-budget:].tolist())
            
            # VSUMM prediction (returns only probs, not tuple)
            feats_vsumm = sample.feats
            feats_t_vsumm = torch.from_numpy(feats_vsumm).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs_vsumm = vsumm_model(feats_t_vsumm)  # DSN returns only probs (T, 1)
                probs_vsumm = probs_vsumm.squeeze().cpu().numpy()  # (T,)
            
            sel_idx_vsumm = sorted(np.argsort(probs_vsumm)[-budget:].tolist())
            
            # Compute metrics
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            
            mpr_v11 = float(np.mean(percentiles[sel_idx_v11]))
            mpr_vsumm = float(np.mean(percentiles[sel_idx_vsumm]))
            
            k10 = max(1, int(T * 0.1))
            top10_idx = set(np.argsort(quality)[-k10:])
            top10_v11 = len(set(sel_idx_v11) & top10_idx) / k10
            top10_vsumm = len(set(sel_idx_vsumm) & top10_idx) / k10
            
            # Overlap
            overlap = len(set(sel_idx_v11) & set(sel_idx_vsumm))
            overlap_pct = overlap / budget * 100
            
            result = {
                "video_id": idx + 1,
                "scene_name": scene_dir.name,
                "total_frames": T,
                "budget": budget,
                "v11_mpr": mpr_v11,
                "vsumm_mpr": mpr_vsumm,
                "v11_top10": top10_v11,
                "vsumm_top10": top10_vsumm,
                "overlap": overlap,
                "overlap_pct": overlap_pct,
                "v11_indices": sel_idx_v11[:10],  # First 10
                "vsumm_indices": sel_idx_vsumm[:10],
            }
            results.append(result)
            
            print(f"V11:   MPR={mpr_v11:.3f}, Top10={top10_v11:.3f}")
            print(f"VSUMM: MPR={mpr_vsumm:.3f}, Top10={top10_vsumm:.3f}")
            print(f"Selection overlap: {overlap}/{budget} ({overlap_pct:.1f}%)")
            
            # Create visualizations
            grid_v11 = create_keyframe_grid(sample.frames, sel_idx_v11, 
                                           title=f"V11 (MPR={mpr_v11:.2f})")
            grid_vsumm = create_keyframe_grid(sample.frames, sel_idx_vsumm,
                                             title=f"VSUMM (MPR={mpr_vsumm:.2f})")
            
            if grid_v11 is not None and grid_vsumm is not None:
                # Stack vertically
                combined = np.vstack([grid_v11, grid_vsumm])
                viz_path = output_dir / f"comparison_video{idx+1:02d}_{scene_dir.name}.jpg"
                cv2.imwrite(str(viz_path), combined)
                print(f"Saved: {viz_path}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Save results as JSON
    json_path = output_dir / "demo_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Create summary table
    df = pd.DataFrame(results)
    
    # Summary table
    print("\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    print(df[[  'video_id', 'scene_name', 'total_frames', 'budget', 
                'v11_mpr', 'vsumm_mpr', 'v11_top10', 'vsumm_top10', 
                'overlap_pct']].to_string(index=False))
    
    # Save as CSV
    csv_path = output_dir / "demo_comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    
    # Mean statistics
    print("\n" + "="*60)
    print("AVERAGE ACROSS ALL VIDEOS")
    print("="*60)
    print(f"{'Metric':<20} {'V11':>10} {'VSUMM':>10} {'Difference':>12}")
    print("-"*60)
    print(f"{'MPR':<20} {df['v11_mpr'].mean():>10.4f} {df['vsumm_mpr'].mean():>10.4f} "
          f"{df['v11_mpr'].mean() - df['vsumm_mpr'].mean():>12.4f}")
    print(f"{'Top10 Recall':<20} {df['v11_top10'].mean():>10.4f} {df['vsumm_top10'].mean():>10.4f} "
          f"{df['v11_top10'].mean() - df['vsumm_top10'].mean():>12.4f}")
    print(f"{'Avg Overlap %':<20} {df['overlap_pct'].mean():>10.1f}%")
    
    print(f"\n✅ Results saved:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    print(f"   Visualizations: {output_dir}/*.jpg")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v11_checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new/best.pt")
    parser.add_argument("--vsumm_checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_vsumm/sakuga_train/model_epoch60.pth.tar")
    parser.add_argument("--test_root", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test")
    parser.add_argument("--output_dir", type=str, default="demo_outputs/comparison")
    parser.add_argument("--budget_ratio", type=float, default=0.1)
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load test scenes
    all_scenes = build_epoch_index(args.test_root)
    print(f"Loaded {len(all_scenes)} test scenes")
    
    # Load V11
    print("\nLoading V11...")
    v11_ckpt = torch.load(args.v11_checkpoint, map_location="cpu")
    v11_config = v11_ckpt.get("config", {})
    
    v11_model = create_dsn_v8(
        feat_dim=512 + 6,  # CLIP + anime attrs
        use_pcgrad=False,
        num_attn_layers=v11_config.get("num_attn_layers", 2),
        gating_hidden=v11_config.get("gating_hidden", 64),
        lstm_hidden=v11_config.get("lstm_hidden", 128),
    ).to(args.device)
    v11_model.load_state_dict(v11_ckpt["model_state_dict"])
    v11_model.eval()
    
    
    # Load VSUMM - use correct DSN architecture
    print("Loading VSUMM...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "ablation" / "pytorch-vsumm-reinforce"))
    from models import DSN
    
    vsumm_ckpt = torch.load(args.vsumm_checkpoint, map_location="cpu", weights_only=False)
    vsumm_model = DSN(in_dim=512, hid_dim=256, num_layers=1, cell='lstm').to(args.device)
    
    # Handle checkpoint format
    if "state_dict" in vsumm_ckpt:
        state_dict = vsumm_ckpt["state_dict"]
    elif "model_state_dict" in vsumm_ckpt:
        state_dict = vsumm_ckpt["model_state_dict"]
    else:
        state_dict = vsumm_ckpt
    
    # Strip module prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v
    
    vsumm_model.load_state_dict(new_state_dict)
    vsumm_model.eval()
    
    # Run demo
    run_demo_comparison(
        v11_model, vsumm_model, all_scenes, args.output_dir,
        budget_ratio=args.budget_ratio, device=args.device,
        num_videos=args.num_videos
    )


if __name__ == "__main__":
    main()
