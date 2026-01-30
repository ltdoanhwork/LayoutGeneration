#!/usr/bin/env python3
"""
Single-video comparison script: V11 vs VSUMM vs LLMVS.
Loads all 3 models and runs inference on a specified video scene.
Outputs:
- Stacked grid visualization of keyframes.
- JSON report with metrics (MPR, Top10, Overlap).
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
import pandas as pd

# Add paths for submodules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablation" / "pytorch-vsumm-reinforce"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablation" / "LLMVS"))

from src.models.dsn_v8 import create_dsn_v8
from src.datasets import load_scene_dir
from ablation.LLMVS.llmvs_utils.configs import Config as LLMVSConfig

# Attempt imports for other models inside try-except to handle potential missing paths
try:
    from models import DSN as VSUMM_DSN
except ImportError:
    print("Warning: Could not import VSUMM DSN model.")
    VSUMM_DSN = None

try:
    from networks.model_visual import LLMVSVisual
except ImportError:
    print("Warning: Could not import LLMVSVisual model.")
    LLMVSVisual = None


def create_keyframe_grid(frames, selected_indices, title="", max_frames=8):
    """Create visualization grid of keyframes."""
    if not frames or not selected_indices:
        return None
    
    # Select frames to display (equally spaced if more than max)
    if len(selected_indices) > max_frames:
        # Simple sampling
        display_indices = selected_indices[:max_frames]
    else:
        display_indices = selected_indices
        
    key_frames = [frames[i] for i in display_indices if i < len(frames)]
    
    if not key_frames:
        return None
    
    # Resize frames
    target_h, target_w = 180, 320
    resized = [cv2.resize(f, (target_w, target_h)) for f in key_frames]
    
    # Create grid (single row)
    n = len(resized)
    cols = n
    rows = 1
    
    grid = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
    
    for idx, frame in enumerate(resized):
        c = idx
        grid[0:target_h, c*target_w:(c+1)*target_w] = frame
    
    # Add title bar
    title_bar = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return np.vstack([title_bar, grid])


def main():
    parser = argparse.ArgumentParser(description="Run comparison on a single video")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to scene directory")
    
    # Checkpoints
    parser.add_argument("--v11_ckpt", type=str, 
        default="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_recerr_w0.2/best.pt")
    parser.add_argument("--vsumm_ckpt", type=str,
        default="/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_vsumm/sakuga_train/model_epoch60.pth.tar")
    parser.add_argument("--llmvs_ckpt", type=str,
        default="/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_llmvs/optionB_visual/best_model.pth")
    
    parser.add_argument("--output_dir", type=str, default="demo_outputs/single_video_comparison")
    parser.add_argument("--budget_ratio", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = Path(args.video_dir).name
    
    print(f"Processing Video: {video_name}")
    print(f"Budget Ratio: {args.budget_ratio}")
    
    # 1. Load Data
    print("\nLoading data...")
    try:
        sample = load_scene_dir(Path(args.video_dir), load_frames=True, load_anime_attrs=True)
    except Exception as e:
        print(f"Error loading video: {e}")
        return

    if sample.anime_attrs is None or not sample.frames:
        print("Error: Missing attributes or frames.")
        return

    T = len(sample.feats)
    budget = max(3, min(15, int(T * args.budget_ratio)))
    print(f"Frames: {T}, Budget: {budget}")
    
    # 2. Quality Metrics Setup
    quality = sample.anime_attrs.mean(axis=1)
    ranks = np.argsort(np.argsort(quality))
    percentiles = ranks / max(1, T - 1)
    k10 = max(1, int(T * 0.1))
    top10_idx = set(np.argsort(quality)[-k10:])
    
    results = {
        "video": video_name,
        "frames": T,
        "budget": budget,
        "models": {}
    }
    
    grids = []
    
    # ==========================
    # Run V11
    # ==========================
    print("\n--- Running V11 ---")
    try:
        ckpt = torch.load(args.v11_ckpt, map_location="cpu")
        v11_model = create_dsn_v8(
            feat_dim=512 + 6,
            use_pcgrad=False,
            num_attn_layers=ckpt["config"].get("num_attn_layers", 2),
            gating_hidden=ckpt["config"].get("gating_hidden", 64),
            lstm_hidden=ckpt["config"].get("lstm_hidden", 128),
        ).to(args.device)
        v11_model.load_state_dict(ckpt["model_state_dict"])
        v11_model.eval()
        
        feats_input = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
        feats_t = torch.from_numpy(feats_input).float().unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            probs, _ = v11_model(feats_t)
            probs = probs.squeeze(0).cpu().numpy()
        
        sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
        
        # Metrics
        mpr = float(np.mean(percentiles[sel_idx]))
        top10 = len(set(sel_idx) & top10_idx) / k10
        results["models"]["V11"] = {"mpr": mpr, "top10": top10, "indices": sel_idx}
        print(f"V11: MPR={mpr:.3f}, Top10={top10:.3f}")
        
        grids.append(create_keyframe_grid(sample.frames, sel_idx, title=f"V11 (MPR={mpr:.2f})", max_frames=budget))
        
    except Exception as e:
        print(f"V11 Failed: {e}")

    # ==========================
    # Run VSUMM
    # ==========================
    print("\n--- Running VSUMM ---")
    if VSUMM_DSN:
        try:
            vsumm_model = VSUMM_DSN(in_dim=512, hid_dim=256, num_layers=1, cell='lstm').to(args.device)
            ckpt = torch.load(args.vsumm_ckpt, map_location="cpu")
            str_ckpt = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            clean_ckpt = {k.replace("module.", ""): v for k, v in str_ckpt.items()}
            vsumm_model.load_state_dict(clean_ckpt)
            vsumm_model.eval()
            
            feats_t = torch.from_numpy(sample.feats).float().unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                probs = vsumm_model(feats_t).squeeze().cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
            # Metrics
            mpr = float(np.mean(percentiles[sel_idx]))
            top10 = len(set(sel_idx) & top10_idx) / k10
            results["models"]["VSUMM"] = {"mpr": mpr, "top10": top10, "indices": sel_idx}
            print(f"VSUMM: MPR={mpr:.3f}, Top10={top10:.3f}")
            
            grids.append(create_keyframe_grid(sample.frames, sel_idx, title=f"VSUMM (MPR={mpr:.2f})", max_frames=budget))
            
        except Exception as e:
            print(f"VSUMM Failed: {e}")
    else:
        print("Skipping VSUMM (Model def not found)")

    # ==========================
    # Run LLMVS
    # ==========================
    print("\n--- Running LLMVS ---")
    if LLMVSVisual:
        try:
            config = LLMVSConfig(reduced_dim=2048, input_dim=512, model='LLMVSVisual')
            llmvs_model = LLMVSVisual(config).to(args.device)
            ckpt = torch.load(args.llmvs_ckpt, map_location="cpu")
            str_ckpt = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            clean_ckpt = {k.replace("module.", ""): v for k, v in str_ckpt.items()}
            llmvs_model.load_state_dict(clean_ckpt, strict=False)
            llmvs_model.eval()
            
            feats_t = torch.from_numpy(sample.feats).float().unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                scores = llmvs_model(feats_t).squeeze().cpu().numpy()
            
            sel_idx = sorted(np.argsort(scores)[-budget:].tolist())
            
            # Metrics
            mpr = float(np.mean(percentiles[sel_idx]))
            top10 = len(set(sel_idx) & top10_idx) / k10
            results["models"]["LLMVS"] = {"mpr": mpr, "top10": top10, "indices": sel_idx}
            print(f"LLMVS: MPR={mpr:.3f}, Top10={top10:.3f}")
            
            grids.append(create_keyframe_grid(sample.frames, sel_idx, title=f"LLMVS (MPR={mpr:.2f})", max_frames=budget))
            
        except Exception as e:
            print(f"LLMVS Failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping LLMVS (Model def not found)")

    # ==========================
    # Save Outputs
    # ==========================
    if grids:
        final_img = np.vstack(grids)
        img_Name = f"{video_name}_comparison.jpg"
        cv2.imwrite(str(Path(args.output_dir) / img_Name), final_img)
        print(f"\nSaved visualization to {Path(args.output_dir) / img_Name}")
    
    json_name = f"{video_name}_metrics.json"
    with open(Path(args.output_dir) / json_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {Path(args.output_dir) / json_name}")


if __name__ == "__main__":
    main()
