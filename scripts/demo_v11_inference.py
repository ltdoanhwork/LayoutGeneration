#!/usr/bin/env python3
"""
Demo script for V11 Final - Generate keyframe selections with budget_ratio=0.1
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dsn_v8 import create_dsn_v8
from src.datasets import load_scene_dir, build_epoch_index


def visualize_selection(frames, selected_indices, output_path, max_display=10):
    """Create a grid visualization of selected keyframes."""
    if not frames or not selected_indices:
        return
    
    # Limit display
    display_indices = selected_indices[:max_display]
    key_frames = [frames[i] for i in display_indices if i < len(frames)]
    
    if not key_frames:
        return
    
    # Create grid
    n = len(key_frames)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    
    h, w = key_frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for idx, frame in enumerate(key_frames):
        r = idx // cols
        c = idx % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = cv2.resize(frame, (w, h))
    
    cv2.imwrite(output_path, grid)
    print(f"  Saved visualization: {output_path}")


def run_v11_demo(checkpoint_path, test_scenes, output_dir, budget_ratio=0.1, device="cuda"):
    """Run V11 inference demo."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading V11 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})
    
    # Setup model
    feat_dim = config.get("feat_dim", 512)
    use_anime = not config.get("no_anime_attrs", False)
    full_feat_dim = feat_dim + (6 if use_anime else 0)
    
    model = create_dsn_v8(
        feat_dim=full_feat_dim,
        use_pcgrad=False,
        num_attn_layers=config.get("num_attn_layers", 2),
        gating_hidden=config.get("gating_hidden", 64),
        lstm_hidden=config.get("lstm_hidden", 128),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print(f"\nRunning V11 Demo (budget_ratio={budget_ratio})...")
    print(f"Testing on {len(test_scenes)} scenes\n")
    
    results = []
    
    for idx, scene_dir in enumerate(test_scenes[:5]):  # Demo on first 5 scenes
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
            budget = max(1, int(T * budget_ratio))
            
            # Predict
            with torch.no_grad():
                probs, _ = model(feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
            # Compute quality scores
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            mpr = float(np.mean(percentiles[sel_idx]))
            
            result = {
                "scene": scene_dir.name,
                "total_frames": T,
                "selected": len(sel_idx),
                "budget_ratio": budget_ratio,
                "mpr": mpr,
                "indices": sel_idx
            }
            results.append(result)
            
            # Visualize
            viz_path = output_dir / f"v11_scene{idx:02d}_{scene_dir.name}.jpg"
            visualize_selection(sample.frames, sel_idx, str(viz_path))
            
            print(f"Scene {idx+1}: {scene_dir.name}")
            print(f"  Total frames: {T}, Selected: {len(sel_idx)}")
            print(f"  MPR: {mpr:.3f}")
            print(f"  Selected indices: {sel_idx[:10]}{'...' if len(sel_idx) > 10 else ''}")
            print()
            
        except Exception as e:
            print(f"Error processing {scene_dir}: {e}")
            continue
    
    # Save results
    results_path = output_dir / "v11_demo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ V11 demo complete! Results saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="V11 Demo with budget_ratio=0.1")
    parser.add_argument("--checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new/best.pt")
    parser.add_argument("--test_root", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test")
    parser.add_argument("--output_dir", type=str, default="demo_outputs/v11")
    parser.add_argument("--budget_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load test scenes
    test_scenes = build_epoch_index(args.test_root)
    print(f"Found {len(test_scenes)} test scenes")
    
    # Run demo
    run_v11_demo(
        args.checkpoint,
        test_scenes,
        args.output_dir,
        budget_ratio=args.budget_ratio,
        device=args.device
    )


if __name__ == "__main__":
    main()
