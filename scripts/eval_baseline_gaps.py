#!/usr/bin/env python3
"""
Quick Feature Gap evaluation for VSUMM to compare with V11 Final.
"""

import sys
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import load_scene_dir, build_epoch_index


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


def random_baseline(n_frames: int, budget: int, seed: int = 42) -> list:
    """Random selection baseline."""
    np.random.seed(seed)
    return sorted(np.random.choice(n_frames, min(budget, n_frames), replace=False).tolist())


def uniform_baseline(n_frames: int, budget: int) -> list:
    """Uniform spacing baseline."""
    if budget >= n_frames:
        return list(range(n_frames))
    indices = np.linspace(0, n_frames - 1, budget, dtype=int)
    return sorted(set(indices.tolist()))


def main():
    val_root = "/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test"
    val_scenes = build_epoch_index(val_root)
    
    print(f"Evaluating baselines on {len(val_scenes)} scenes...")
    
    random_gaps = []
    uniform_gaps = []
    
    for scene_dir in tqdm(val_scenes, desc="Baseline Eval"):
        try:
            sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=False)
            T = len(sample.feats)
            budget = max(3, min(15, int(T * 0.15)))
            
            # Random baseline
            rand_idx = random_baseline(T, budget)
            rand_gap = feature_distance_gap(sample.feats, rand_idx)
            if not np.isnan(rand_gap):
                random_gaps.append(rand_gap)
            
            # Uniform baseline
            unif_idx = uniform_baseline(T, budget)
            unif_gap = feature_distance_gap(sample.feats, unif_idx)
            if not np.isnan(unif_gap):
                uniform_gaps.append(unif_gap)
                
        except Exception as e:
            continue
    
    results = {
        "random_baseline": {
            "feat_gap_mean": float(np.mean(random_gaps)),
            "feat_gap_std": float(np.std(random_gaps)),
            "n_scenes": len(random_gaps)
        },
        "uniform_baseline": {
            "feat_gap_mean": float(np.mean(uniform_gaps)),
            "feat_gap_std": float(np.std(uniform_gaps)),
            "n_scenes": len(uniform_gaps)
        }
    }
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Random:  {results['random_baseline']['feat_gap_mean']:.4f} ± {results['random_baseline']['feat_gap_std']:.4f}")
    print(f"Uniform: {results['uniform_baseline']['feat_gap_mean']:.4f} ± {results['uniform_baseline']['feat_gap_std']:.4f}")
    
    # Save
    output_path = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new/baseline_gaps.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved to: {output_path}")


if __name__ == "__main__":
    main()
