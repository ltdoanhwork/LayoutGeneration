#!/usr/bin/env python3
"""
Comprehensive comparison: V11 vs VSUMM vs Baselines
Evaluate all methods and create visualization.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import load_scene_dir, build_epoch_index


def feature_distance_gap(features_all, selected_indices):
    """Compute mean minimum distance from all frames to selected frames."""
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


def compute_quality_metrics(anime_attrs, selected_indices):
    """Compute quality metrics (MPR, Top10)."""
    T = len(anime_attrs)
    quality = anime_attrs.mean(axis=1)
    ranks = np.argsort(np.argsort(quality))
    percentiles = ranks / max(1, T - 1)
    
    mpr = float(np.mean(percentiles[selected_indices]))
    
    k10 = max(1, int(T * 0.1))
    top10_idx = set(np.argsort(quality)[-k10:])
    top10 = len(set(selected_indices) & top10_idx) / k10
    
    return {"mpr": mpr, "top10": top10}


def create_comparison_plot(results, output_path):
    """Create visualization comparing all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    # Plot 1: Feature Gap
    ax = axes[0, 0]
    gaps = [results[m]["feat_gap_mean"] for m in methods]
    stds = [results[m].get("feat_gap_std", 0) for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, gaps, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Feature Gap (lower is better)')
    ax.set_title('Coverage: Feature Distance Gap')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: MPR
    ax = axes[0, 1]
    mprs = [results[m].get("mpr", 0) for m in methods]
    ax.bar(x, mprs, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('MPR (higher is better)')
    ax.set_title('Quality: Mean Percentile Rank')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Top10
    ax = axes[1, 0]
    top10s = [results[m].get("top10", 0) for m in methods]
    ax.bar(x, top10s, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Top-10% Recall')
    ax.set_title('Quality: Top-10% Frame Recall')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Trade-off (MPR vs Gap)
    ax = axes[1, 1]
    for idx, m in enumerate(methods):
        ax.scatter(gaps[idx], mprs[idx], s=200, color=colors[idx], 
                  alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(m, (gaps[idx], mprs[idx]), fontsize=8, 
                   ha='center', va='bottom')
    ax.set_xlabel('Feature Gap (lower is better)')
    ax.set_ylabel('MPR (higher is better)')
    ax.set_title('Trade-off: Coverage vs Quality')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved comparison plot: {output_path}")


def main():
    # Load results from gap comparison
    v11_results_path = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new/gap_comparison_results.json")
    baseline_results_path = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new/baseline_gaps.json")
    
    all_results = {}
    
    # Load V11 results (use best checkpoint)
    if v11_results_path.exists():
        with open(v11_results_path) as f:
            v11_data = json.load(f)
            # Find best checkpoint
            best_ckpt = "best.pt"
            for k in v11_data.keys():
                if "best_epoch" in k and "58" in k:  # Best epoch 58
                    best_ckpt = k
                    break
            
            if best_ckpt in v11_data:
                all_results["V11 (Best)"] = v11_data[best_ckpt]
    
    # Load baseline results
    if baseline_results_path.exists():
        with open(baseline_results_path) as f:
            baseline_data = json.load(f)
            all_results["Random"] = baseline_data["random_baseline"]
            all_results["Uniform"] = baseline_data["uniform_baseline"]
    
    # TODO: Add VSUMM results when available
    
    if len(all_results) < 2:
        print("Not enough results to compare")
        return
    
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON")
    print("="*60)
    
    print(f"\n{'Method':<20} {'FeatGap':>10} {'MPR':>10} {'Top10':>10}")
    print("-"*60)
    
    for method, results in all_results.items():
        feat_gap = results.get("feat_gap_mean", float("nan"))
        mpr = results.get("mpr", float("nan"))
        top10 = results.get("top10", float("nan"))
        print(f"{method:<20} {feat_gap:>10.4f} {mpr:>10.4f} {top10:>10.4f}")
    
    # Create visualization
    output_dir = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new")
    plot_path = output_dir / "comparison_plot.png"
    create_comparison_plot(all_results, plot_path)
    
    # Save combined results
    summary_path = output_dir / "comprehensive_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Comprehensive comparison saved to: {summary_path}")


if __name__ == "__main__":
    main()
