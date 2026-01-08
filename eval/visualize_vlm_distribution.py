#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Distribution Visualizer

Visualize the distribution of VLM quality scores for selected keyframes.
Shows how selected frames compare to the overall quality distribution.
"""

import os
import argparse
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# VLM Quality dimensions
VLM_DIMS = [
    "Line Art", "Sakuga", "Composition", "Color Harmony",
    "Expression", "Motion Blur", "Background", "Visual Impact"
]

def load_vlm_scores(scene_dir: Path) -> Optional[np.ndarray]:
    """Load VLM quality scores from scene directory."""
    vlm_path = scene_dir / "vlm_quality.npy"
    if vlm_path.exists():
        return np.load(vlm_path)
    return None

def create_vlm_dashboard(
    vlm_scores: np.ndarray,       # (T, 8)
    sel_idx: List[int],
    save_path: str,
    title: str = "VLM Quality Distribution"
):
    """Create comprehensive VLM quality dashboard."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    T = len(vlm_scores)
    K = len(sel_idx)
    
    # Aggregate score (mean across 8 dims)
    agg_all = vlm_scores.mean(axis=1)
    agg_sel = vlm_scores[sel_idx].mean(axis=1)
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Histogram of aggregate VLM score
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(agg_all, bins=30, alpha=0.6, label=f'All ({T})', color='steelblue', density=True)
    ax1.hist(agg_sel, bins=15, alpha=0.8, label=f'Selected ({K})', color='coral', density=True)
    ax1.axvline(np.mean(agg_all), color='steelblue', linestyle='--', label=f'All μ={np.mean(agg_all):.3f}')
    ax1.axvline(np.mean(agg_sel), color='coral', linestyle='--', label=f'Sel μ={np.mean(agg_sel):.3f}')
    ax1.set_xlabel('Aggregate VLM Score')
    ax1.set_ylabel('Density')
    ax1.set_title('VLM Score Distribution')
    ax1.legend(fontsize=8)
    
    # 2. CDF comparison
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_all = np.sort(agg_all)
    cdf_all = np.arange(1, T+1) / T
    ax2.plot(sorted_all, cdf_all, label='All Frames', color='steelblue', linewidth=2)
    for i, idx in enumerate(sel_idx):
        y = np.searchsorted(sorted_all, agg_all[idx]) / T
        ax2.scatter([agg_all[idx]], [y], color='coral', s=50, zorder=5)
    ax2.set_xlabel('VLM Score')
    ax2.set_ylabel('CDF')
    ax2.set_title('Cumulative Distribution')
    ax2.legend()
    
    # 3. Per-dimension comparison (bar chart)
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(8)
    width = 0.35
    all_means = vlm_scores.mean(axis=0)
    sel_means = vlm_scores[sel_idx].mean(axis=0)
    ax3.bar(x - width/2, all_means, width, label='All', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, sel_means, width, label='Selected', color='coral', alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.split()[0] for d in VLM_DIMS], rotation=45, ha='right')
    ax3.set_ylabel('Mean Score')
    ax3.set_title('Per-Dimension Comparison')
    ax3.legend()
    
    # 4. Radar chart
    ax4 = fig.add_subplot(gs[1, 0], polar=True)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    all_vals = all_means.tolist() + [all_means[0]]
    sel_vals = sel_means.tolist() + [sel_means[0]]
    ax4.plot(angles, all_vals, 'o-', linewidth=2, label='All', color='steelblue')
    ax4.fill(angles, all_vals, alpha=0.25, color='steelblue')
    ax4.plot(angles, sel_vals, 'o-', linewidth=2, label='Selected', color='coral')
    ax4.fill(angles, sel_vals, alpha=0.25, color='coral')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels([d.split()[0] for d in VLM_DIMS], fontsize=8)
    ax4.set_title('Quality Radar')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # 5. Timeline with VLM scores
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.plot(range(T), agg_all, color='steelblue', alpha=0.6, linewidth=1, label='All')
    ax5.scatter(sel_idx, agg_all[sel_idx], color='coral', s=80, zorder=5, edgecolors='black', label='Selected')
    ax5.axhline(np.percentile(agg_all, 90), color='green', linestyle='--', alpha=0.7, label='P90')
    ax5.axhline(np.percentile(agg_all, 50), color='orange', linestyle='--', alpha=0.7, label='P50')
    ax5.set_xlabel('Frame Index')
    ax5.set_ylabel('VLM Score')
    ax5.set_title('VLM Quality Timeline')
    ax5.legend(loc='upper right')
    
    # 6. Percentile rank of selected frames
    ax6 = fig.add_subplot(gs[2, 0])
    ranks = np.argsort(np.argsort(agg_all)) / (T - 1)
    sel_ranks = ranks[sel_idx]
    ax6.hist(sel_ranks, bins=20, color='coral', alpha=0.8, edgecolor='black')
    ax6.axvline(np.mean(sel_ranks), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(sel_ranks):.3f}')
    ax6.set_xlabel('Percentile Rank')
    ax6.set_ylabel('Count')
    ax6.set_title('Selected Frame Percentile Distribution')
    ax6.legend()
    
    # 7. Box plot per dimension
    ax7 = fig.add_subplot(gs[2, 1:])
    data_all = [vlm_scores[:, i] for i in range(8)]
    data_sel = [vlm_scores[sel_idx, i] for i in range(8)]
    positions_all = np.arange(8) * 2
    positions_sel = np.arange(8) * 2 + 0.6
    bp1 = ax7.boxplot(data_all, positions=positions_all, widths=0.5, patch_artist=True)
    bp2 = ax7.boxplot(data_sel, positions=positions_sel, widths=0.4, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    for patch in bp2['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.8)
    ax7.set_xticks(np.arange(8) * 2 + 0.3)
    ax7.set_xticklabels([d.split()[0] for d in VLM_DIMS], rotation=45, ha='right')
    ax7.set_ylabel('Score')
    ax7.set_title('Per-Dimension Box Plot (Blue=All, Orange=Selected)')
    
    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Dashboard saved: {save_path}")

def visualize_dataset_vlm(
    dataset_dir: str,
    output_dir: str,
    max_scenes: int = None
):
    """Visualize VLM distribution across multiple scenes."""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect VLM scores from all scenes
    all_vlm = []
    scene_count = 0
    
    for video_dir in dataset_path.iterdir():
        if not video_dir.is_dir():
            continue
        for scene_dir in video_dir.iterdir():
            if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
                continue
            
            vlm = load_vlm_scores(scene_dir)
            if vlm is not None:
                all_vlm.append(vlm)
                scene_count += 1
                
            if max_scenes and scene_count >= max_scenes:
                break
        if max_scenes and scene_count >= max_scenes:
            break
    
    if not all_vlm:
        print("No VLM scores found in dataset!")
        return
    
    # Concatenate all scores
    combined = np.concatenate(all_vlm, axis=0)
    print(f"Loaded VLM scores from {scene_count} scenes ({len(combined)} frames)")
    
    # Create aggregate visualization
    # Simulate selection (top 10% per-dimension average)
    agg = combined.mean(axis=1)
    n_select = max(1, int(len(agg) * 0.10))
    sel_idx = np.argsort(agg)[-n_select:].tolist()
    
    create_vlm_dashboard(
        combined, sel_idx,
        save_path=str(output_path / "vlm_dataset_distribution.png"),
        title=f"VLM Quality Distribution ({scene_count} scenes, {len(combined)} frames)"
    )
    
    # Save statistics
    stats = {
        "n_scenes": scene_count,
        "n_frames": len(combined),
        "mean_per_dim": combined.mean(axis=0).tolist(),
        "std_per_dim": combined.std(axis=0).tolist(),
        "overall_mean": float(agg.mean()),
        "overall_std": float(agg.std()),
        "percentiles": {
            "p10": float(np.percentile(agg, 10)),
            "p25": float(np.percentile(agg, 25)),
            "p50": float(np.percentile(agg, 50)),
            "p75": float(np.percentile(agg, 75)),
            "p90": float(np.percentile(agg, 90)),
        }
    }
    
    with open(output_path / "vlm_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📊 VLM Dataset Statistics:")
    print(f"   Overall Mean: {stats['overall_mean']:.3f} ± {stats['overall_std']:.3f}")
    print(f"   P50 (Median): {stats['percentiles']['p50']:.3f}")
    print(f"   P90: {stats['percentiles']['p90']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Visualize VLM Quality Distribution")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset with VLM scores")
    parser.add_argument("--output_dir", type=str, default="runs/vlm_visualization", help="Output directory")
    parser.add_argument("--max_scenes", type=int, default=None, help="Max scenes to process")
    
    args = parser.parse_args()
    
    visualize_dataset_vlm(args.dataset_dir, args.output_dir, args.max_scenes)

if __name__ == "__main__":
    main()
