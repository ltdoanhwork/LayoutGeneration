#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution Visualization for Anime Keyframe Selection

Visualizes the quality distribution of videos and shows how well
the model selects high-quality frames relative to the distribution.

Creates:
1. Quality histogram with selected frames highlighted
2. Cumulative distribution function (CDF) with selection overlay
3. Per-attribute radar chart
4. Box plot comparing selected vs all frames
5. Selection percentile timeline

Usage:
    python -m eval.visualize_distribution \
        --data_dir outputs/distribution_data \
        --output_dir outputs/distribution_viz

"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Plotting (with fallback)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not found. Visualization disabled.")

# Import distribution metrics
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rl.distribution_metrics import (
    DistributionAwareMetrics,
    ATTR_NAMES,
    ATTR_INDEX,
)


def create_quality_histogram(
    quality_all: np.ndarray,
    quality_selected: np.ndarray,
    sel_idx: List[int],
    save_path: Optional[str] = None,
    title: str = "Quality Distribution",
) -> Optional[plt.Figure]:
    """
    Create histogram showing distribution with selected frames highlighted.
    
    Args:
        quality_all: Quality scores for all frames
        quality_selected: Quality scores for selected frames
        sel_idx: Indices of selected frames
        save_path: Path to save figure (optional)
        title: Plot title
        
    Returns:
        matplotlib Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # All frames histogram
    bins = np.linspace(quality_all.min(), quality_all.max(), 30)
    ax.hist(quality_all, bins=bins, alpha=0.5, label='All Frames', 
            color='steelblue', edgecolor='black')
    
    # Selected frames histogram
    if len(quality_selected) > 0:
        ax.hist(quality_selected, bins=bins, alpha=0.8, label='Selected Frames',
                color='coral', edgecolor='black')
    
    # Add percentile lines
    p50 = np.percentile(quality_all, 50)
    p75 = np.percentile(quality_all, 75)
    p90 = np.percentile(quality_all, 90)
    
    ax.axvline(p50, color='green', linestyle='--', linewidth=2, label='Median (P50)')
    ax.axvline(p75, color='orange', linestyle='--', linewidth=2, label='P75')
    ax.axvline(p90, color='red', linestyle='--', linewidth=2, label='P90')
    
    ax.set_xlabel('Quality Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def create_cdf_plot(
    quality_all: np.ndarray,
    quality_selected: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Cumulative Distribution",
) -> Optional[plt.Figure]:
    """
    Create CDF plot showing where selected frames fall in distribution.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort for CDF
    sorted_all = np.sort(quality_all)
    cdf_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
    
    ax.plot(sorted_all, cdf_all, 'b-', linewidth=2, label='All Frames CDF')
    
    # Mark selected frames on CDF
    if len(quality_selected) > 0:
        for q in quality_selected:
            # Find CDF value
            cdf_val = np.sum(quality_all <= q) / len(quality_all)
            ax.scatter([q], [cdf_val], color='coral', s=100, zorder=5, edgecolor='black')
        
        # Add mean line for selected
        mean_sel = np.mean(quality_selected)
        ax.axvline(mean_sel, color='coral', linestyle='--', linewidth=2, 
                   label=f'Selected Mean (P{100*np.mean(quality_all <= mean_sel):.0f})')
    
    # Reference lines
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.75, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.90, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Quality Score', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def create_per_attribute_radar(
    attrs_all: np.ndarray,
    sel_idx: List[int],
    save_path: Optional[str] = None,
    title: str = "Per-Attribute Percentiles",
) -> Optional[plt.Figure]:
    """
    Create radar chart showing percentile rank for each attribute.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    metrics = DistributionAwareMetrics()
    per_attr = metrics.compute_per_attribute_percentile(attrs_all, sel_idx)
    
    # Prepare data
    attr_names = [name.capitalize() for name in ATTR_NAMES]
    values = [per_attr.get(f"percentile_{name}", 0.5) for name in ATTR_NAMES]
    
    # Close the radar chart
    values = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(attr_names), endpoint=False).tolist()
    angles = angles + [angles[0]]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot
    ax.fill(angles, values, color='coral', alpha=0.25)
    ax.plot(angles, values, color='coral', linewidth=2, label='Selected Frames')
    
    # Reference line (random = 0.5)
    reference = [0.5] * (len(attr_names) + 1)
    ax.plot(angles, reference, color='gray', linestyle='--', linewidth=1, label='Random (P50)')
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attr_names, fontsize=11)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    
    # Add value annotations
    for angle, val, name in zip(angles[:-1], values[:-1], attr_names):
        ax.annotate(f'{val:.2f}', xy=(angle, val), fontsize=9, ha='center')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def create_selection_timeline(
    quality_all: np.ndarray,
    sel_idx: List[int],
    save_path: Optional[str] = None,
    title: str = "Selection Timeline",
) -> Optional[plt.Figure]:
    """
    Create timeline showing frame quality with selected frames marked.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    T = len(quality_all)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    
    # Top: Quality over time
    ax1.plot(range(T), quality_all, 'b-', alpha=0.7, linewidth=1, label='Frame Quality')
    
    # Highlight selected frames
    sel_mask = np.zeros(T, dtype=bool)
    sel_mask[sel_idx] = True
    ax1.scatter(sel_idx, quality_all[sel_idx], color='coral', s=100, 
                zorder=5, label='Selected', edgecolor='black')
    
    # Percentile lines
    p50 = np.percentile(quality_all, 50)
    p75 = np.percentile(quality_all, 75)
    p90 = np.percentile(quality_all, 90)
    
    ax1.axhline(p50, color='green', linestyle='--', alpha=0.5, label='P50')
    ax1.axhline(p75, color='orange', linestyle='--', alpha=0.5, label='P75')
    ax1.axhline(p90, color='red', linestyle='--', alpha=0.5, label='P90')
    
    ax1.set_ylabel('Quality Score', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T)
    
    # Bottom: Selection bar
    colors = np.where(sel_mask, 'coral', 'steelblue')
    alphas = np.where(sel_mask, 1.0, 0.3)
    
    for i in range(T):
        ax2.axvline(i, color=colors[i], alpha=alphas[i], linewidth=1)
    
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Selection', fontsize=10)
    ax2.set_yticks([])
    ax2.set_xlim(0, T)
    
    # Add selection markers
    ax2.scatter(sel_idx, [0.5] * len(sel_idx), color='coral', s=50, marker='|', linewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def create_comparison_boxplot(
    quality_all: np.ndarray,
    quality_selected: np.ndarray,
    per_attr_all: Optional[np.ndarray] = None,
    per_attr_selected: Optional[np.ndarray] = None,
    sel_idx: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    title: str = "Selection Quality Comparison",
) -> Optional[plt.Figure]:
    """
    Create box plot comparing all vs selected frame quality.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Aggregate quality
    ax1 = axes[0]
    data = [quality_all, quality_selected]
    bp = ax1.boxplot(data, labels=['All Frames', 'Selected'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.8)
    
    ax1.set_ylabel('Quality Score', fontsize=12)
    ax1.set_title('Aggregate Quality', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    ax1.scatter([1], [np.mean(quality_all)], color='black', s=100, zorder=5, marker='D', label='Mean')
    ax1.scatter([2], [np.mean(quality_selected)], color='black', s=100, zorder=5, marker='D')
    ax1.legend()
    
    # Right: Per-attribute comparison
    ax2 = axes[1]
    if per_attr_all is not None and per_attr_selected is not None and sel_idx is not None:
        x = np.arange(len(ATTR_NAMES))
        width = 0.35
        
        mean_all = per_attr_all.mean(axis=0)
        mean_sel = per_attr_selected.mean(axis=0)
        
        bars1 = ax2.bar(x - width/2, mean_all, width, label='All Frames', 
                       color='steelblue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, mean_sel, width, label='Selected',
                       color='coral', alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([n.capitalize() for n in ATTR_NAMES], rotation=45, ha='right')
        ax2.set_ylabel('Mean Score', fontsize=12)
        ax2.set_title('Per-Attribute Comparison', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'Per-attribute data not available', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Per-Attribute Comparison', fontsize=12)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def create_comprehensive_dashboard(
    attrs_all: np.ndarray,
    sel_idx: List[int],
    metrics_result: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Distribution Analysis Dashboard",
) -> Optional[plt.Figure]:
    """
    Create comprehensive dashboard with all visualizations.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    metrics_computer = DistributionAwareMetrics()
    quality = metrics_computer.compute_aggregate_quality(attrs_all)
    
    T = len(attrs_all)
    sel_idx_valid = [i for i in sel_idx if 0 <= i < T]
    quality_selected = quality[sel_idx_valid] if sel_idx_valid else np.array([])
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Histogram (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(quality.min(), quality.max(), 25)
    ax1.hist(quality, bins=bins, alpha=0.5, label='All', color='steelblue', edgecolor='black')
    if len(quality_selected) > 0:
        ax1.hist(quality_selected, bins=bins, alpha=0.8, label='Selected', color='coral', edgecolor='black')
    ax1.axvline(np.percentile(quality, 90), color='red', linestyle='--', label='P90')
    ax1.set_xlabel('Quality')
    ax1.set_ylabel('Count')
    ax1.set_title('Quality Distribution', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. CDF (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_q = np.sort(quality)
    cdf = np.arange(1, len(sorted_q) + 1) / len(sorted_q)
    ax2.plot(sorted_q, cdf, 'b-', linewidth=2)
    if len(quality_selected) > 0:
        for q in quality_selected:
            cdf_val = np.sum(quality <= q) / len(quality)
            ax2.scatter([q], [cdf_val], color='coral', s=80, zorder=5, edgecolor='black')
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Quality')
    ax2.set_ylabel('CDF')
    ax2.set_title('Cumulative Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics summary (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # Create metrics text
    metrics_text = [
        f"Mean Percentile Rank: {metrics_result.get('mean_percentile_rank', 0):.3f}",
        f"Z-Score Improvement: {metrics_result.get('zscore_improvement', 0):.3f}",
        f"Top-10% Coverage: {metrics_result.get('top_10_coverage', 0):.1%}",
        f"Top-10% Precision: {metrics_result.get('top_10_precision', 0):.1%}",
        f"Above P90 Ratio: {metrics_result.get('above_p90_ratio', 0):.1%}",
        f"Above Median Ratio: {metrics_result.get('above_median_ratio', 0):.1%}",
        "",
        f"Total Frames: {T}",
        f"Selected Frames: {len(sel_idx_valid)}",
    ]
    
    text = '\n'.join(metrics_text)
    ax3.text(0.1, 0.9, text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.set_title('Metrics Summary', fontweight='bold')
    
    # 4. Radar chart (middle-left)
    ax4 = fig.add_subplot(gs[1, 0], polar=True)
    per_attr = metrics_computer.compute_per_attribute_percentile(attrs_all, sel_idx_valid)
    attr_names = [n.capitalize() for n in ATTR_NAMES]
    values = [per_attr.get(f"percentile_{n}", 0.5) for n in ATTR_NAMES]
    values = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(ATTR_NAMES), endpoint=False).tolist()
    angles = angles + [angles[0]]
    
    ax4.fill(angles, values, color='coral', alpha=0.25)
    ax4.plot(angles, values, color='coral', linewidth=2)
    ax4.plot(angles, [0.5] * len(angles), color='gray', linestyle='--', alpha=0.5)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(attr_names, fontsize=9)
    ax4.set_ylim(0, 1)
    ax4.set_title('Per-Attribute Percentiles', fontweight='bold', pad=20)
    
    # 5. Timeline (middle, spanning 2 columns)
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.plot(range(T), quality, 'b-', alpha=0.6, linewidth=0.5)
    if sel_idx_valid:
        ax5.scatter(sel_idx_valid, quality[sel_idx_valid], color='coral', 
                   s=60, zorder=5, edgecolor='black', label='Selected')
    ax5.axhline(np.percentile(quality, 50), color='green', linestyle='--', alpha=0.5, label='P50')
    ax5.axhline(np.percentile(quality, 90), color='red', linestyle='--', alpha=0.5, label='P90')
    ax5.set_xlabel('Frame Index')
    ax5.set_ylabel('Quality')
    ax5.set_title('Quality Timeline', fontweight='bold')
    ax5.legend(fontsize=8, loc='upper right')
    ax5.set_xlim(0, T)
    ax5.grid(True, alpha=0.3)
    
    # 6. Box plots (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    data = [quality, quality_selected] if len(quality_selected) > 0 else [quality]
    labels = ['All', 'Selected'] if len(quality_selected) > 0 else ['All']
    bp = ax6.boxplot(data, labels=labels, patch_artist=True)
    if len(data) > 0:
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.5)
    if len(data) > 1:
        bp['boxes'][1].set_facecolor('coral')
        bp['boxes'][1].set_alpha(0.8)
    ax6.set_ylabel('Quality')
    ax6.set_title('Quality Comparison', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Per-attribute bars (bottom spanning 2 columns)
    ax7 = fig.add_subplot(gs[2, 1:])
    x = np.arange(len(ATTR_NAMES))
    width = 0.35
    
    mean_all = attrs_all.mean(axis=0)
    mean_sel = attrs_all[sel_idx_valid].mean(axis=0) if sel_idx_valid else mean_all
    
    ax7.bar(x - width/2, mean_all, width, label='All Frames', color='steelblue', alpha=0.7)
    ax7.bar(x + width/2, mean_sel, width, label='Selected', color='coral', alpha=0.8)
    ax7.set_xticks(x)
    ax7.set_xticklabels([n.capitalize() for n in ATTR_NAMES])
    ax7.set_ylabel('Mean Score')
    ax7.set_title('Per-Attribute Mean Comparison', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def create_aggregate_radar_chart(
    agg_per_attr: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Aggregate Per-Attribute Percentiles",
) -> Optional[plt.Figure]:
    """
    Create radar chart for aggregate attribute percentiles.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    attr_names = [name.capitalize() for name in ATTR_NAMES]
    values = [agg_per_attr.get(f"percentile_{name}", 0.5) for name in ATTR_NAMES]
    
    # Close the chart
    values = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(attr_names), endpoint=False).tolist()
    angles = angles + [angles[0]]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot
    ax.fill(angles, values, color='coral', alpha=0.25)
    ax.plot(angles, values, color='coral', linewidth=2, label='Selected (Avg)')
    
    # Reference
    ax.plot(angles, [0.5] * len(angles), color='gray', linestyle='--', label='Random')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attr_names, fontsize=12)
    ax.set_ylim(0, 1)
    
    for angle, val in zip(angles[:-1], values[:-1]):
        ax.annotate(f'{val:.2f}', xy=(angle, val), fontsize=10, ha='center')
        
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def visualize_aggregate_distribution(
    results: List[Dict[str, Any]],
    output_dir: str,
    prefix: str = "aggregate",
):
    """
    Visualize aggregated distribution metrics across multiple videos.
    
    Args:
        results: List of per-video distribution data
        output_dir: Directory to save visualizations
        prefix: Filename prefix
    """
    if not HAS_MATPLOTLIB or len(results) == 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    mean_percentiles = []
    top_10_coverages = []
    zscore_improvements = []
    above_p90_ratios = []
    
    for r in results:
        metrics = r.get("metrics", {})
        mean_percentiles.append(metrics.get("mean_percentile_rank", 0.5))
        top_10_coverages.append(metrics.get("top_10_coverage", 0))
        zscore_improvements.append(metrics.get("zscore_improvement", 0))
        above_p90_ratios.append(metrics.get("above_p90_ratio", 0))
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Mean percentile distribution (Top Left)
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(mean_percentiles, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='gray', linestyle='--', label='Random (0.5)')
    ax.axvline(np.mean(mean_percentiles), color='coral', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(mean_percentiles):.3f}')
    ax.set_xlabel('Mean Percentile Rank')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Mean Percentile Ranks', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Top-10 coverage distribution (Top Center)
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(top_10_coverages, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(top_10_coverages), color='steelblue', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(top_10_coverages):.1%}')
    ax.set_xlabel('Top-10% Coverage')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Top-10% Coverage', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Z-score improvement distribution (Top Right)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(zscore_improvements, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='gray', linestyle='--', label='No improvement')
    ax.axvline(np.mean(zscore_improvements), color='coral', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(zscore_improvements):.3f}')
    ax.set_xlabel('Z-Score Improvement')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Z-Score Improvements', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Scatter: Mean percentile vs Z-score (Bottom Left)
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(mean_percentiles, zscore_improvements, alpha=0.6, edgecolor='black')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Percentile Rank')
    ax.set_ylabel('Z-Score Improvement')
    ax.set_title('Percentile vs Z-Score', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Aggregate Radar Chart (Bottom Center - spanning 2 cols)
    # Calculate aggregate per-attribute percentiles
    agg_per_attr = {}
    for name in ATTR_NAMES:
        key = f"percentile_{name}"
        values = []
        for r in results:
            metrics = r.get("metrics", {})
            if key in metrics:
                values.append(metrics[key])
        if values:
            agg_per_attr[key] = float(np.mean(values))
            
    # Manually create polar axes at bottom right
    ax = fig.add_subplot(gs[1, 1:], polar=True)
    
    attr_names = [name.capitalize() for name in ATTR_NAMES]
    values = [agg_per_attr.get(f"percentile_{name}", 0.5) for name in ATTR_NAMES]
    
    # Close the chart
    values = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(attr_names), endpoint=False).tolist()
    angles = angles + [angles[0]]
    
    # Plot Radar
    ax.fill(angles, values, color='coral', alpha=0.25)
    ax.plot(angles, values, color='coral', linewidth=2, label='Dataset Avg')
    ax.plot(angles, [0.5] * len(angles), color='gray', linestyle='--', label='Random')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attr_names)
    ax.set_title("Aggregate Per-Attribute Percentiles", fontweight='bold', pad=20)
    ax.set_ylim(0, 1)

    
    fig.suptitle('Aggregate Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{prefix}_aggregate.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved aggregate visualization: {save_path}")


def main():
    """Main entry point for distribution visualization."""
    parser = argparse.ArgumentParser(description="Visualize keyframe selection distribution")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing distribution data JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for visualizations")
    parser.add_argument("--aggregate_only", action="store_true",
                       help="Only create aggregate visualization")
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all distribution data files
    data_files = list(Path(args.data_dir).glob("*_distribution.json"))
    
    if not data_files:
        print(f"No distribution data files found in {args.data_dir}")
        return
    
    print(f"Found {len(data_files)} distribution files")
    
    results = []
    
    for data_file in data_files:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        video_id = data.get("video_id", data_file.stem.replace("_distribution", ""))
        attrs_all = np.array(data.get("attrs_all", []))
        sel_idx = data.get("frame_indices_selected", [])
        metrics = data.get("metrics", {})
        
        if len(attrs_all) == 0:
            print(f"  Skipping {video_id}: no data")
            continue
        
        results.append({
            "video_id": video_id,
            "attrs_all": attrs_all,
            "sel_idx": sel_idx,
            "metrics": metrics,
        })
        
        if not args.aggregate_only:
            # Create per-video dashboard
            save_path = os.path.join(args.output_dir, f"{video_id}_dashboard.png")
            create_comprehensive_dashboard(attrs_all, sel_idx, metrics, save_path=save_path,
                                          title=f"Distribution Analysis: {video_id}")
            print(f"  Created dashboard: {video_id}")
    
    # Create aggregate visualization
    if results:
        visualize_aggregate_distribution(results, args.output_dir, "test_set")
        print(f"\n✅ Visualization complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
