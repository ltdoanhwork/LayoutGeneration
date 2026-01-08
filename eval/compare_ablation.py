#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Comparison Visualization

Creates comparison plots for ablation study results across epochs.
Generates per-group comparison charts for all 6 metrics + per-attribute breakdown.
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Ablation experiment groups (5 consolidated groups)
ABLATION_GROUPS = {
    "1_reward": {
        "name": "Reward Design",
        "experiments": ["1_baseline", "1_no_div", "1_strong_div", "1_rec_opt", "1_frechet_opt", "1_combined_rep"],
        "labels": ["Baseline", "No Div", "Strong Div", "RecErr Opt", "Frechet Opt", "Combined Rep"],
    },
    "2_features": {
        "name": "Input & Features",
        "experiments": ["2_full_features", "2_visual_only", "2_low_feat_dim", "2_high_feat_dim"],
        "labels": ["Full (CLIP+Anime)", "Visual Only", "Low Dim (256)", "High Dim (768)"],
    },
    "3_architecture": {
        "name": "Architecture",
        "experiments": ["3_no_attn", "3_attn_1L", "3_attn_2L", "3_attn_4L", "3_gate_small", "3_gate_large", "3_lstm_small", "3_lstm_large"],
        "labels": ["No Attn", "1 Layer", "2 Layers", "4 Layers", "Gate Small", "Gate Large", "LSTM Small", "LSTM Large"],
    },
    "4_budget_exploration": {
        "name": "Budget & Exploration",
        "experiments": ["4_budget_low", "4_budget_default", "4_budget_high", "4_low_entropy", "4_high_entropy"],
        "labels": ["Budget 10%", "Budget 15%", "Budget 25%", "Low Entropy", "High Entropy"],
    },
    "5_learning_rate": {
        "name": "Learning Rate",
        "experiments": ["5_lr_1e-5", "5_lr_1e-4", "5_lr_5e-4"],
        "labels": ["LR=1e-5", "LR=1e-4", "LR=5e-4"],
    },
}

# Metrics to compare
METRICS = {
    "mpr": {"name": "MPR", "higher_better": True, "category": "Aesthetic"},
    "top10": {"name": "Top10", "higher_better": True, "category": "Aesthetic"},
    "RecErr": {"name": "RecErr", "higher_better": False, "category": "Representativeness"},
    "Frechet": {"name": "Frechet", "higher_better": False, "category": "Representativeness"},
    "TempCov": {"name": "TempCov", "higher_better": True, "category": "Diversity"},
    "composite_score": {"name": "Composite", "higher_better": True, "category": "Overall"},
}

ATTR_NAMES = ["sharpness", "colorfulness", "brightness", "sakuga", "cinematic", "expression"]


def load_experiment_metrics(exp_dir: Path) -> Dict[int, Dict[str, float]]:
    """Load all epoch metrics for an experiment."""
    metrics_by_epoch = {}
    
    for epoch_dir in sorted(exp_dir.glob("ep*")):
        if not epoch_dir.is_dir():
            continue
        
        val_results = epoch_dir / "val_results.json"
        if not val_results.exists():
            continue
        
        try:
            epoch_num = int(epoch_dir.name.replace("ep", ""))
            with open(val_results) as f:
                metrics_by_epoch[epoch_num] = json.load(f)
        except Exception as e:
            print(f"Error loading {val_results}: {e}")
            continue
    
    return metrics_by_epoch


def create_metric_comparison_plot(
    group_data: Dict[str, Dict[int, float]],
    labels: List[str],
    metric_name: str,
    higher_better: bool,
    save_path: str,
    title: str,
):
    """Create line plot comparing a metric across experiments."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_data)))
    
    for (exp_name, epoch_data), label, color in zip(group_data.items(), labels, colors):
        epochs = sorted(epoch_data.keys())
        values = [epoch_data[e] for e in epochs]
        
        ax.plot(epochs, values, 'o-', label=label, color=color, markersize=4, linewidth=2)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add direction indicator
    direction = "↑ Higher is better" if higher_better else "↓ Lower is better"
    ax.text(0.02, 0.98, direction, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', style='italic', color='gray')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_group_comparison_dashboard(
    ablation_root: Path,
    group_key: str,
    group_info: Dict[str, Any],
    output_dir: Path,
):
    """Create comprehensive dashboard for an ablation group."""
    if not HAS_MATPLOTLIB:
        return
    
    # Load data for all experiments in group
    all_data = {}
    for exp_name in group_info["experiments"]:
        exp_dir = ablation_root / exp_name
        if exp_dir.exists():
            all_data[exp_name] = load_experiment_metrics(exp_dir)
    
    if not all_data:
        print(f"No data found for group {group_key}")
        return
    
    group_out_dir = output_dir / group_key
    group_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create per-metric comparison plots
    for metric_key, metric_info in METRICS.items():
        metric_data = {}
        for exp_name, epochs_data in all_data.items():
            metric_data[exp_name] = {
                e: d.get(metric_key, 0.0) for e, d in epochs_data.items()
            }
        
        save_path = group_out_dir / f"{metric_key}_comparison.png"
        create_metric_comparison_plot(
            metric_data,
            group_info["labels"][:len(metric_data)],
            metric_info["name"],
            metric_info["higher_better"],
            str(save_path),
            f"{group_info['name']}: {metric_info['name']} ({metric_info['category']})"
        )
    
    # Create per-attribute radar comparison (final epoch)
    create_final_radar_comparison(all_data, group_info, group_out_dir)
    
    # Create summary table
    create_summary_table(all_data, group_info, group_out_dir)
    
    print(f"✅ Created comparison for: {group_info['name']}")


def create_final_radar_comparison(
    all_data: Dict[str, Dict[int, Dict[str, float]]],
    group_info: Dict[str, Any],
    output_dir: Path,
):
    """Create radar chart comparing final epoch per-attribute metrics."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    attr_names = [name.capitalize() for name in ATTR_NAMES]
    angles = np.linspace(0, 2 * np.pi, len(attr_names), endpoint=False).tolist()
    angles += [angles[0]]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    
    for (exp_name, epochs_data), label, color in zip(
        all_data.items(), 
        group_info["labels"][:len(all_data)], 
        colors
    ):
        if not epochs_data:
            continue
        
        # Get final epoch
        final_epoch = max(epochs_data.keys())
        final_data = epochs_data[final_epoch]
        
        values = [final_data.get(f"percentile_{name}", 0.5) for name in ATTR_NAMES]
        values += [values[0]]
        
        ax.plot(angles, values, 'o-', label=label, color=color, linewidth=2, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Reference line
    ax.plot(angles, [0.5] * len(angles), 'k--', alpha=0.3, label='Random')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attr_names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"{group_info['name']}: Per-Attribute Comparison (Final Epoch)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    fig.savefig(output_dir / "radar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_summary_table(
    all_data: Dict[str, Dict[int, Dict[str, float]]],
    group_info: Dict[str, Any],
    output_dir: Path,
):
    """Create summary JSON comparing final metrics across experiments."""
    summary = {
        "group": group_info["name"],
        "experiments": {},
    }
    
    for exp_name, epochs_data in all_data.items():
        if not epochs_data:
            continue
        
        final_epoch = max(epochs_data.keys())
        final_metrics = epochs_data[final_epoch]
        
        exp_idx = group_info["experiments"].index(exp_name) if exp_name in group_info["experiments"] else -1
        label = group_info["labels"][exp_idx] if exp_idx >= 0 else exp_name
        
        summary["experiments"][label] = {
            "final_epoch": final_epoch,
            **{k: final_metrics.get(k, 0.0) for k in METRICS.keys()},
            "per_attr": {
                name: final_metrics.get(f"percentile_{name}", 0.5) for name in ATTR_NAMES
            }
        }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def create_all_metrics_overview(ablation_root: Path, output_dir: Path):
    """Create overview comparing best results across all experiments."""
    if not HAS_MATPLOTLIB:
        return
    
    all_experiments = {}
    
    # Load all experiments
    for exp_dir in sorted(ablation_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        
        epochs_data = load_experiment_metrics(exp_dir)
        if epochs_data:
            # Get best composite score
            best_epoch = max(epochs_data.keys(), 
                           key=lambda e: epochs_data[e].get("composite_score", 0))
            all_experiments[exp_dir.name] = {
                "best_epoch": best_epoch,
                "metrics": epochs_data[best_epoch]
            }
    
    if not all_experiments:
        return
    
    # Create bar chart for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    exp_names = list(all_experiments.keys())
    x = np.arange(len(exp_names))
    
    for i, (metric_key, metric_info) in enumerate(METRICS.items()):
        ax = axes[i]
        values = [all_experiments[name]["metrics"].get(metric_key, 0) for name in exp_names]
        
        colors = ['coral' if metric_info["higher_better"] else 'steelblue' for _ in values]
        if metric_info["higher_better"]:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        colors[best_idx] = 'green'
        
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(f"{metric_info['name']} ({metric_info['category']})", fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    fig.suptitle("Ablation Study: Overview of All Experiments", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "overview_all_experiments.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare Ablation Study Results")
    parser.add_argument("--ablation_root", type=str, required=True,
                       help="Root directory containing ablation experiments")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for comparison plots")
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib required for visualization")
        return
    
    ablation_root = Path(args.ablation_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading ablation results from: {ablation_root}")
    print(f"Output directory: {output_dir}")
    
    # Create per-group comparisons
    for group_key, group_info in ABLATION_GROUPS.items():
        create_group_comparison_dashboard(ablation_root, group_key, group_info, output_dir)
    
    # Create overall overview
    create_all_metrics_overview(ablation_root, output_dir)
    
    print(f"\n✅ Ablation comparison complete! Results in: {output_dir}")


if __name__ == "__main__":
    main()
