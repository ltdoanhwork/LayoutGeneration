#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep CLIP-IQA Ablation Visualization
Creates line charts for metrics across epochs.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Define the experiment groups for Deep IQA
ABLATION_GROUPS = {
    "1_prompt_sensitivity": {
        "name": "Prompt Engineering Sensitivity",
        "experiments": ["G1_prompt_1pair", "G1_prompt_2pair", "G1_prompt_3pair", "G1_prompt_4pair", "G1_prompt_5pair"],
        "labels": ["1 Pair", "2 Pairs", "3 Pairs (Baseline)", "4 Pairs", "5 Pairs (Augmented)"],
    },
    "2_component_isolation": {
        "name": "Component Contribution",
        # Removed G2_baseline_plain as requested
        "experiments": ["G2_track_A_feat", "G2_track_B_reward", "G1_prompt_3pair"],
        "labels": ["Feat Only", "Reward Only", "Combined (Full)"],
        # Only show MPR and Top10
        "metrics_whitelist": ["mpr", "top10"]
    },
}

METRICS = {
    "mpr": {"name": "MPR", "higher_better": True, "category": "Aesthetic"},
    "top10": {"name": "Top10", "higher_better": True, "category": "Aesthetic"},
    "RecErr": {"name": "RecErr", "higher_better": False, "category": "Representativeness"},
    "Frechet": {"name": "Frechet", "higher_better": False, "category": "Representativeness"},
    "TempCov": {"name": "TempCov", "higher_better": True, "category": "Diversity"},
    "composite_score": {"name": "Composite", "higher_better": True, "category": "Overall"},
    # Add LPIPS/DISTS Gap if available (usually only in final output, but loop logic handles missing keys)
}

def load_all_epochs(exp_dir: Path) -> Dict[int, Dict[str, float]]:
    data = {}
    if not exp_dir.exists():
        return data
        
    for ep_dir in exp_dir.glob("ep*"):
        if not ep_dir.is_dir(): continue
        try:
            ep_num = int(ep_dir.name.replace("ep", ""))
            json_path = ep_dir / "val_results.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data[ep_num] = json.load(f)
        except:
            continue
    return data

def plot_group_metrics(root_dir: Path, output_dir: Path, group_key: str, group_info: Dict):
    if not HAS_MATPLOTLIB: return

    # Load data
    exp_data = {}
    for exp_name in group_info["experiments"]:
        exp_path = root_dir / exp_name
        data = load_all_epochs(exp_path)
        if data:
            exp_data[exp_name] = data
    
    if not exp_data:
        print(f"Skipping {group_key}: No data found.")
        return

    # Filter metrics if whitelist exists
    metric_keys = list(METRICS.keys())
    if "metrics_whitelist" in group_info:
        metric_keys = [m for m in metric_keys if m in group_info["metrics_whitelist"]]

    ncols = 2
    nrows = (len(metric_keys) + 1) // 2
    if len(metric_keys) == 1:
        nrows, ncols = 1, 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    if len(metric_keys) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_info["experiments"])))
    
    for idx, m_key in enumerate(metric_keys):
        # Handle cases where we have fewer metrics than subplots
        if idx >= len(axes): break
        
        ax = axes[idx]
        m_info = METRICS[m_key]
        
        for i, exp_name in enumerate(group_info["experiments"]):
            if exp_name not in exp_data: continue
            
            label = group_info["labels"][i]
            color = colors[i]
            
            epochs = sorted(exp_data[exp_name].keys())
            values = [exp_data[exp_name][e].get(m_key, np.nan) for e in epochs]
            
            # Simple Smoothing
            if len(values) > 5:
                values_smooth = np.convolve(values, np.ones(3)/3, mode='same')
                ax.plot(epochs, values_smooth, '-', color=color, alpha=0.8, linewidth=1.5)
                ax.plot(epochs, values, 'o', color=color, alpha=0.3, markersize=3)
            else:
                ax.plot(epochs, values, 'o-', color=color, label=label, linewidth=2)
                
            # Add final legend entry only once
            if idx == 0: # Hack to handle legend
                pass

        ax.set_title(f"{m_info['name']} ({m_info['category']})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(m_info['name'])
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(metric_keys), len(axes)):
        axes[i].axis('off')

    # Global Legend
    handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(group_info["experiments"]))]
    fig.legend(handles, group_info["labels"], loc='lower center', ncol=min(len(group_info["labels"]), 4), bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08 + (0.02 * (len(group_info["labels"]) // 5))) # Adjust bottom based on legend rows
    
    out_file = output_dir / f"{group_key}_metrics.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved plot: {out_file}")

def plot_prompt_contribution(root_dir: Path, output_dir: Path):
    """
    Analyze and plot the contribution of prompt pairs (1-5) on MPR and Top10.
    Uses the '1_prompt_sensitivity' group data.
    """
    if not HAS_MATPLOTLIB: return
    
    group_info = ABLATION_GROUPS["1_prompt_sensitivity"]
    metrics_to_plot = ["mpr", "top10"]
    
    # x-axis: Number of pairs (extracted from experiment names or labels)
    x_pairs = [1, 2, 3, 4, 5]
    
    # Store best values
    best_results = {m: [] for m in metrics_to_plot}
    
    print("\n--- Prompt Contribution Analysis ---")
    
    for i, exp_name in enumerate(group_info["experiments"]):
        exp_path = root_dir / exp_name
        data = load_all_epochs(exp_path)
        
        if not data:
            print(f"Warning: No data for {exp_name}")
            for m in metrics_to_plot:
                best_results[m].append(0.0)
            continue
            
        # Find best value for each metric across all epochs
        for m in metrics_to_plot:
            values = [data[e].get(m, 0.0) for e in data]
            best_val = max(values) if values else 0.0
            best_results[m].append(best_val)
            
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot MPR
    ax = axes[0]
    ax.plot(x_pairs, best_results["mpr"], 'o-', color='tab:blue', linewidth=2, markersize=8)
    ax.set_title("MPR vs Number of Prompt Pairs")
    ax.set_xlabel("Number of Prompt Pairs")
    ax.set_ylabel("Best MPR Score")
    ax.set_xticks(x_pairs)
    ax.grid(True, alpha=0.3)
    
    # Annotate values
    for x, y in zip(x_pairs, best_results["mpr"]):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    # Plot Top10
    ax = axes[1]
    ax.plot(x_pairs, best_results["top10"], 's-', color='tab:green', linewidth=2, markersize=8)
    ax.set_title("Top10 Recall vs Number of Prompt Pairs")
    ax.set_xlabel("Number of Prompt Pairs")
    ax.set_ylabel("Best Top10 Score")
    ax.set_xticks(x_pairs)
    ax.grid(True, alpha=0.3)
    
    # Annotate values
    for x, y in zip(x_pairs, best_results["top10"]):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    out_file = output_dir / "3_prompt_contribution_analysis.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved prompt contribution plot: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="runs/ablation_deep_iqa")
    parser.add_argument("--out", type=str, default="runs/ablation_deep_iqa/plots")
    args = parser.parse_args()
    
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot Standard Groups
    for key, info in ABLATION_GROUPS.items():
        plot_group_metrics(root, out, key, info)
        
    # 2. Plot Prompt Contribution Analysis
    plot_prompt_contribution(root, out)

if __name__ == "__main__":
    main()
