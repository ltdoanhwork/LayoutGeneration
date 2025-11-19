#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize validation results from batch evaluation

Creates plots for:
1. Metrics comparison across epochs
2. Method comparison (DSN vs baselines)
3. Per-video performance
"""
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_validation_results(val_output_dir: str):
    """Load all validation results from different epochs"""
    val_dir = Path(val_output_dir)
    results = {}
    
    for epoch_dir in sorted(val_dir.glob("ep*")):
        epoch_num = int(epoch_dir.name[2:])
        summary_path = epoch_dir / "summary_results.json"
        
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                results[epoch_num] = json.load(f)
    
    return results


def plot_metrics_over_epochs(results, output_dir):
    """Plot how metrics change over epochs"""
    epochs = sorted(results.keys())
    
    metrics_to_plot = [
        'RecErr_mean',
        'Frechet_mean',
        'SceneCoverage_mean',
        'TemporalCoverage@tau_mean',
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        values = []
        for ep in epochs:
            agg = results[ep].get('aggregate_metrics', {})
            val = agg.get(metric)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values.append(val)
            else:
                values.append(None)
        
        # Plot
        valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        
        if valid_values:
            axes[idx].plot(valid_epochs, valid_values, marker='o', linewidth=2, markersize=8)
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel(metric.replace('_mean', ''), fontsize=12)
            axes[idx].set_title(f'{metric.replace("_mean", "")} over Epochs', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            # Mark best epoch
            if metric in ['RecErr_mean', 'Frechet_mean']:  # Lower is better
                best_idx = np.argmin(valid_values)
            else:  # Higher is better
                best_idx = np.argmax(valid_values)
            
            axes[idx].scatter([valid_epochs[best_idx]], [valid_values[best_idx]], 
                            color='red', s=200, zorder=5, marker='*', 
                            label=f'Best: Epoch {valid_epochs[best_idx]}')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'metrics_over_epochs.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/metrics_over_epochs.png")
    plt.close()


def plot_method_comparison(results, epoch, output_dir):
    """Compare DSN vs baselines for a specific epoch"""
    if epoch not in results:
        print(f"⚠️  Epoch {epoch} not found in results")
        return
    
    # Get first video's detailed results
    video_results = results[epoch].get('results', {})
    if not video_results:
        print("⚠️  No video results found")
        return
    
    # Take first video as example
    first_video = list(video_results.keys())[0]
    metrics = video_results[first_video]['metrics']
    
    methods = ['method', 'uniform', 'middle_of_scene', 'motion_peak']
    method_labels = ['DSN', 'Uniform', 'Middle of Scene', 'Motion Peak']
    
    metrics_to_compare = ['RecErr', 'Frechet', 'SceneCoverage', 'TemporalCoverage@tau']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metrics_to_compare):
        values = []
        for method in methods:
            if method in metrics:
                val = metrics[method].get(metric_name)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    values.append(val)
                else:
                    values.append(0)
            else:
                values.append(0)
        
        # Bar plot
        bars = axes[idx].bar(method_labels, values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        axes[idx].set_ylabel(metric_name, fontsize=12)
        axes[idx].set_title(f'{metric_name} Comparison (Epoch {epoch})', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=10)
        
        # Highlight best
        if metric_name in ['RecErr', 'Frechet']:  # Lower is better
            best_idx = np.argmin(values)
        else:  # Higher is better
            best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'method_comparison_ep{epoch}.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/method_comparison_ep{epoch}.png")
    plt.close()


def plot_per_video_performance(results, epoch, output_dir):
    """Plot per-video performance for DSN method"""
    if epoch not in results:
        print(f"⚠️  Epoch {epoch} not found in results")
        return
    
    video_results = results[epoch].get('results', {})
    if not video_results:
        print("⚠️  No video results found")
        return
    
    videos = sorted(video_results.keys())
    rec_errs = []
    frechets = []
    
    for vid in videos:
        m = video_results[vid]['metrics'].get('method', {})
        rec_errs.append(m.get('RecErr', 0))
        frechets.append(m.get('Frechet', 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # RecErr
    ax1.bar(range(len(videos)), rec_errs, color='#3498db', alpha=0.7)
    ax1.set_xlabel('Video Index', fontsize=12)
    ax1.set_ylabel('RecErr', fontsize=12)
    ax1.set_title(f'RecErr per Video (Epoch {epoch})', fontsize=14, fontweight='bold')
    ax1.axhline(y=np.mean(rec_errs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rec_errs):.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Frechet
    ax2.bar(range(len(videos)), frechets, color='#e74c3c', alpha=0.7)
    ax2.set_xlabel('Video Index', fontsize=12)
    ax2.set_ylabel('Frechet Distance', fontsize=12)
    ax2.set_title(f'Frechet Distance per Video (Epoch {epoch})', fontsize=14, fontweight='bold')
    ax2.axhline(y=np.mean(frechets), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(frechets):.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'per_video_performance_ep{epoch}.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/per_video_performance_ep{epoch}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize validation results")
    parser.add_argument("--val_output_dir", required=True, help="Path to validation output directory")
    parser.add_argument("--output_dir", default=None, help="Where to save plots (default: val_output_dir/plots)")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch to visualize (default: latest)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading validation results from {args.val_output_dir}...")
    results = load_validation_results(args.val_output_dir)
    
    if not results:
        print("❌ No validation results found")
        return
    
    print(f"✅ Loaded results for {len(results)} epochs: {sorted(results.keys())}")
    
    # Output directory
    output_dir = args.output_dir if args.output_dir else Path(args.val_output_dir) / "plots"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot metrics over epochs
    if len(results) > 1:
        print("\n📊 Plotting metrics over epochs...")
        plot_metrics_over_epochs(results, output_dir)
    
    # Plot method comparison and per-video for specific epoch
    epoch_to_plot = args.epoch if args.epoch else max(results.keys())
    print(f"\n📊 Plotting method comparison for epoch {epoch_to_plot}...")
    plot_method_comparison(results, epoch_to_plot, output_dir)
    
    print(f"\n📊 Plotting per-video performance for epoch {epoch_to_plot}...")
    plot_per_video_performance(results, epoch_to_plot, output_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""
python -m eval.visualize_validation \
    --val_output_dir outputs/val_runs/test_no_ckpt \
    --output_dir outputs/val_runs/run1/plots
"""