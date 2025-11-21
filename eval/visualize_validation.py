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


def load_extra_metrics(val_output_dir: str):
    """Load and aggregate extra metrics from all videos across epochs"""
    val_dir = Path(val_output_dir)
    extra_metrics = {}
    
    for epoch_dir in sorted(val_dir.glob("ep*")):
        epoch_num = int(epoch_dir.name[2:])
        eval_results_dir = epoch_dir / "eval_results"
        
        if not eval_results_dir.exists():
            continue
        
        # Collect all extra_metrics.json files for this epoch
        epoch_metrics = []
        for extra_metrics_file in eval_results_dir.glob("*/extra_metrics.json"):
            try:
                with open(extra_metrics_file, 'r') as f:
                    metrics = json.load(f)
                    epoch_metrics.append(metrics)
            except Exception as e:
                print(f"⚠️  Failed to load {extra_metrics_file}: {e}")
        
        if epoch_metrics:
            # Aggregate metrics
            aggregated = {}
            metric_names = epoch_metrics[0].keys()
            
            for metric_name in metric_names:
                values = [m[metric_name] for m in epoch_metrics if metric_name in m and m[metric_name] is not None]
                if values:
                    aggregated[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            extra_metrics[epoch_num] = aggregated
    
    return extra_metrics


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


def plot_extra_metrics_over_epochs(extra_metrics, output_dir):
    """Plot extra metrics (LPIPS, MS-SWD) over epochs"""
    if not extra_metrics:
        print("⚠️  No extra metrics found")
        return
    
    epochs = sorted(extra_metrics.keys())
    
    # Determine which metrics are available
    all_metric_names = set()
    for ep_metrics in extra_metrics.values():
        all_metric_names.update(ep_metrics.keys())
    
    metric_names = sorted(all_metric_names)
    
    if not metric_names:
        print("⚠️  No metric names found in extra_metrics")
        return
    
    # Create subplots based on number of metrics
    n_metrics = len(metric_names)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric_name in enumerate(metric_names):
        means = []
        stds = []
        valid_epochs = []
        
        for ep in epochs:
            if metric_name in extra_metrics[ep]:
                means.append(extra_metrics[ep][metric_name]['mean'])
                stds.append(extra_metrics[ep][metric_name]['std'])
                valid_epochs.append(ep)
        
        if means:
            ax = axes[idx]
            means = np.array(means)
            stds = np.array(stds)
            
            # Plot with error bars
            ax.errorbar(valid_epochs, means, yerr=stds, marker='o', linewidth=2, 
                       markersize=8, capsize=5, capthick=2, label='Mean ± Std')
            ax.fill_between(valid_epochs, means - stds, means + stds, alpha=0.2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} over Epochs', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Mark best epoch (for LPIPS lower is better, for diversity higher might be better)
            if 'Gap' in metric_name or 'Color' in metric_name:  # Lower is better
                best_idx = np.argmin(means)
                best_label = 'Best (min)'
            else:  # Higher is better (diversity)
                best_idx = np.argmax(means)
                best_label = 'Best (max)'
            
            ax.scatter([valid_epochs[best_idx]], [means[best_idx]], 
                      color='red', s=200, zorder=5, marker='*', 
                      label=f'{best_label}: Epoch {valid_epochs[best_idx]}')
            ax.legend()
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'extra_metrics_over_epochs.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/extra_metrics_over_epochs.png")
    plt.close()


def plot_extra_metrics_detailed(extra_metrics, output_dir):
    """Plot detailed view of extra metrics with min/max ranges"""
    if not extra_metrics:
        print("⚠️  No extra metrics found")
        return
    
    epochs = sorted(extra_metrics.keys())
    all_metric_names = set()
    for ep_metrics in extra_metrics.values():
        all_metric_names.update(ep_metrics.keys())
    
    metric_names = sorted(all_metric_names)
    
    for metric_name in metric_names:
        means = []
        mins = []
        maxs = []
        valid_epochs = []
        
        for ep in epochs:
            if metric_name in extra_metrics[ep]:
                means.append(extra_metrics[ep][metric_name]['mean'])
                mins.append(extra_metrics[ep][metric_name]['min'])
                maxs.append(extra_metrics[ep][metric_name]['max'])
                valid_epochs.append(ep)
        
        if means:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            means = np.array(means)
            mins = np.array(mins)
            maxs = np.array(maxs)
            
            # Plot mean line
            ax.plot(valid_epochs, means, marker='o', linewidth=2.5, 
                   markersize=10, label='Mean', color='#2ecc71', zorder=3)
            
            # Plot min/max range
            ax.fill_between(valid_epochs, mins, maxs, alpha=0.2, 
                           color='#3498db', label='Min-Max Range')
            ax.plot(valid_epochs, mins, '--', linewidth=1.5, 
                   color='#3498db', alpha=0.7, label='Min')
            ax.plot(valid_epochs, maxs, '--', linewidth=1.5, 
                   color='#e74c3c', alpha=0.7, label='Max')
            
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel(metric_name, fontsize=14)
            ax.set_title(f'{metric_name} - Detailed View Across Epochs', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            # Mark best epoch
            if 'Gap' in metric_name or 'Color' in metric_name:
                best_idx = np.argmin(means)
            else:
                best_idx = np.argmax(means)
            
            ax.scatter([valid_epochs[best_idx]], [means[best_idx]], 
                      color='gold', s=300, zorder=5, marker='*', 
                      edgecolors='black', linewidths=2,
                      label=f'Best: Epoch {valid_epochs[best_idx]}')
            ax.legend(fontsize=11)
            
            plt.tight_layout()
            safe_name = metric_name.replace('/', '_').replace('@', '_at_')
            plt.savefig(Path(output_dir) / f'extra_metric_{safe_name}_detailed.png', 
                       dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {output_dir}/extra_metric_{safe_name}_detailed.png")
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
    
    # Load extra metrics
    print(f"\nLoading extra metrics from {args.val_output_dir}...")
    extra_metrics = load_extra_metrics(args.val_output_dir)
    
    if extra_metrics:
        print(f"✅ Loaded extra metrics for {len(extra_metrics)} epochs: {sorted(extra_metrics.keys())}")
        # Print summary of available metrics
        if extra_metrics:
            first_epoch = sorted(extra_metrics.keys())[0]
            metric_names = list(extra_metrics[first_epoch].keys())
            print(f"   Available metrics: {', '.join(metric_names)}")
    else:
        print("⚠️  No extra metrics found")
    
    # Output directory
    output_dir = args.output_dir if args.output_dir else Path(args.val_output_dir) / "plots"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot metrics over epochs
    if len(results) > 1:
        print("\n📊 Plotting metrics over epochs...")
        plot_metrics_over_epochs(results, output_dir)
    
    # Plot extra metrics over epochs
    if extra_metrics and len(extra_metrics) > 1:
        print("\n📊 Plotting extra metrics over epochs...")
        plot_extra_metrics_over_epochs(extra_metrics, output_dir)
        
        print("\n📊 Plotting detailed extra metrics...")
        plot_extra_metrics_detailed(extra_metrics, output_dir)
    
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
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/outputs/val_runs/advanced_v1 \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/outputs/val_runs/advanced_v1/plots
"""