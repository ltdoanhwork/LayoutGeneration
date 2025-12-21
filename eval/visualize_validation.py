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
            try:
                with open(summary_path, 'r') as f:
                    results[epoch_num] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping {summary_path}: Invalid or empty JSON file ({e})")
            except Exception as e:
                print(f"⚠️  Skipping {summary_path}: {e}")
    
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


def plot_anime_quality_over_epochs(results, output_dir):
    """Plot V6 anime quality improvement metrics over epochs"""
    epochs = sorted(results.keys())
    
    # Per-attribute anime metrics
    attr_metrics = [
        'Anime_Sharpness_Mean', 'Anime_Colorfulness_Mean', 'Anime_Brightness_Mean',
        'Anime_Sakuga_Mean', 'Anime_Cinematic_Mean', 'Anime_Expression_Mean'
    ]
    
    # Check if any anime metrics exist
    has_anime_metrics = False
    for ep in epochs:
        aq = results[ep].get('anime_quality_metrics', {})
        if any(aq.get(m) is not None for m in attr_metrics):
            has_anime_metrics = True
            break
    
    if not has_anime_metrics:
        print("⚠️  No anime quality metrics found")
        return
    
    # Create per-attribute plot
    n_metrics = len(attr_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    for idx, metric in enumerate(attr_metrics):
        values = []
        stds = []
        std_metric = metric.replace('_Mean', '_Std')
        
        for ep in epochs:
            aq = results[ep].get('anime_quality_metrics', {})
            val = aq.get(metric)
            std = aq.get(std_metric)
            
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values.append(val)
                stds.append(std if std is not None else 0.0)
            else:
                values.append(None)
                stds.append(None)
        
        # Filter valid
        valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        valid_stds = [s for s, v in zip(stds, values) if v is not None]
        
        if valid_values:
            ax = axes[idx]
            valid_values = np.array(valid_values)
            valid_stds = np.array(valid_stds)
            
            ax.errorbar(valid_epochs, valid_values, yerr=valid_stds, 
                       marker='o', linewidth=2, markersize=8, capsize=5,
                       color='#2980b9', label='Mean ± Std')
            ax.fill_between(valid_epochs, valid_values - valid_stds, 
                           valid_values + valid_stds, alpha=0.2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.replace('_Mean', ''), fontsize=12)
            ax.set_title(f'{metric.replace("_Mean", "")} over Epochs', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Mark best epoch (higher is better for quality)
            best_idx = np.argmax(valid_values)
            ax.scatter([valid_epochs[best_idx]], [valid_values[best_idx]], 
                      color='red', s=200, zorder=5, marker='*', 
                      label=f'Best (max): Epoch {valid_epochs[best_idx]}')
            ax.legend()
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'anime_quality_over_epochs.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/anime_quality_over_epochs.png")
    plt.close()
    
    # Plot Top-10 metrics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    top10_metrics = [
        ('Top10_Recall_mean', 'Top-10 Recall', '#27ae60', ax1),
        ('Top10_Precision_mean', 'Top-10 Precision', '#e67e22', ax2),
        ('Quality_Improvement_mean', 'Quality Improvement', '#8e44ad', ax3),
    ]
    
    for metric, label, color, ax in top10_metrics:
        values = []
        for ep in epochs:
            aq = results[ep].get('anime_quality_metrics', {})
            val = aq.get(metric)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values.append(val)
            else:
                values.append(None)
        
        valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        
        if valid_values:
            ax.plot(valid_epochs, valid_values, marker='o', linewidth=2.5, 
                   markersize=10, color=color, label=label)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f'{label} over Epochs', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Best marker
            best_idx = np.argmax(valid_values)
            ax.scatter([valid_epochs[best_idx]], [valid_values[best_idx]], 
                      color='red', s=200, zorder=5, marker='*', 
                      label=f'Best: Epoch {valid_epochs[best_idx]}')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'quality_improvement_over_epochs.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/quality_improvement_over_epochs.png")
    plt.close()


def plot_v8_lagrangian_over_epochs(val_output_dir: str, output_dir: str):
    """
    V8: Plot Lagrangian multipliers (lambda_rec, lambda_cov, lambda_div) over epochs.
    
    These are saved in the TensorBoard logs but can also be in summary_results.json.
    """
    val_dir = Path(val_output_dir)
    epochs = []
    lambda_rec = []
    lambda_cov = []
    lambda_div = []
    
    for epoch_dir in sorted(val_dir.glob("ep*")):
        epoch_num = int(epoch_dir.name[2:])
        summary_path = epoch_dir / "summary_results.json"
        
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                
                # Check if V8 metrics exist
                v8_metrics = data.get("v8_metrics", {})
                if v8_metrics:
                    epochs.append(epoch_num)
                    lambda_rec.append(v8_metrics.get("lambda_rec", None))
                    lambda_cov.append(v8_metrics.get("lambda_cov", None))
                    lambda_div.append(v8_metrics.get("lambda_div", None))
            except:
                pass
    
    if not epochs:
        print("⚠️  No V8 Lagrangian metrics found in validation results")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    lambdas = [
        (lambda_rec, 'λ_rec (RecErr)', '#e74c3c'),
        (lambda_cov, 'λ_cov (Coverage)', '#3498db'),
        (lambda_div, 'λ_div (Diversity)', '#2ecc71'),
    ]
    
    for idx, (values, label, color) in enumerate(lambdas):
        valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        
        if valid_values:
            axes[idx].plot(valid_epochs, valid_values, marker='o', linewidth=2.5, 
                          markersize=10, color=color, label=label)
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel('Multiplier Value', fontsize=12)
            axes[idx].set_title(f'{label} over Epochs', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'v8_lagrangian_multipliers.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/v8_lagrangian_multipliers.png")
    plt.close()


def plot_v8_constraint_satisfaction(val_output_dir: str, output_dir: str, 
                                     rec_threshold: float = 0.35,
                                     cov_threshold: float = 0.3,
                                     div_threshold: float = 0.25):
    """
    V8: Plot constraint satisfaction rates over epochs.
    
    Shows what percentage of videos satisfy each constraint.
    """
    val_dir = Path(val_output_dir)
    epochs = []
    rec_err_rates = []
    
    for epoch_dir in sorted(val_dir.glob("ep*")):
        epoch_num = int(epoch_dir.name[2:])
        summary_path = epoch_dir / "summary_results.json"
        
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                
                agg = data.get("aggregate_metrics", {})
                rec_err = agg.get("RecErr_mean")
                
                if rec_err is not None:
                    epochs.append(epoch_num)
                    # Check if constraint is satisfied
                    rec_err_rates.append(1.0 if rec_err <= rec_threshold else 0.0)
            except:
                pass
    
    if not epochs:
        print("⚠️  No constraint metrics found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(epochs, rec_err_rates, alpha=0.3, color='#2ecc71')
    ax.plot(epochs, rec_err_rates, marker='o', linewidth=2.5, markersize=10, 
            color='#27ae60', label=f'RecErr ≤ {rec_threshold}')
    
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target (100%)')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Satisfaction Rate', fontsize=12)
    ax.set_title('V8 Constraint Satisfaction over Epochs', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'v8_constraint_satisfaction.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/v8_constraint_satisfaction.png")
    plt.close()


def plot_v8_gating_weights(val_output_dir: str, output_dir: str):
    """
    V8: Plot gating weight (alpha_t) statistics over epochs.
    
    Shows mean, std, and distribution of alpha values.
    """
    val_dir = Path(val_output_dir)
    epochs = []
    gating_means = []
    gating_stds = []
    rec_dominant_rates = []
    
    for epoch_dir in sorted(val_dir.glob("ep*")):
        epoch_num = int(epoch_dir.name[2:])
        summary_path = epoch_dir / "summary_results.json"
        
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                
                v8_metrics = data.get("v8_metrics", {})
                if v8_metrics:
                    gating_mean = v8_metrics.get("gating_mean")
                    gating_std = v8_metrics.get("gating_std")
                    rec_dominant = v8_metrics.get("gating_rec_dominant")
                    
                    if gating_mean is not None:
                        epochs.append(epoch_num)
                        gating_means.append(gating_mean)
                        gating_stds.append(gating_std if gating_std else 0.0)
                        rec_dominant_rates.append(rec_dominant if rec_dominant else 0.5)
            except:
                pass
    
    if not epochs:
        print("⚠️  No V8 gating metrics found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Mean gating weight
    gating_means = np.array(gating_means)
    gating_stds = np.array(gating_stds)
    
    ax1.errorbar(epochs, gating_means, yerr=gating_stds, marker='o', linewidth=2.5,
                markersize=10, capsize=5, color='#9b59b6', label='Mean α_t ± Std')
    ax1.fill_between(epochs, gating_means - gating_stds, gating_means + gating_stds, alpha=0.2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balance (0.5)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Gating Weight (α_t)', fontsize=12)
    ax1.set_title('Mean Gating Weight over Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Rec-dominant rate
    ax2.bar(epochs, rec_dominant_rates, color='#3498db', alpha=0.7, label='Rec-Dominant')
    ax2.bar(epochs, [1 - r for r in rec_dominant_rates], bottom=rec_dominant_rates,
            color='#e74c3c', alpha=0.7, label='Anime-Dominant')
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Fraction of Frames', fontsize=12)
    ax2.set_title('Gating Distribution: Rec vs Anime Dominant', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'v8_gating_weights.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/v8_gating_weights.png")
    plt.close()


def plot_v8_combined_dashboard(results: dict, output_dir: str):
    """
    V8: Create combined dashboard with all key V8 metrics.
    """
    epochs = sorted(results.keys())
    
    fig = plt.figure(figsize=(20, 12))
    
    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Anime Quality Improvement (PRIMARY OBJECTIVE)
    ax1 = fig.add_subplot(gs[0, 0])
    quality_imp = []
    for ep in epochs:
        aq = results[ep].get('anime_quality_metrics', {})
        val = aq.get('Quality_Improvement_mean')
        quality_imp.append(val if val is not None else 0.0)
    
    ax1.plot(epochs, quality_imp, marker='o', linewidth=2.5, color='#8e44ad', markersize=8)
    ax1.fill_between(epochs, 0, quality_imp, alpha=0.2, color='#8e44ad')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Quality Improvement')
    ax1.set_title('🎨 Anime Quality (PRIMARY)', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. RecErr (CONSTRAINT)
    ax2 = fig.add_subplot(gs[0, 1])
    rec_errs = []
    for ep in epochs:
        agg = results[ep].get('aggregate_metrics', {})
        val = agg.get('RecErr_mean')
        rec_errs.append(val if val is not None else 1.0)
    
    ax2.plot(epochs, rec_errs, marker='s', linewidth=2.5, color='#e74c3c', markersize=8)
    ax2.axhline(y=0.35, color='green', linestyle='--', label='Threshold (0.35)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RecErr')
    ax2.set_title('📏 RecErr (CONSTRAINT)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Top-10 Recall
    ax3 = fig.add_subplot(gs[0, 2])
    top10_recall = []
    for ep in epochs:
        aq = results[ep].get('anime_quality_metrics', {})
        val = aq.get('Top10_Recall_mean')
        top10_recall.append(val if val is not None else 0.0)
    
    ax3.plot(epochs, top10_recall, marker='^', linewidth=2.5, color='#27ae60', markersize=8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Top-10 Recall')
    ax3.set_title('🎯 Top-10 Recall (Outlier Hunting)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-attribute quality
    ax4 = fig.add_subplot(gs[1, :2])
    attrs = ['Sakuga', 'Cinematic', 'Sharpness', 'Colorfulness']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for attr, color in zip(attrs, colors):
        attr_vals = []
        for ep in epochs:
            aq = results[ep].get('anime_quality_metrics', {})
            val = aq.get(f'Anime_{attr}_Mean')
            attr_vals.append(val if val is not None else 0.0)
        ax4.plot(epochs, attr_vals, marker='o', linewidth=2, color=color, 
                label=attr, markersize=6, alpha=0.8)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Attribute Score')
    ax4.set_title('🎭 Per-Attribute Quality Scores', fontsize=12, fontweight='bold')
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Find best epoch
    if quality_imp:
        best_anime_epoch = epochs[np.argmax(quality_imp)]
        best_anime_val = max(quality_imp)
    else:
        best_anime_epoch = epochs[-1]
        best_anime_val = 0.0
    
    if rec_errs:
        best_rec_epoch = epochs[np.argmin(rec_errs)]
        best_rec_val = min(rec_errs)
    else:
        best_rec_epoch = epochs[-1]
        best_rec_val = 1.0
    
    summary_text = f"""
V8 Training Summary
═══════════════════════

🏆 Best Anime Quality
   Epoch: {best_anime_epoch}
   Value: {best_anime_val:.4f}

📏 Best RecErr
   Epoch: {best_rec_epoch}
   Value: {best_rec_val:.4f}

📊 Final Epoch ({epochs[-1]})
   Quality: {quality_imp[-1] if quality_imp else 0:.4f}
   RecErr: {rec_errs[-1] if rec_errs else 1:.4f}
   Top-10 Recall: {top10_recall[-1] if top10_recall else 0:.2%}
"""
    
    ax5.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax5.transAxes,
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.suptitle('V8 Constrained MORL Training Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(Path(output_dir) / 'v8_dashboard.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/v8_dashboard.png")
    plt.close()


def main():

    parser = argparse.ArgumentParser(description="Visualize validation results")
    parser.add_argument("--val_output_dir", required=True, help="Path to validation output directory")
    parser.add_argument("--output_dir", default=None, help="Where to save plots (default: val_output_dir/plots)")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch to visualize (default: latest)")
    parser.add_argument("--save_images", action="store_true", help="Automatically save all visualizations")
    
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
    
    # V6: Plot anime quality improvement metrics
    if len(results) > 1:
        print("\n📊 Plotting V6 anime quality improvement metrics...")
        plot_anime_quality_over_epochs(results, output_dir)
    
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
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_raft_motion/val_runs/dsn_raft_motion \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_raft_motion/val_runs/dsn_raft_motion/plots

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_advanced_v1_no_motion_100_samples_test_sakura \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_advanced_v1_no_motion_100_samples_test_sakura/plots

 python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_runs_baseline_100_samples/val_runs/baseline_v1\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_runs_baseline_100_samples/val_runs/baseline_v1/plots

 python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_a_features/val_runs\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_a_features/val_runs/plots

 python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_b_rewards/val_runs\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_b_rewards/val_runs/plots

 python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_c_combined/val_runs\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_c_combined/val_runs/plots

  python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_d_anime/val_runs\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_d_anime/val_runs/plots  

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_d_anime_reward_v2/val_runs\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_track_d_anime_reward_v2/val_runs/plots 

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_rl_multi_video/val_runs\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_rl_multi_video/val_runs/plots

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_anime_premium_v1/val_runs\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_anime_premium_v1/val_runs/plots 

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_anime_v3/val_runs\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_anime_v3/val_runs/plots 

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_anime_v4/val_runs\
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_anime_v4/val_runs/plots     

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_anime_v4_scaled/checkpoint_eval \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_anime_v4_scaled/plots  

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v5_plus/val_runs \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v5_plus/plots 
python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v5_anime_focus/val_runs \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v5_anime_focus/plots 

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v5_plus_transnetv2/val_runs \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v5_plus_transnetv2/plots 

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v5_large_transnetv2/val_runs \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v5_large_transnetv2/plots 

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v6_quality_aligned/val_runs \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v6_quality_aligned/plots 

python -m eval.visualize_validation \
    --val_output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v7_dual_objective/val_runs \
    --output_dir /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v7_dual_objective/plots 


""" 