#!/usr/bin/env python3
"""
Analyze and visualize ablation experiment results.

Usage:
    python analyze_ablation_results.py <ablation_results_dir>

Example:
    python analyze_ablation_results.py ablation_results/
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_ablation_results(base_dir):
    """Load all ablation experiment results from CSV files."""
    modes = ['full', 'wo_cap', 'wo_asp', 'wo_ov', 'only_cap', 'only_asp', 'only_ov']
    results = []
    
    for mode in modes:
        csv_path = os.path.join(base_dir, mode, 'layout_metrics.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results.append(df)
        else:
            print(f"Warning: Missing metrics for {mode}: {csv_path}")
    
    if not results:
        print("Error: No results found!")
        return None
    
    return pd.concat(results, ignore_index=True)

def plot_metric_comparison(df, output_dir):
    """Create bar chart comparing metrics across ablation modes."""
    metrics = ['AD', 'nAD', 'ARD', 'AIE']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Convert to numeric, handling any string formatting
        values = pd.to_numeric(df[metric], errors='coerce')
        
        # Create bar chart
        bars = ax.bar(df['ablation_mode'], values, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Highlight the full model
        full_idx = df[df['ablation_mode'] == 'full'].index
        if len(full_idx) > 0:
            bars[full_idx[0]].set_color('green')
            bars[full_idx[0]].set_alpha(0.8)
        
        # Formatting
        ax.set_title(f'{metric} by Ablation Mode', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xlabel('Ablation Mode', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    
    return output_path

def plot_loss_correspondence(df, output_dir):
    """Create plot showing correspondence between losses and metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define which metrics should be affected by which losses
    correspondences = [
        ('L_cap', 'nAD', ['full', 'wo_cap', 'only_cap']),
        ('L_asp', 'ARD', ['full', 'wo_asp', 'only_asp']),
        ('L_ov', 'AIE', ['full', 'wo_ov', 'only_ov'])
    ]
    
    for i, (loss_name, metric_name, relevant_modes) in enumerate(correspondences):
        ax = axes[i]
        
        # Filter to relevant modes
        df_subset = df[df['ablation_mode'].isin(relevant_modes)]
        
        values = pd.to_numeric(df_subset[metric_name], errors='coerce')
        colors = ['green' if m == 'full' else 'red' if 'wo_' in m else 'orange' 
                  for m in df_subset['ablation_mode']]
        
        bars = ax.bar(df_subset['ablation_mode'], values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_title(f'{loss_name} → {metric_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xlabel('Ablation Mode', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'loss_correspondence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved loss correspondence plot: {output_path}")
    
    return output_path

def generate_summary_table(df, output_dir):
    """Generate a formatted summary table."""
    print("\n" + "="*80)
    print("ABLATION EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print()
    
    # Format numeric columns
    numeric_cols = ['AD', 'nAD', 'ARD', 'AIE']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Print table
    print(df.to_string(index=False))
    print()
    
    # Save to text file
    summary_path = os.path.join(output_dir, 'ablation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION EXPERIMENT RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Add interpretation
        f.write("="*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n\n")
        f.write("Expected behavior:\n")
        f.write("  - Removing L_cap (wo_cap) should increase AD/nAD\n")
        f.write("  - Removing L_asp (wo_asp) should increase ARD\n")
        f.write("  - Removing L_ov (wo_ov) should increase AIE\n\n")
        
        # Check if expectations are met
        full_row = df[df['ablation_mode'] == 'full']
        wo_cap_row = df[df['ablation_mode'] == 'wo_cap']
        wo_asp_row = df[df['ablation_mode'] == 'wo_asp']
        wo_ov_row = df[df['ablation_mode'] == 'wo_ov']
        
        if not full_row.empty:
            f.write("Verification:\n")
            
            if not wo_cap_row.empty:
                ad_increase = wo_cap_row['nAD'].values[0] > full_row['nAD'].values[0]
                f.write(f"  ✓ wo_cap increases nAD: {ad_increase}\n")
            
            if not wo_asp_row.empty:
                ard_increase = wo_asp_row['ARD'].values[0] > full_row['ARD'].values[0]
                f.write(f"  ✓ wo_asp increases ARD: {ard_increase}\n")
            
            if not wo_ov_row.empty:
                aie_increase = wo_ov_row['AIE'].values[0] > full_row['AIE'].values[0]
                f.write(f"  ✓ wo_ov increases AIE: {aie_increase}\n")
    
    print(f"✓ Saved summary table: {summary_path}")
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ablation_results.py <ablation_results_dir>")
        print()
        print("Example:")
        print("  python analyze_ablation_results.py ablation_results/")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)
    
    print(f"Loading ablation results from: {base_dir}")
    print()
    
    # Load results
    df = load_ablation_results(base_dir)
    if df is None:
        sys.exit(1)
    
    print(f"Loaded {len(df)} ablation experiments")
    print()
    
    # Generate visualizations
    plot_metric_comparison(df, base_dir)
    plot_loss_correspondence(df, base_dir)
    
    # Generate summary
    generate_summary_table(df, base_dir)
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("Generated files:")
    print(f"  - {base_dir}/ablation_comparison.png")
    print(f"  - {base_dir}/loss_correspondence.png")
    print(f"  - {base_dir}/ablation_summary.txt")
    print()

if __name__ == '__main__':
    main()
