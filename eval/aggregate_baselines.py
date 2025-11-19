#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_baselines.py

Aggregate baseline metrics across all videos in a batch evaluation run.
Reads eval_*.csv files from each video's eval_results folder and computes mean metrics.

Usage:
    python -m eval.aggregate_baselines --eval_dir outputs/val_runs/test_no_ckpt/eval_results
"""
import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def load_csv_metrics(csv_path: str) -> Dict[str, Any]:
    """Load metrics from a CSV file (eval_*.csv format)."""
    if not os.path.exists(csv_path):
        return {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return {}
    
    # Assuming single row with metrics
    metrics = {}
    for key, val in rows[0].items():
        try:
            metrics[key] = float(val)
        except (ValueError, TypeError):
            metrics[key] = val
    
    return metrics


def aggregate_method_metrics(eval_results_dir: str, method_name: str) -> Dict[str, float]:
    """
    Aggregate metrics for a specific method across all videos.
    
    Args:
        eval_results_dir: Path to eval_results directory containing video folders
        method_name: Name of the method (e.g., 'method', 'uniform', 'middle_of_scene', 'motion_peak')
    
    Returns:
        Dictionary of aggregated metrics
    """
    csv_filename = f"eval_{method_name}.csv"
    all_metrics = []
    
    # Scan all video folders
    for video_folder in Path(eval_results_dir).iterdir():
        if not video_folder.is_dir():
            continue
        
        csv_path = video_folder / csv_filename
        if csv_path.exists():
            metrics = load_csv_metrics(str(csv_path))
            if metrics:
                all_metrics.append(metrics)
    
    if not all_metrics:
        return {}
    
    # Aggregate
    aggregated = {}
    metric_keys = all_metrics[0].keys()
    
    for key in metric_keys:
        values = []
        for m in all_metrics:
            val = m.get(key)
            if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                values.append(val)
        
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))
    
    return aggregated


def print_summary(summary_path: str):
    """Print summary from summary_results.json in a readable format."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print("\n" + "="*80)
    print("BATCH EVALUATION SUMMARY")
    print("="*80)
    print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
    print(f"Total Processed: {summary['statistics']['total_processed']}")
    print(f"Total Failed: {summary['statistics']['total_failed']}")
    
    print("\n" + "-"*80)
    print("AGGREGATE METRICS (DSN Method)")
    print("-"*80)
    
    agg = summary.get('aggregate_metrics', {})
    for key, val in agg.items():
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            print(f"  {key:35s}: {val:.4f}")
        else:
            print(f"  {key:35s}: N/A")


def main():
    parser = argparse.ArgumentParser(description="Aggregate baseline metrics from batch evaluation")
    parser.add_argument("--eval_dir", required=True, help="Path to eval_results directory")
    parser.add_argument("--summary_json", default=None, help="Path to summary_results.json (optional)")
    parser.add_argument("--out_json", default=None, help="Output JSON file for aggregated baselines")
    parser.add_argument("--methods", nargs='+', 
                        default=['uniform', 'middle_of_scene', 'motion_peak'],
                        help="Baseline methods to aggregate")
    
    args = parser.parse_args()
    
    # Print main summary if provided
    if args.summary_json and os.path.exists(args.summary_json):
        print_summary(args.summary_json)
    
    # Aggregate baselines
    print("\n" + "="*80)
    print("BASELINE METHODS AGGREGATION")
    print("="*80)
    
    baseline_aggregates = {}
    
    for method in args.methods:
        print(f"\n{'-'*80}")
        print(f"Method: {method.upper()}")
        print(f"{'-'*80}")
        
        agg = aggregate_method_metrics(args.eval_dir, method)
        baseline_aggregates[method] = agg
        
        if agg:
            for key, val in sorted(agg.items()):
                print(f"  {key:35s}: {val:.4f}")
        else:
            print(f"  No data found for method '{method}'")
    
    # Save to JSON if requested
    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(baseline_aggregates, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Baseline aggregates saved to: {args.out_json}")


if __name__ == "__main__":
    main()
