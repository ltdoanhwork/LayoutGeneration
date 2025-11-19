#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick script to print summary results in a readable format.

Usage:
    python -m eval.print_summary --summary outputs/val_runs/test_no_ckpt/summary_results.json
"""
import json
import argparse
import numpy as np


def print_summary(summary_path: str):
    """Print summary from summary_results.json in a readable format."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print("\n" + "="*80)
    print("BATCH EVALUATION SUMMARY")
    print("="*80)
    print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
    
    config = summary.get('config', {})
    print(f"\nConfiguration:")
    print(f"  Videos Dir: {config.get('videos_dir', 'N/A')}")
    print(f"  Output Dir: {config.get('output_base_dir', 'N/A')}")
    print(f"  Checkpoint: {config.get('checkpoint', 'None (random weights)')}")
    print(f"  Feature Dim: {config.get('feat_dim', 'N/A')}")
    print(f"  Budget Ratio: {config.get('budget_ratio', 'N/A')}")
    
    stats = summary.get('statistics', {})
    print(f"\nStatistics:")
    print(f"  Total Processed: {stats.get('total_processed', 0)}")
    print(f"  Total Failed: {stats.get('total_failed', 0)}")
    
    print("\n" + "-"*80)
    print("AGGREGATE METRICS (Mean across all videos)")
    print("-"*80)
    
    agg = summary.get('aggregate_metrics', {})
    if agg:
        for key, val in sorted(agg.items()):
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                print(f"  {key:40s}: {val:.6f}")
            else:
                print(f"  {key:40s}: N/A")
    else:
        print("  No aggregate metrics found.")
    
    # Print per-video summary
    results = summary.get('results', {})
    if results:
        print("\n" + "-"*80)
        print(f"PER-VIDEO RESULTS ({len(results)} videos)")
        print("-"*80)
        
        for video_id, data in sorted(results.items()):
            metrics = data.get('metrics', {})
            print(f"\n  Video: {video_id}")
            print(f"    RecErr: {metrics.get('RecErr', 'N/A')}")
            print(f"    Frechet: {metrics.get('Frechet', 'N/A')}")
            print(f"    SceneCoverage: {metrics.get('SceneCoverage', 'N/A')}")
            print(f"    TemporalCoverage@tau: {metrics.get('TemporalCoverage@tau', 'N/A')}")
            print(f"    LPIPS_PerceptualGap: {metrics.get('LPIPS_PerceptualGap', 'N/A')}")
            print(f"    LPIPS_DiversitySel: {metrics.get('LPIPS_DiversitySel', 'N/A')}")
            print(f"    MS_SWD_Color: {metrics.get('MS_SWD_Color', 'N/A')}")
    
    # Print errors if any
    errors = summary.get('errors', {})
    if errors:
        print("\n" + "-"*80)
        print(f"ERRORS ({len(errors)} videos)")
        print("-"*80)
        for video_id, error_msg in sorted(errors.items()):
            print(f"  {video_id}: {error_msg}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Print batch evaluation summary")
    parser.add_argument("--summary", required=True, help="Path to summary_results.json")
    
    args = parser.parse_args()
    print_summary(args.summary)


if __name__ == "__main__":
    main()
