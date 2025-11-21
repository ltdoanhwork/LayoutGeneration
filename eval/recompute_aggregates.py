#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute aggregate metrics from existing summary_results.json

Usage:
    python -m eval.recompute_aggregates --summary /home/serverai/ltdoanh/LayoutGeneration/outputs/val_runs/advanced_v1/ep1/summary_results.json
"""
import json
import argparse
import numpy as np


def safe_mean(xs):
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(vals)) if vals else None


def recompute_aggregates(summary_path: str):
    """Recompute aggregate metrics from summary_results.json"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    results = summary.get('results', {})
    if not results:
        print("No results found in summary.")
        return
    
    # Collect metrics from "method" (DSN)
    rec = []; fre = []; scov = []; tcov = []; lp_gap = []; lp_div = []; ms = []
    
    for vid, r in results.items():
        # Metrics are nested: r["metrics"]["method"] contains DSN results
        m = r["metrics"].get("method", r["metrics"])
        rec.append(m.get("RecErr"))
        fre.append(m.get("Frechet"))
        scov.append(m.get("SceneCoverage"))
        tcov.append(m.get("TemporalCoverage@tau"))
        lp_gap.append(m.get("LPIPS_PerceptualGap"))
        lp_div.append(m.get("LPIPS_DiversitySel"))
        ms.append(m.get("MS_SWD_Color"))
    
    # Compute aggregates
    summary["aggregate_metrics"] = {
        "RecErr_mean": safe_mean(rec),
        "Frechet_mean": safe_mean(fre),
        "SceneCoverage_mean": safe_mean(scov),
        "TemporalCoverage@tau_mean": safe_mean(tcov),
        "LPIPS_PerceptualGap_mean": safe_mean(lp_gap),
        "LPIPS_DiversitySel_mean": safe_mean(lp_div),
        "MS_SWD_Color_mean": safe_mean(ms),
    }
    
    # Save back
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Recomputed aggregate metrics and saved to: {summary_path}")
    print("\nAggregate Metrics:")
    for key, val in summary["aggregate_metrics"].items():
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            print(f"  {key:40s}: {val:.6f}")
        else:
            print(f"  {key:40s}: N/A")


def main():
    parser = argparse.ArgumentParser(description="Recompute aggregate metrics from summary")
    parser.add_argument("--summary", required=True, help="Path to summary_results.json")
    
    args = parser.parse_args()
    recompute_aggregates(args.summary)


if __name__ == "__main__":
    main()
