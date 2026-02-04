#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

# Base directory
base_dir = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_final_reorg")

# Experiment mapping
experiments = {
    "1. Reward Design": [
        ("1_baseline_div_only", "Baseline (Div Only)"),
        ("1_no_diversity", "No Diversity"),
        ("1_strong_diversity", "Strong Diversity"),
        ("1_rec_opt_ours", "RecErr Opt"),
        ("1_frechet_opt", "Fréchet Opt"),
        ("1_combined_rep", "Combined Rep"),
    ],
    "2. Input Signals": [
        ("2_visual_only", "Visual Features Only"),
        ("2_full_features", "Full (Visual + Aes)"),
    ],
    "3. Architecture": [
        ("3_no_attn", "No Attention"),
        ("3_attn_1L", "Attn 1-Layer"),
        ("3_attn_2L", "Attn 2-Layer (Base)"),
        ("3_attn_4L", "Attn 4-Layer"),
        ("3_gate_small", "Gate Small"),
        ("3_gate_large", "Gate Large"),
    ],
    "4. RL Budget": [
        ("4_budget_10", "Budget 10%"),
        ("4_budget_15", "Budget 15%"),
        ("4_budget_25", "Budget 25%"),
    ],
    "5. Entropy Reg.": [
        ("5_low_entropy", "Low Entropy"),
        ("5_high_entropy", "High Entropy"),
    ],
    "6. Scene Detection": [
        ("6_pyscene_diverse", "PyScene Diverse"),
        ("6_pyscene_fixed_short", "PyScene Fixed Short"),
        ("6_tnv2_diverse", "TNv2 Diverse"),
        ("6_tnv2_fixed_long", "TNv2 Fixed Long"),
        ("6_tnv2_fixed_short", "TNv2 Fixed Short"),
    ],
}

def get_best_epoch_metrics(exp_dir):
    dir_path = base_dir / exp_dir
    if not dir_path.exists():
        return None
    
    # Find all best_epoch files
    best_files = list(dir_path.glob("best_epoch_*_score_*.pt"))
    if not best_files:
        # Try looking for just final_metrics.json if no checkpoints
        final_metrics = dir_path / "final_metrics.json"
        if final_metrics.exists():
             with open(final_metrics) as f:
                data = json.load(f)
                return data # This might lack some keys, but better than nothing
        return None
        
    # Parse scores and epochs
    best_file = None
    max_score = -1.0
    best_epoch = -1
    
    for f in best_files:
        match = re.search(r"best_epoch_(\d+)_score_([\d\.]+)\.pt", f.name)
        if match:
            epoch = int(match.group(1))
            score = float(match.group(2))
            if score > max_score:
                max_score = score
                best_epoch = epoch
                best_file = f
    
    if best_epoch != -1:
        val_results = dir_path / f"ep{best_epoch}" / "val_results.json"
        if val_results.exists():
            with open(val_results) as f:
                return json.load(f)
    
    return None

# Metrics configuration: key, display format, direction (max or min)
metrics_config = [
    ("mpr", "{:.3f}", "max"),
    ("top10", "{:.3f}", "max"),
    ("RecErr", "{:.4f}", "min"),
    ("Frechet", "{:.3f}", "min"),
    ("TempCov", "{:.3f}", "max"),
    ("composite_score", "{:.3f}", "max")
]

# Print LaTeX table header
print("\\begin{table*}[t]")
print("\\centering")
print("\\scriptsize")
print("\\setlength{\\tabcolsep}{4pt}")
print("\\renewcommand{\\arraystretch}{1.1}")
print("\\begin{tabular}{@{} l c c c c c c @{}}")
print("\\toprule")
print("\\textbf{Ablation Setting} & \\textbf{MPR} $\\uparrow$ & \\textbf{Top10} $\\uparrow$ & \\textbf{RecErr} $\\downarrow$ & \\textbf{FD} $\\downarrow$ & \\textbf{TempCov} $\\uparrow$ & \\textbf{Comp.} $\\uparrow$ \\\\")
print("\\midrule")

for category, exps in experiments.items():
    print(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{category}}}}} \\\\")
    
    # Collect data for the category
    category_data = []
    for exp_dir, exp_name in exps:
        metrics = get_best_epoch_metrics(exp_dir)
        category_data.append((exp_name, metrics))
    
    # Find best values for each metric in this category
    best_values = {}
    for key, _, direction in metrics_config:
        values = [d[key] for _, d in category_data if d and key in d]
        if values:
            if direction == "max":
                best_values[key] = max(values)
            else:
                best_values[key] = min(values)
        else:
            best_values[key] = None

    # Print rows
    for exp_name, metrics in category_data:
        row_str = f"\\hspace{{3mm}} {exp_name}"
        
        if metrics:
            for key, fmt, _ in metrics_config:
                val = metrics.get(key)
                if val is not None:
                    val_str = fmt.format(val)
                    # Check if this is the best value (within small tolerance)
                    is_best = False
                    best_val = best_values.get(key)
                    if best_val is not None:
                        if abs(val - best_val) < 1e-6:
                            is_best = True
                    
                    if is_best:
                        row_str += f" & \\textbf{{{val_str}}}"
                    else:
                        row_str += f" & {val_str}"
                else:
                    row_str += " & --"
        else:
            row_str += " & -- & -- & -- & -- & -- & --"
            
        print(row_str + " \\\\")
    
    if category != "6. Scene Detection":
        print("\\midrule")

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Detailed ablation study on reward shaping, feature configuration, architecture depth, RL stability settings, and scene detection methods.}")
print("\\label{tab:ablation_results}")
print("\\end{table*}")
