#!/usr/bin/env python3
import json
import os
from pathlib import Path

# Base directory
base_dir = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_final_reorg")
baseline_dir = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_recerr_w0.2")

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

# Add Baseline to mapping explicitly or just use logic?
# User wants comparison. 
# `1_rec_opt_ours` IS the baseline config run in ablation.
# `training_v11_recerr_w0.2` is the original baseline run.
# They should be very close.
# I will use the ablation runs as primary.
# `1_rec_opt_ours` corresponds to the "Ours" baseline.

def load_metrics(exp_dir):
    json_path = base_dir / exp_dir / "ablation_eval_consistent.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)

# Metrics config with direction
# Metrics in json: mpr, top10, rec_err, fd, temp_cov, lpips, dists
metrics_conf = [
    ("mpr", "MPR", "max"),
    ("top10", "Top10", "max"),
    ("rec_err", "RecErr", "min"),
    ("fd", "FD", "min"),
    ("temp_cov", "TempCov", "max"),
    ("lpips", "LPIPS", "min"),
    ("dists", "DISTS", "min")
]

print("\\begin{table*}[t]")
print("\\centering")
print("\\scriptsize")
print("\\setlength{\\tabcolsep}{3pt}") # Reduce padding to fit 7 cols
print("\\renewcommand{\\arraystretch}{1.1}")
print("\\begin{tabular}{@{} l c c c c c c c @{}}")
print("\\toprule")
# Header
headers = [m[1] for m in metrics_conf]
arrows = ["$\\uparrow$" if m[2]=="max" else "$\\downarrow$" for m in metrics_conf]
header_str = " & ".join([f"\\textbf{{{h}}} {a}" for h, a in zip(headers, arrows)])
print(f"\\textbf{{Ablation Setting}} & {header_str} \\\\")
print("\\midrule")

for category, exps in experiments.items():
    print(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{category}}}}} \\\\")
    
    # Get all metrics for this cat to find best
    cat_metrics = []
    for d, n in exps:
        m = load_metrics(d)
        cat_metrics.append(m)
        
    best_vals = {}
    for key, _, direction in metrics_conf:
        valid_vals = [m[key] for m in cat_metrics if m and not isinstance(m[key], str) and float(m[key]) != 0.0] 
        # Note: 0.0 check is heuristic, sometimes valid, but usually implies failure or NaN in my script
        if valid_vals:
            if direction == "max": best_vals[key] = max(valid_vals)
            else: best_vals[key] = min(valid_vals)
        else:
            best_vals[key] = None
            
    for i, (exp_dir, exp_name) in enumerate(exps):
        m = cat_metrics[i]
        row = [f"\\hspace{{3mm}} {exp_name}"]
        if m:
            for key, _, _ in metrics_conf:
                val = m.get(key, 0.0)
                if val == 0.0: # Assume fail
                    row.append("--")
                else:
                    fmt = "{:.3f}" if key != "rec_err" else "{:.4f}"
                    val_str = fmt.format(val)
                    if best_vals[key] is not None and abs(val - best_vals[key]) < 1e-6:
                        row.append(f"\\textbf{{{val_str}}}")
                    else:
                        row.append(val_str)
        else:
            row.extend(["--"] * len(metrics_conf))
        print(" & ".join(row) + " \\\\")
    
    if "Scene" not in category:
        print("\\midrule")

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Detailed ablation study with consistent metrics (evaluated on 50-scene subset). Best results in each category are bolded.}")
print("\\label{tab:ablation_results}")
print("\\end{table*}")
