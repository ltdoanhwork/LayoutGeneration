#!/usr/bin/env python3
"""
Generate ablation table from evaluation results with mean±std format.
"""
import json
from pathlib import Path

# Base directories
ABLATION_REORG = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_final_reorg")
ABLATION_DEEP = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_deep_iqa")
BASELINE = Path("/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_recerr_w0.2")

# Experiment mapping
EXPERIMENTS = {
    "1. Reward Design": [
        ("1_baseline_div_only", "Diversity Only", ABLATION_REORG),
        ("1_no_diversity", "No Diversity", ABLATION_REORG),
        ("1_strong_diversity", "Strong Diversity", ABLATION_REORG),
        ("training_v11_recerr_w0.2", "Ours (Baseline)", BASELINE.parent),
        ("1_frechet_opt", "Fréchet Opt", ABLATION_REORG),
        ("1_combined_rep", "Combined Rep", ABLATION_REORG),
    ],
    "2. Input Signals": [
        ("2_visual_only", "Visual Features Only", ABLATION_REORG),
        ("2_full_features", "Full (Visual + Aes)", ABLATION_REORG),
    ],
    "3. Architecture": [
        ("3_no_attn", "No Attention", ABLATION_REORG),
        ("3_attn_1L", "Attn 1-Layer", ABLATION_REORG),
        ("3_attn_2L", "Attn 2-Layer (Base)", ABLATION_REORG),
        ("3_attn_4L", "Attn 4-Layer", ABLATION_REORG),
        ("3_gate_small", "Gate Small", ABLATION_REORG),
        ("3_gate_large", "Gate Large", ABLATION_REORG),
    ],
    "4. RL Budget": [
        ("4_budget_10", "Budget 10%", ABLATION_REORG),
        ("4_budget_15", "Budget 15%", ABLATION_REORG),
        ("4_budget_25", "Budget 25%", ABLATION_REORG),
    ],
    "5. Entropy Reg.": [
        ("5_low_entropy", "Low Entropy", ABLATION_REORG),
        ("5_high_entropy", "High Entropy", ABLATION_REORG),
    ],
    "6. Scene Detection": [
        ("6_pyscene_diverse", "PyScene Diverse", ABLATION_REORG),
        ("6_pyscene_fixed_short", "PyScene Fixed Short", ABLATION_REORG),
        ("6_tnv2_diverse", "TNv2 Diverse", ABLATION_REORG),
        ("6_tnv2_fixed_long", "TNv2 Fixed Long", ABLATION_REORG),
        ("6_tnv2_fixed_short", "TNv2 Fixed Short", ABLATION_REORG),
    ],
    "7. Prompt Count": [
        ("G1_prompt_1pair", "1-Pair", ABLATION_DEEP),
        ("G1_prompt_2pair", "2-Pair", ABLATION_DEEP),
        ("G1_prompt_3pair", "3-Pair (Baseline)", ABLATION_DEEP),
        ("G1_prompt_4pair", "4-Pair", ABLATION_DEEP),
        ("G1_prompt_5pair", "5-Pair (Full)", ABLATION_DEEP),
    ],
}

# Metrics config: (json_key, header, direction, format)
METRICS = [
    ("mpr", "MPR", "max", "{:.3f}"),
    ("top10", "Top10", "max", "{:.3f}"),
    ("rec_err", "RecErr", "min", "{:.4f}"),
    ("fd", "FD", "min", "{:.3f}"),
    ("temp_cov", "TempCov", "max", "{:.2f}"),
    ("lpips", "LPIPS", "min", "{:.3f}"),
    ("dists", "DISTS", "min", "{:.3f}"),
]

def load_metrics(exp_dir, exp_name):
    """Load metrics from ablation_eval_v2.json"""
    if exp_name == "training_v11_recerr_w0.2":
        json_path = BASELINE / "ablation_eval_v2.json"
    else:
        json_path = exp_dir / exp_name / "ablation_eval_v2.json"
    
    if not json_path.exists():
        return None
    
    with open(json_path) as f:
        return json.load(f)

def format_metric(val_dict, fmt, is_best=False):
    """Format metric with mean±std"""
    if val_dict is None or val_dict.get("mean", 0) == 0:
        return "--"
    
    mean = val_dict["mean"]
    std = val_dict.get("std", 0)
    
    # Format as mean±std
    mean_str = fmt.format(mean)
    std_str = fmt.format(std)
    
    # Compact format: just mean for table clarity
    # Full format with std can be in appendix
    if is_best:
        return f"\\textbf{{{mean_str}}}"
    return mean_str

def main():
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\scriptsize")
    print("\\setlength{\\tabcolsep}{3pt}")
    print("\\renewcommand{\\arraystretch}{1.1}")
    print("\\begin{tabular}{@{} l c c c c c c c @{}}")
    print("\\toprule")
    
    # Header
    arrow_up = "$\\uparrow$"
    arrow_down = "$\\downarrow$"
    headers = [f"\\textbf{{{m[1]}}} {arrow_up if m[2]=='max' else arrow_down}" for m in METRICS]
    print(f"\\textbf{{Ablation Setting}} & {' & '.join(headers)} \\\\")
    print("\\midrule")
    
    for category, exps in EXPERIMENTS.items():
        print(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{category}}}}} \\\\")
        
        # Load all metrics for category to find best
        cat_data = []
        for exp_name, display_name, base_dir in exps:
            m = load_metrics(base_dir, exp_name)
            cat_data.append((exp_name, display_name, m))
        
        # Find best for each metric
        best_vals = {}
        for key, _, direction, _ in METRICS:
            valid_vals = []
            for _, _, m in cat_data:
                if m and key in m and m[key].get("mean", 0) != 0:
                    valid_vals.append(m[key]["mean"])
            
            if valid_vals:
                best_vals[key] = max(valid_vals) if direction == "max" else min(valid_vals)
            else:
                best_vals[key] = None
        
        # Print rows
        for exp_name, display_name, m in cat_data:
            row = [f"\\hspace{{3mm}} {display_name}"]
            
            if m:
                for key, _, _, fmt in METRICS:
                    val_dict = m.get(key)
                    is_best = (val_dict and best_vals.get(key) and 
                               abs(val_dict.get("mean", 0) - best_vals[key]) < 1e-6)
                    row.append(format_metric(val_dict, fmt, is_best))
            else:
                row.extend(["--"] * len(METRICS))
            
            print(" & ".join(row) + " \\\\")
        
        # Separator between groups (except last)
        if category != "7. Prompt Count":
            print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Ablation study with mean metrics (evaluated on 50-scene subset). Bold indicates best in category.}")
    print("\\label{tab:ablation_results}")
    print("\\end{table*}")

if __name__ == "__main__":
    main()
