#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import re
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import load_scene_dir, build_epoch_index
from src.models.dsn_v8 import create_dsn_v8
from src.distance_selector.registry import create_metric
from src.rl.distribution_metrics import DistributionAwareMetrics, ATTR_NAMES

# Optimize perceptual gap
def perceptual_gap_batched(metric_name, metric_obj, all_frames, selected_indices, device):
    if not selected_indices or not all_frames: return float("nan")
    
    # Preprocess
    with torch.no_grad():
        try:
            Ts_all = [metric_obj.preprocess_bgr(f) for f in all_frames]
        except Exception as e:
            # print(f"Preprocess error: {e}")
            return float("nan")
            
    if not Ts_all: return float("nan")
    
    t_all = torch.cat(Ts_all, dim=0) # (T, C, H, W)
    t_all = t_all.to(device)
    
    sel_indices_list = [i for i in selected_indices if i < len(Ts_all)]
    if not sel_indices_list: return float("nan")
    
    t_sel = t_all[sel_indices_list] # (K, C, H, W)
    
    T = t_all.shape[0]
    K = t_sel.shape[0]
    
    min_dists = []
    
    if metric_name == "lpips":
        model = metric_obj._lpips
    elif metric_name == "dists":
        model = metric_obj._dists
    else:
        return float("nan")
        
    batch_size = 5 # Small batch size to be safe with VRAM (5 * 15 = 75 pairs)
    
    with torch.no_grad():
        for i in range(0, T, batch_size):
            end = min(i + batch_size, T)
            curr = t_all[i:end] 
            B_curr = curr.shape[0]
            
            # Expand for broadcasting match
            # (B_curr, 1, C, H, W) -> (B_curr, K, C, H, W) -> Flatten -> (B_curr*K, C, H, W)
            curr_exp = curr.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(-1, *curr.shape[1:])
            
            # (1, K, C, H, W) -> (B_curr, K, C, H, W) -> Flatten
            target_exp = t_sel.unsqueeze(0).expand(B_curr, -1, -1, -1, -1).reshape(-1, *t_sel.shape[1:])
            
            d = model(curr_exp, target_exp) 
            d = d.view(B_curr, K)
            
            mins, _ = d.min(dim=1)
            min_dists.extend(mins.cpu().numpy().tolist())
            
    return float(np.mean(min_dists))

def feature_distance_gap(features_all, selected_indices):
    if not selected_indices or len(features_all) == 0: return float("nan")
    feats_all = features_all / (np.linalg.norm(features_all, axis=1, keepdims=True) + 1e-12)
    feats_sel = feats_all[selected_indices]
    gaps = []
    for i in range(len(feats_all)):
        similarities = feats_sel @ feats_all[i]
        min_distance = 1.0 - np.max(similarities)
        gaps.append(min_distance)
    return float(np.mean(gaps))

def calculate_frechet(features_all, selected_indices):
    # Simplified Frechet calculation logic (reuse from original)
    # ... (omitted for brevity, assume simple impl or copy if crucial, here assuming not main bottleneck)
    # Actually need it for FD metric
    import scipy.linalg
    if not selected_indices or len(features_all) < 2: return float("nan")
    mu1 = np.mean(features_all, axis=0)
    sigma1 = np.cov(features_all, rowvar=False)
    mu2 = np.mean(features_all[selected_indices], axis=0)
    sigma2 = np.cov(features_all[selected_indices], rowvar=False)
    
    diff = mu1 - mu2
    try:
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean): covmean = covmean.real
        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
    except:
        return float("nan")

def temporal_coverage(selected_indices):
    if len(selected_indices) < 2: return 0.0
    sorted_idx = sorted(selected_indices)
    gaps = [sorted_idx[i+1] - sorted_idx[i] for i in range(len(sorted_idx)-1)]
    return np.std(gaps) if gaps else 0.0

def eval_experiment(exp_name, checkpoint_path, test_scenes, lpips_metric, dists_metric, device="cuda"):
    print(f"Evaluating {exp_name} on {len(test_scenes)} scenes...")
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})
    
    use_anime = not config.get("no_anime_attrs", False)
    feat_dim = config.get("feat_dim", 512)
    full_feat_dim = feat_dim + (6 if use_anime else 0)
    
    model = create_dsn_v8(
        feat_dim=full_feat_dim,
        num_attn_layers=config.get("num_attn_layers", 2),
        gating_hidden=config.get("gating_hidden", 64),
        lstm_hidden=config.get("lstm_hidden", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Accumulators
    metrics = {k: [] for k in ["mpr", "top10", "rec_err", "fd", "temp_cov", "lpips", "dists"]}
    
    for scene_dir in test_scenes:
        try:
            sample = load_scene_dir(scene_dir, load_frames=True, load_anime_attrs=True)
            if sample.anime_attrs is None or not sample.frames: continue
            
            # Predict
            feats_input = np.concatenate([sample.feats, sample.anime_attrs], axis=1) if use_anime else sample.feats
            feats_t = torch.from_numpy(feats_input).float().unsqueeze(0).to(device)
            T = len(sample.feats)
            budget = max(3, min(15, int(T * 0.15)))
            
            with torch.no_grad():
                probs, _ = model(feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
            # Compute Metrics
            metrics["rec_err"].append(feature_distance_gap(sample.feats, sel_idx))
            metrics["fd"].append(calculate_frechet(sample.feats, sel_idx))
            metrics["temp_cov"].append(temporal_coverage(sel_idx))
            
            # MPR/Top10
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            metrics["mpr"].append(np.mean(percentiles[sel_idx]))
            
            k10 = max(1, int(T * 0.1))
            top10_idx = set(np.argsort(quality)[-k10:])
            metrics["top10"].append(len(set(sel_idx) & top10_idx) / k10)
            
            # Expensive Metrics (Batched)
            metrics["lpips"].append(perceptual_gap_batched("lpips", lpips_metric, sample.frames, sel_idx, device))
            metrics["dists"].append(perceptual_gap_batched("dists", dists_metric, sample.frames, sel_idx, device))
            
        except Exception as e:
            # print(f"Scene error: {e}")
            continue
            
    # Aggregate
    final_res = {}
    for k, v in metrics.items():
        vals = [x for x in v if not np.isnan(x)]
        final_res[k] = float(np.mean(vals)) if vals else 0.0
        
    # Calculate Composite
    # Comp = MPR + Top10 + (1-RecErr) + (1-FD) + TempCov (if normalized? No, usually composite logic is specific)
    # User's logic: Comp = MPR + Top10 + (1-RecErr) + ??
    # Previous ablation table: Comp ~ 1.9. MPR~0.8, Top10~0.6. Sum=1.4.
    # RecErr~0.05. 1-0.05 = 0.95. Sum=2.35.
    # Maybe Comp = MPR + Top10 + (0.1/RecErr)? No.
    # Let's look at `extract_ablation_metrics.py`? No, that just read the value.
    # In `val_results.json`: "composite_score": 1.74
    # I don't know the exact formula, but I can include "Composite" key if I knew it.
    # But for the table "Metric (dir.)", Composite is not explicitly listed in User's Main Table snippet.
    # User's ablation table had it.
    # I will stick to reporting the metrics requested: MPR, Top10, RecErr, FD, TempCov, LPIPS, DISTS.
    # I can re-calculate Composite if I find the formula, or just omit it if the new table format (from user snippet) doesn't have it.
    # User snippet: RecErr, FD, MPR, Top10, TempCov, DistGap, LPIPS Gap.
    # It does NOT have Composite column!
    # So I can drop Composite.
    
    return final_res

def get_best_checkpoint(exp_dir):
    best_files = list(Path(exp_dir).glob("best_epoch_*.pt"))
    if not best_files: return None, None
    
    best_file = None
    max_score = -1.0
    for f in best_files:
        match = re.search(r"score_([\d\.]+)\.pt", f.name)
        if match:
            s_val = float(match.group(1))
            if s_val > max_score:
                max_score = s_val
                best_file = f
    return best_file, max_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/ablation_final_reorg")
    parser.add_argument("--baseline", default="runs/training_v11_recerr_w0.2")
    parser.add_argument("--test_root", default="data/sakuga_dataset_v11_new_test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = args.device
    
    # Init Metrics
    print("Initializing Metrics...")
    lpips_metric = create_metric("lpips", net="alex", device=device)
    dists_metric = create_metric("dists", device=device)
    
    # Load Scenes (Subset 50)
    all_scenes = build_epoch_index(args.test_root)
    test_scenes = all_scenes[:50]
    print(f"Using {len(test_scenes)} scenes for evaluation.")
    
    # Experiments List
    # Get all subdirs in root
    root_path = Path(args.root)
    exp_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    # Add baseline
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        exp_dirs.insert(0, baseline_path)
    
    results_map = {}
    
    for exp_dir in tqdm(exp_dirs, desc="Experiments"):
        ckpt_path, score = get_best_checkpoint(exp_dir)
        if not ckpt_path:
            # Try epXX/checkpoint.pt if training ongoing? Or just skip
            # User wants successful runs.
            continue
            
        out_file = exp_dir / "ablation_eval_consistent.json"
        
        if out_file.exists():
            with open(out_file) as f:
                experiment_results = json.load(f)
        else:
            experiment_results = eval_experiment(exp_dir.name, ckpt_path, test_scenes, lpips_metric, dists_metric, device)
            with open(out_file, "w") as f:
                json.dump(experiment_results, f, indent=2)
                
        results_map[exp_dir.name] = experiment_results
        
    print("Done.")

if __name__ == "__main__":
    main()
