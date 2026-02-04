#!/usr/bin/env python3
"""
Batch evaluation script for ablation study with:
- Mean ± Std output format
- Support for multiple experiment directories
- Fixed DISTS metric with image resizing
"""
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

def perceptual_gap_batched(metric_name, metric_obj, all_frames, selected_indices, device):
    """Compute batched perceptual gap (LPIPS or DISTS)."""
    if not selected_indices or not all_frames: 
        return float("nan")
    
    # Preprocess (Keep on CPU to avoid OOM)
    Ts_all_cpu = []
    with torch.no_grad():
        try:
            for f in all_frames:
                t = metric_obj.preprocess_bgr(f)
                Ts_all_cpu.append(t.cpu())
        except Exception as e:
            print(f"Preprocess error: {e}")
            return float("nan")
            
    if not Ts_all_cpu: 
        return float("nan")
    
    try:
        t_all_cpu = torch.cat(Ts_all_cpu, dim=0)
    except Exception as e:
        print(f"Stack error: {e}")
        return float("nan")

    sel_indices_list = [i for i in selected_indices if i < len(Ts_all_cpu)]
    if not sel_indices_list: 
        return float("nan")
    
    t_sel = t_all_cpu[sel_indices_list].to(device)
    T = t_all_cpu.shape[0]
    K = t_sel.shape[0]
    
    min_dists = []
    
    if metric_name == "lpips":
        model = metric_obj._lpips
    elif metric_name == "dists":
        model = metric_obj._dists
    else:
        return float("nan")
        
    batch_size = 20
    
    with torch.no_grad():
        for i in range(0, T, batch_size):
            try:
                end = min(i + batch_size, T)
                curr = t_all_cpu[i:end].to(device)
                B_curr = curr.shape[0]
                
                curr_exp = curr.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(-1, *curr.shape[1:])
                target_exp = t_sel.unsqueeze(0).expand(B_curr, -1, -1, -1, -1).reshape(-1, *t_sel.shape[1:])
                
                d = model(curr_exp, target_exp) 
                
                if d.dim() > 1:
                    d = d.view(d.size(0))
                
                d = d.view(B_curr, K)
                mins, _ = d.min(dim=1)
                min_dists.extend(mins.cpu().numpy().tolist())
            except Exception as e:
                print(f"Batch calc error: {e}")
                return float("nan")
            
    return float(np.mean(min_dists))

def feature_distance_gap(features_all, selected_indices):
    if not selected_indices or len(features_all) == 0: 
        return float("nan")
    feats_all = features_all / (np.linalg.norm(features_all, axis=1, keepdims=True) + 1e-12)
    feats_sel = feats_all[selected_indices]
    gaps = []
    for i in range(len(feats_all)):
        similarities = feats_sel @ feats_all[i]
        min_distance = 1.0 - np.max(similarities)
        gaps.append(min_distance)
    return float(np.mean(gaps))

def calculate_frechet(features_all, selected_indices):
    import scipy.linalg
    if not selected_indices or len(features_all) < 2: 
        return float("nan")
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
        if np.iscomplexobj(covmean): 
            covmean = covmean.real
        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
    except:
        return float("nan")

def temporal_coverage(selected_indices):
    if len(selected_indices) < 2: 
        return 0.0
    sorted_idx = sorted(selected_indices)
    gaps = [sorted_idx[i+1] - sorted_idx[i] for i in range(len(sorted_idx)-1)]
    return np.std(gaps) if gaps else 0.0

def eval_experiment(exp_name, checkpoint_path, test_scenes, lpips_metric, dists_metric, device="cuda"):
    """Evaluate experiment and return per-scene metrics."""
    print(f"Evaluating {exp_name} on {len(test_scenes)} scenes...")
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    
    # Per-scene metrics
    metrics = {k: [] for k in ["mpr", "top10", "rec_err", "fd", "temp_cov", "lpips", "dists"]}
    
    for scene_dir in tqdm(test_scenes, desc=exp_name, leave=False):
        try:
            sample = load_scene_dir(scene_dir, load_frames=True, load_anime_attrs=True)
            if sample.anime_attrs is None or not sample.frames: 
                continue
            
            feats_input = np.concatenate([sample.feats, sample.anime_attrs], axis=1) if use_anime else sample.feats
            feats_t = torch.from_numpy(feats_input).float().unsqueeze(0).to(device)
            T = len(sample.feats)
            budget = max(3, min(15, int(T * 0.15)))
            
            with torch.no_grad():
                probs, _ = model(feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
            # Compute all metrics
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
            
            # Perceptual metrics
            metrics["lpips"].append(perceptual_gap_batched("lpips", lpips_metric, sample.frames, sel_idx, device))
            metrics["dists"].append(perceptual_gap_batched("dists", dists_metric, sample.frames, sel_idx, device))
            
        except Exception as e:
            continue
    
    # Compute mean ± std for each metric
    final_res = {}
    for k, v in metrics.items():
        vals = [x for x in v if not np.isnan(x)]
        if vals:
            final_res[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }
        else:
            final_res[k] = {"mean": 0.0, "std": 0.0}
    
    final_res["n_scenes"] = len([x for x in metrics["mpr"] if not np.isnan(x)])
    return final_res

def get_best_checkpoint(exp_dir):
    """Find best checkpoint in experiment directory."""
    best_files = list(Path(exp_dir).glob("best_epoch_*.pt"))
    if not best_files: 
        return None, None
    
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
    parser.add_argument("--roots", nargs="+", default=[
        "runs/ablation_final_reorg",
        "runs/ablation_deep_iqa"
    ])
    parser.add_argument("--baseline", default="runs/training_v11_recerr_w0.2")
    parser.add_argument("--test_root", default="data/sakuga_dataset_v11_new_test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if results exist")
    args = parser.parse_args()
    
    device = args.device
    torch.cuda.empty_cache()
    
    print("Initializing Metrics...")
    lpips_metric = create_metric("lpips", net="alex", device=device)
    dists_metric = create_metric("dists", device=device)
    
    all_scenes = build_epoch_index(args.test_root)
    test_scenes = all_scenes[:50]
    print(f"Using {len(test_scenes)} scenes for evaluation.")
    
    # Collect all experiment directories
    exp_dirs = []
    
    # Add baseline first
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        exp_dirs.append(baseline_path)
    
    # Add experiments from all roots
    for root in args.roots:
        root_path = Path(root)
        if root_path.exists():
            for d in sorted(root_path.iterdir()):
                if d.is_dir():
                    exp_dirs.append(d)
    
    print(f"Found {len(exp_dirs)} experiments to evaluate.")
    
    results_map = {}
    
    for exp_dir in tqdm(exp_dirs, desc="Experiments"):
        ckpt_path, score = get_best_checkpoint(exp_dir)
        if not ckpt_path:
            continue
            
        out_file = exp_dir / "ablation_eval_v2.json"
        
        if out_file.exists() and not args.force:
            with open(out_file) as f:
                experiment_results = json.load(f)
        else:
            experiment_results = eval_experiment(
                exp_dir.name, ckpt_path, test_scenes, 
                lpips_metric, dists_metric, device
            )
            with open(out_file, "w") as f:
                json.dump(experiment_results, f, indent=2)
                
        results_map[exp_dir.name] = experiment_results
        
    print("Done.")
    
    # Print summary
    print("\n=== Summary ===")
    for name, res in results_map.items():
        mpr = res.get("mpr", {})
        lpips = res.get("lpips", {})
        dists = res.get("dists", {})
        print(f"{name}: MPR={mpr.get('mean', 0):.3f}±{mpr.get('std', 0):.3f}, "
              f"LPIPS={lpips.get('mean', 0):.3f}±{lpips.get('std', 0):.3f}, "
              f"DISTS={dists.get('mean', 0):.3f}±{dists.get('std', 0):.3f}")

if __name__ == "__main__":
    main()
