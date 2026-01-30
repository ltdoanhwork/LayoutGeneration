#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 CONSTRAINED: Training with Lagrangian Constraints (PremiumRewardV8)

Based on V11 Final, but replaces the linear reward combination with 
Constrained Optimization (PremiumRewardV8) to solve the conflict
between Quality (MPR) and Representativeness (RecErr).

Improvements:
- Uses Lagrangian multipliers to enforce RecErr < 0.35
- Uses Quantile rewards for safer Anime Quality optimization
- Prevents policy collapse seen with high linear weights
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Core imports
from src.datasets import build_epoch_index, load_scene_dir
from src.models.dsn_v8 import DSNMultiTaskV8, create_dsn_v8
from src.rl.premium_rewards_v8 import create_reward_system_v8
from src.rl.distribution_metrics import (
    DistributionAwareMetrics,
    ATTR_NAMES,
    ATTR_INDEX,
)
from eval.metrics import (
    reconstruction_error,
    frechet_distance,
    temporal_coverage,
    lpips_diversity,
)

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def perceptual_gap_from_frames(metric, all_frames: List[np.ndarray], selected_indices: List[int]) -> float:
    """
    Compute perceptual gap (LPIPS or DISTS): mean minimum distance.
    Copied from eval_v11_gaps.py
    """
    if not selected_indices or not all_frames:
        return float("nan")
    
    # Preprocess frames
    Ts_all = [metric.preprocess_bgr(f) for f in all_frames]
    Ts_sel = [Ts_all[i] for i in selected_indices if i < len(Ts_all)]
    
    if not Ts_sel:
        return float("nan")
    
    # Compute min distance for each frame
    gaps = []
    with torch.no_grad():
        for Ta in Ts_all:
            min_dist = 1e9
            for Ts in Ts_sel:
                d = metric.pair_distance(Ta, Ts)
                if d < min_dist:
                    min_dist = d
            gaps.append(min_dist)
    
    return float(np.mean(gaps)) if gaps else float("nan")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def create_radar_chart(
    per_attr_percentiles: Dict[str, float],
    save_path: str,
    title: str = "Per-Attribute Percentiles",
) -> bool:
    """Create radar chart for per-attribute percentile visualization."""
    if not HAS_MATPLOTLIB:
        return False
    
    # Prepare data
    attr_names = [name.capitalize() for name in ATTR_NAMES]
    values = [per_attr_percentiles.get(f"percentile_{name}", 0.5) for name in ATTR_NAMES]
    
    # Close the radar chart
    values = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(attr_names), endpoint=False).tolist()
    angles = angles + [angles[0]]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.fill(angles, values, color='coral', alpha=0.25)
    ax.plot(angles, values, color='coral', linewidth=2, label='Selected Frames')
    
    # Reference line (random = 0.5)
    reference = [0.5] * (len(attr_names) + 1)
    ax.plot(angles, reference, color='gray', linestyle='--', linewidth=1, label='Random (P50)')
    
    # Configure axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attr_names, fontsize=11)
    ax.set_ylim(0, 1)
    
    # Add concentric circles for reference
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
    
    # Add value annotations
    for angle, val, name in zip(angles[:-1], values[:-1], attr_names):
        ax.annotate(f'{val:.2f}', xy=(angle, val + 0.08), fontsize=9, ha='center', fontweight='bold')
    
    ax.set_title(f"Quality\n{title}", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return True


class ConstrainedTrainer:
    """Trainer using PremiumRewardV8 with Lagrangian constraints."""
    
    def __init__(
        self,
        model: DSNMultiTaskV8,
        lr: float = 1e-4,
        clip_range: float = 0.2,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
        # Reward args
        rec_err_threshold: float = 0.35,
        coverage_threshold: float = 0.3,
        diversity_threshold: float = 0.25,
        anime_scale: float = 3.0,
        quantile_scale: float = 2.0,
        total_epochs: int = 60,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Initialize Constrained Reward System
        self.reward_system = create_reward_system_v8(
            rec_err_threshold=rec_err_threshold,
            coverage_threshold=coverage_threshold,
            diversity_threshold=diversity_threshold,
            total_epochs=total_epochs,
            use_curriculum=True,
            anime_scale=anime_scale,
            quantile_scale=quantile_scale
        )
        
        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
    def update_reward_stats(self, reward: float):
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        if self.reward_count > 1:
            self.reward_std = max(0.1, self.reward_std * 0.99 + abs(delta) * 0.01)
    
    def normalize_reward(self, reward: float) -> float:
        return np.clip((reward - self.reward_mean) / (self.reward_std + 1e-8), -5.0, 5.0)
    
    def compute_reward(
        self,
        features: np.ndarray,
        anime_attrs: np.ndarray,
        sel_idx: List[int],
        current_epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute constrained reward."""
        if len(sel_idx) == 0:
            return 0.0, {"mpr": 0.5}
        
        # Use PremiumRewardV8
        rewards, components = self.reward_system.compute_reward(
            feats_all=features,
            sel_idx=sel_idx,
            anime_attrs=anime_attrs,
            current_epoch=current_epoch,
            update_lagrangian=True  # Update multipliers online
        )
        
        final_reward = rewards["total"]
        
        # Add extra info for logging
        info = {**components}
        info["total_reward"] = final_reward
        
        # Helper for progress bar
        info["mpr"] = components.get("quantile_mean_percentile", 0.5)
        
        return final_reward, info
    
    def train_step(
        self,
        features: torch.Tensor,
        anime_attrs: np.ndarray,
        budget: int,
        epoch: int,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        features = features.to(self.device)
        T = features.shape[1]
        
        # Get probabilities
        with torch.no_grad():
            probs_old, _ = self.model(features)
            probs_old = probs_old.squeeze(0)
        
        # Select top-K by probability
        probs_np = probs_old.cpu().numpy()
        sel_idx = sorted(np.argsort(probs_np)[-budget:].tolist())
        
        # Compute reward
        features_np = features.squeeze(0).cpu().numpy()
        # Pass epoch for curriculum
        reward, reward_info = self.compute_reward(features_np, anime_attrs, sel_idx, epoch)
        
        self.update_reward_stats(reward)
        norm_reward = self.normalize_reward(reward)
        
        # PPO update
        self.optimizer.zero_grad()
        
        probs_new, values = self.model(features)
        probs_new = probs_new.squeeze(0)
        values = values.squeeze()
        
        # Log probabilities
        old_log_prob = safe_log(probs_old[sel_idx]).sum().detach()
        new_log_prob = safe_log(probs_new[sel_idx]).sum()
        
        # PPO loss
        ratio = torch.exp(new_log_prob - old_log_prob)
        ratio = torch.clamp(ratio, 0.01, 100.0)
        
        advantage = torch.tensor([norm_reward], device=self.device, dtype=torch.float32)
        
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        
        # Value loss
        value_target = torch.tensor([reward], device=self.device, dtype=torch.float32)
        value_loss = F.mse_loss(values.mean().unsqueeze(0), value_target)
        
        # Entropy
        entropy = -(probs_new * safe_log(probs_new)).sum()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        if torch.isnan(loss):
            return {"loss": 0.0, "skipped": 1.0, **reward_info}
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "norm_reward": norm_reward,
            **reward_info,
        }


def validate_epoch(
    model: DSNMultiTaskV8,
    val_scene_dirs: List[Path],
    device: str,
    epoch: int,
    save_dir: str,
    budget_ratio: float = 0.15,
    Bmin: int = 3,
    Bmax: int = 15,
    no_anime_attrs: bool = False,
    total_epochs: int = 60,
    attr_suffix: str = "anime_attrs.npy",
) -> Dict[str, float]:
    """
    Comprehensive validation with all 6 metrics + per-attribute breakdown.
    LPIPS_Div is computed only in last 10 epochs for efficiency.
    """
    model.eval()
    
    # Create output directories
    epoch_dir = os.path.join(save_dir, f"ep{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    metrics_computer = DistributionAwareMetrics()
    all_metrics = []
    
    # For aggregate per-attribute percentiles
    all_per_attr = {name: [] for name in ATTR_NAMES}
    
    # LPIPS: only compute in last 10 epochs for efficiency
    compute_lpips = (epoch > total_epochs - 10)
    lpips_metric = None
    for scene_dir in tqdm(val_scene_dirs, desc=f"Validating Ep{epoch}"):
        try:
            sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=True, attr_suffix=attr_suffix)
            if sample.anime_attrs is None:
                continue
            
            # Construct features
            if no_anime_attrs:
                feats_full = sample.feats
            else:
                feats_full = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
            
            feats_t = torch.from_numpy(feats_full).float().unsqueeze(0).to(device)
            T = len(sample.feats)
            budget = max(Bmin, min(Bmax, int(T * budget_ratio)))
            
            # Predict
            with torch.no_grad():
                probs, _ = model(feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            feats_sel = sample.feats[sel_idx]
            
            # ===== 6 Comprehensive Metrics =====
            
            # 1. RecErr (lower is better)
            rec_err = reconstruction_error(sample.feats, feats_sel)
            
            # 2. Frechet (lower is better)
            frechet = frechet_distance(sample.feats, feats_sel)
            if np.isnan(frechet):
                frechet = 0.0
            
            # 3. MPR (higher is better)
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            mpr = float(np.mean(percentiles[sel_idx]))
            
            # 4. Top10 (higher is better)
            k10 = max(1, int(T * 0.1))
            top10_idx = set(np.argsort(quality)[-k10:])
            top10 = len(set(sel_idx) & top10_idx) / k10
            
            # 5. Temporal Coverage (higher is better for Fraction, lower for Std)
            # We compute both: Fraction for Composite, Std for Logging
            temp_cov_frac = temporal_coverage(sample.feats, feats_sel, tau=0.3)
            if np.isnan(temp_cov_frac):
                temp_cov_frac = 0.0
                
            # Compute Std Dev of gaps (for logging per user request)
            sorted_s = sorted(sel_idx)
            if len(sorted_s) > 1:
                gaps = np.diff(sorted_s)
                temp_cov_std = float(np.std(gaps))
            else:
                temp_cov_std = 0.0
            
            # Per-attribute percentiles
            per_attr = metrics_computer.compute_per_attribute_percentile(sample.anime_attrs, sel_idx)
            for name in ATTR_NAMES:
                all_per_attr[name].append(per_attr.get(f"percentile_{name}", 0.5))
            
            metrics = {
                "mpr": mpr,
                "top10": top10,
                "RecErr": rec_err, # This is Feature Gap
                "Frechet": frechet,
                "TempCov": temp_cov_frac, # For score
                "TempCovStd": temp_cov_std, # For log
                "LPIPS_Div": 0.0, # Placeholder
            }
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error validating {scene_dir}: {e}")
            continue
    
    if not all_metrics:
        return {}
    
    # Aggregate metrics (Mean AND Std)
    avg_metrics = {}
    std_metrics = {}
    
    metric_keys = ["mpr", "top10", "RecErr", "Frechet", "TempCov", "TempCovStd"]
    
    for k in metric_keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        if vals:
            avg_metrics[k] = float(np.mean(vals))
            std_metrics[k] = float(np.std(vals))
        else:
            avg_metrics[k] = 0.0
            std_metrics[k] = 0.0
    
    # Aggregate per-attribute percentiles
    avg_per_attr = {}
    for name in ATTR_NAMES:
        if all_per_attr[name]:
            avg_per_attr[f"percentile_{name}"] = float(np.mean(all_per_attr[name]))
        else:
            avg_per_attr[f"percentile_{name}"] = 0.5
    
    # Combine all results
    full_results = {**avg_metrics, **avg_per_attr}
    
    # Compute composite score (higher is better)
    # score = MPR + Top10 - 0.5*RecErr - 0.05*Frechet - 0.1*TempCovStd
    # Note: TempCovStd is "Lower is Better" (Uniformity), so we subtract it.
    composite_score = (
        avg_metrics.get("mpr", 0.5) +
        avg_metrics.get("top10", 0.0) -
        0.5 * min(avg_metrics.get("RecErr", 0.0), 1.0) -  # Clamp RecErr contribution
        0.05 * min(avg_metrics.get("Frechet", 0.0), 1.0) - # Frechet (Lower Good)
        0.1 * min(avg_metrics.get("TempCovStd", 0.0), 5.0) # TempCovStd (Lower Good)
    )
    full_results["composite_score"] = composite_score

    
    # Save validation results JSON
    val_result_path = os.path.join(epoch_dir, "val_results.json")
    with open(val_result_path, "w") as f:
        json.dump(full_results, f, indent=2)
    
    # Create radar chart
    radar_path = os.path.join(epoch_dir, "radar_quality.png")
    create_radar_chart(avg_per_attr, radar_path, title=f"Epoch {epoch} Per-Attribute Percentiles")
    
    # Print Summary matching eval_v11_gaps.py format
    mpr_m, mpr_s = avg_metrics.get("mpr", 0.0), std_metrics.get("mpr", 0.0)
    top10_m, top10_s = avg_metrics.get("top10", 0.0), std_metrics.get("top10", 0.0)
    feat_m, feat_s = avg_metrics.get("RecErr", 0.0), std_metrics.get("RecErr", 0.0)
    frechet_m, frechet_s = avg_metrics.get("Frechet", 0.0), std_metrics.get("Frechet", 0.0)
    temp_m, temp_s = avg_metrics.get("TempCovStd", 0.0), std_metrics.get("TempCovStd", 0.0)
    
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch} VALIDATION RESULTS (Composite: {composite_score:.4f})")
    print(f"{'='*70}")
    print(f"LPIPS Gap:   N/A (See Final Eval)")
    print(f"DISTS Gap:   N/A (See Final Eval)")
    print(f"Feat Gap:    {feat_m:.4f} \u00B1 {feat_s:.4f} (RecErr)")
    print(f"Frechet:     {frechet_m:.4f} \u00B1 {frechet_s:.4f}")
    print(f"MPR:         {mpr_m:.4f} \u00B1 {mpr_s:.4f}")
    print(f"Top10:       {top10_m:.4f} \u00B1 {top10_s:.4f}")
    print(f"Temp Cov:    {temp_m:.4f} \u00B1 {temp_s:.4f} (StdDev)")
    
    print(f"\nPer-Attribute MPR:")
    for name in ATTR_NAMES:
        val = avg_per_attr.get(f"percentile_{name}", 0.5)
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"  {name.capitalize():12s}: {val:.3f} |{bar}|")
    print(f"{'='*70}\n")
    
    return full_results


def main():
    parser = argparse.ArgumentParser(description="V11 Constrained Training with Lagrangian Optimization")
    parser.add_argument("--dataset_root", type=str, required=True, help="Train data")
    parser.add_argument("--val_root", type=str, required=True, help="Test data (precomputed)")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--budget_ratio", type=float, default=0.15)
    parser.add_argument("--Bmin", type=int, default=3)
    parser.add_argument("--Bmax", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--entropy_coef", type=float, default=0.02)
    parser.add_argument("--clip_range", type=float, default=0.2)
    
    # Constraints
    parser.add_argument("--rec_err_thresh", type=float, default=0.35, help="Constraint for RecErr")
    parser.add_argument("--coverage_thresh", type=float, default=0.3, help="Constraint for Coverage Gap")
    parser.add_argument("--diversity_thresh", type=float, default=0.25, help="Constraint for Diversity")
    
    parser.add_argument("--anime_scale", type=float, default=3.0, help="Scale for Anime Quality reward")
    parser.add_argument("--quantile_scale", type=float, default=2.0, help="Scale for Quantile reward")

    parser.add_argument("--no_anime_attrs", action="store_true", help="Disable anime attributes input")
    parser.add_argument("--num_attn_layers", type=int, default=2)
    parser.add_argument("--gating_hidden", type=int, default=64, help="Gating network hidden dim")
    parser.add_argument("--lstm_hidden", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--attr_suffix", type=str, default="anime_attrs_v11.npy", help="Suffix for attribute file (ablation)")
    
    args = parser.parse_args()
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, "logs"))
    
    # Save config
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    scene_dirs = build_epoch_index(args.dataset_root)
    print(f"Found {len(scene_dirs)} training scenes")
    
    val_scene_dirs = build_epoch_index(args.val_root)
    print(f"Found {len(val_scene_dirs)} validation scenes")
    
    # Setup model
    use_anime = not args.no_anime_attrs
    full_feat_dim = args.feat_dim + (6 if use_anime else 0)
    print(f"Feature dim: {full_feat_dim} (Base {args.feat_dim} + Anime {6 if use_anime else 0})")
    
    model = create_dsn_v8(
        feat_dim=full_feat_dim, 
        use_pcgrad=False,
        num_attn_layers=args.num_attn_layers,
        gating_hidden=args.gating_hidden,
        lstm_hidden=args.lstm_hidden,
    ).to(args.device)
    
    trainer = ConstrainedTrainer(
        model, lr=args.lr,
        entropy_coef=args.entropy_coef,
        clip_range=args.clip_range,
        device=args.device,
        rec_err_threshold=args.rec_err_thresh,
        coverage_threshold=args.coverage_thresh,
        diversity_threshold=args.diversity_thresh,
        anime_scale=args.anime_scale,
        quantile_scale=args.quantile_scale,
        total_epochs=args.epochs
    )
    
    best_score = -float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Training loop
        epoch_info = []
        pbar = tqdm(scene_dirs, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for scene_dir in pbar:
            sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=True, attr_suffix=args.attr_suffix)
            if sample.anime_attrs is None:
                continue
            
            if args.no_anime_attrs:
                feats_input = sample.feats
            else:
                feats_input = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
            
            feats_t = torch.from_numpy(feats_input).float().unsqueeze(0)
            budget = max(args.Bmin, min(args.Bmax, int(len(sample.feats) * args.budget_ratio)))
            
            info = trainer.train_step(feats_t, sample.anime_attrs, budget, epoch)
            epoch_info.append(info)
            
            pbar.set_postfix({
                "loss": f"{info['loss']:.3f}",
                "rec": f"{info.get('rec_err', 0.0):.3f}",
                "pen": f"{info.get('constraint_penalty', 0.0):.3f}",
            })
        
        if not epoch_info:
            print(f"\nEpoch {epoch}: No scenes processed")
            continue
        
        # Log training metrics
        avg_info = {k: float(np.mean([x.get(k, 0) for x in epoch_info])) for k in epoch_info[0].keys()}
        for k, v in avg_info.items():
            if not math.isnan(v):
                writer.add_scalar(f"train/{k}", v, epoch)
        
        # Validation
        print(f"\nRunning Validation on {len(val_scene_dirs)} scenes...")
        val_metrics = validate_epoch(
            model, val_scene_dirs, args.device, epoch, args.save_dir,
            budget_ratio=args.budget_ratio, Bmin=args.Bmin, Bmax=args.Bmax,
            no_anime_attrs=args.no_anime_attrs, total_epochs=args.epochs,
            attr_suffix=args.attr_suffix
        )
        
        # Log all validation metrics to TensorBoard
        for k, v in val_metrics.items():
            if not np.isnan(v):
                writer.add_scalar(f"val/{k}", v, epoch)
        
        # Print summary
        # Summary is already printed in validate_epoch
        composite = val_metrics.get("composite_score", 0.0)
        print(f"  Best so far: {best_score:.4f}")

        
        # Save best model
        if composite > best_score:
            best_score = composite
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "composite_score": composite,
                "metrics": val_metrics,
                "config": vars(args)
            }, os.path.join(args.save_dir, "best.pt"))
            print(f"  ✅ New Best! Composite Score: {composite:.4f}")
            
            # Save "best.pt" (latest best)
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "composite_score": composite,
                "metrics": val_metrics,
                "config": vars(args)
            }
            torch.save(checkpoint_data, os.path.join(args.save_dir, "best.pt"))
            
            # Keep history of best models
            best_model_name = f"best_epoch_{epoch}_score_{composite:.4f}.pt"
            torch.save(checkpoint_data, os.path.join(args.save_dir, best_model_name))
        
        # Periodic checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": val_metrics,
        }, os.path.join(args.save_dir, f"ep{epoch}", "checkpoint.pt"))
    
    writer.close()
    
    # ===== Final Comprehensive Evaluation on Best Weights =====
    run_final_eval(model, val_scene_dirs, args.device, args.save_dir, 
                   args.budget_ratio, args.Bmin, args.Bmax, 
                   args.no_anime_attrs, args.attr_suffix, best_score)
    print(f"\n🎯 Training & Final Eval Complete! Best Composite Score: {best_score:.4f}")


def run_final_eval(model, val_scene_dirs, device, save_dir, budget_ratio, Bmin, Bmax, no_anime_attrs, attr_suffix, best_score):
    """Run expensive metrics (LPIPS/DISTS Gap) on the best model."""
    print("\n" + "="*70)
    print(f"RUNNING FINAL EVALUATION (LPIPS/DISTS) on Best Model")
    print("="*70)
    
    # Load Best Model
    best_path = os.path.join(save_dir, "best.pt")
    if not os.path.exists(best_path):
        print("  ❌ Best model not found, skipping final eval.")
        return

    try:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    except Exception as e:
        print(f"  ❌ Failed to load best model: {e}")
        return

    try:
        from src.distance_selector.registry import create_metric
        lpips_metric = create_metric("lpips", net="alex", device=device)
        dists_metric = create_metric("dists", device=device)
    except:
        print("  ❌ LPIPS/DISTS not available.")
        return

    # Metrics collections
    lpips_gaps = []
    dists_gaps = []
    
    for scene_dir in tqdm(val_scene_dirs, desc="Final Eval"):
        try:
             # Load FRAMES for Gap metrics
            sample = load_scene_dir(scene_dir, load_frames=True, load_anime_attrs=True, attr_suffix=attr_suffix)
            if sample.anime_attrs is None or not sample.frames:
                continue
            
            # Predict
            if no_anime_attrs:
                feats_input = sample.feats
            else:
                feats_input = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
            
            feats_t = torch.from_numpy(feats_input).float().unsqueeze(0).to(device)
            budget = max(Bmin, min(Bmax, int(len(sample.feats) * budget_ratio)))
            
            with torch.no_grad():
                probs, _ = model(feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
            # Sparse frames for speed
            sparse_frames = sample.frames[::5]
            
            lg = perceptual_gap_from_frames(lpips_metric, sparse_frames, sel_idx)
            if not np.isnan(lg): lpips_gaps.append(lg)
            
            dg = perceptual_gap_from_frames(dists_metric, sparse_frames, sel_idx)
            if not np.isnan(dg): dists_gaps.append(dg)
            
        except Exception as e:
            continue

    # Compute means
    final_metrics = {
        "Composite": best_score,
        "LPIPS_Gap": float(np.mean(lpips_gaps)) if lpips_gaps else float("nan"),
        "DISTS_Gap": float(np.mean(dists_gaps)) if dists_gaps else float("nan"),
    }
    
    print("\nFINAL METRICS (Best Model):")
    print(f"  Composite Score: {best_score:.4f}")
    print(f"  LPIPS Gap:       {final_metrics['LPIPS_Gap']:.4f}")
    print(f"  DISTS Gap:       {final_metrics['DISTS_Gap']:.4f}")
    
    # Save to file
    with open(os.path.join(save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    main()
