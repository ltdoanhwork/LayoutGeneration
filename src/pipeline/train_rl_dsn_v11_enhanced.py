#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 ENHANCED: Quality + Diversity + RecErr/Frechet Monitoring + Rich Viz

Based on V11 Simplified, but adds:
1. Computation of RecErr and Frechet Distance (logged, not necessarily optimized)
2. Integrated Validation on precomputed test set
3. Automatic Visualization (Reward Distribution, Cityscape, etc.)
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

# Import core components
from src.datasets import build_epoch_index, load_scene_dir
from src.models.dsn_v8 import DSNMultiTaskV8, create_dsn_v8
from src.rl.rewards import reward_combo_v4
from eval.visualize_distribution import create_comprehensive_dashboard

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))

class EnhancedTrainer:
    """
    Enhanced trainer with extra metrics and validation capability.
    """
    
    def __init__(
        self,
        model: DSNMultiTaskV8,
        lr: float = 1e-4,
        clip_range: float = 0.2,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
        diversity_weight: float = 0.5,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.diversity_weight = diversity_weight
        
        # Running statistics
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
    
    def compute_reward_enhanced(
        self,
        features: np.ndarray, # (T, D)
        anime_attrs: np.ndarray,  # (T, 6)
        sel_idx: List[int],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward using reward_combo_v4 to get all metrics (RecErr, Frechet),
        but return the simple PPO reward for optimization.
        """
        T = len(anime_attrs)
        if len(sel_idx) == 0:
            return 0.0, {"mpr": 0.5, "top10": 0.0, "diversity": 0.0, "RecErr": 0.0, "Frechet": 0.0}
        
        # 1. Compute Extra Metrics (RecErr, Frechet) using reward_combo_v4
        # We pass fake weights just to trigger computation, but we won't use the returned sum
        _, components = reward_combo_v4(
            feats_all=features,
            sel_idx=sel_idx,
            w_rec=1.0, 
            w_fd=1.0,
            return_components=True
        )
        rec_err = -components.get("rec", 0.0) # reward_combo returns negative error
        frechet = -components.get("fd", 0.0)
        
        # 2. Compute Optimization Reward (MPR + Diversity) - Same as Simple V11
        # Quality: mean of anime attributes
        quality = anime_attrs.mean(axis=1)
        
        # Compute percentile ranks
        ranks = np.argsort(np.argsort(quality))
        percentiles = ranks / max(1, T - 1)
        
        # Mean Percentile Rank
        sel_percentiles = percentiles[sel_idx]
        mpr = float(np.mean(sel_percentiles))
        
        # Top-10% recall
        k10 = max(1, int(T * 0.1))
        top10_idx = set(np.argsort(quality)[-k10:])
        top10 = len(set(sel_idx) & top10_idx) / k10
        
        # Diversity: penalize clustering
        if len(sel_idx) >= 2:
            sorted_idx = sorted(sel_idx)
            gaps = np.diff(sorted_idx)
            expected_gap = T / (len(sel_idx) + 1)
            min_gap = float(np.min(gaps))
            diversity_score = min(1.0, min_gap / expected_gap)
        else:
            diversity_score = 0.0
        
        # Combined reward (scale to ~[-3, 3])
        quality_reward = (mpr - 0.5) * 6.0  # Range: -3 to 3
        diversity_reward = (diversity_score - 0.5) * 2.0  # Range: -1 to 1
        
        reward = quality_reward + self.diversity_weight * diversity_reward
        
        info = {
            "mpr": mpr,
            "top10": top10,
            "diversity": diversity_score,
            "quality_reward": quality_reward,
            "diversity_reward": diversity_reward,
            "RecErr": rec_err,
            "Frechet": frechet
        }
        
        return reward, info
    
    def train_step(
        self,
        features: torch.Tensor,  # (1, T, D)
        anime_attrs: np.ndarray,
        budget: int,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        features = features.to(self.device)
        T = features.shape[1]
        
        # 1. Get probabilities
        with torch.no_grad():
            probs_old, _ = self.model(features)
            probs_old = probs_old.squeeze(0)  # (T,)
        
        # 2. Select top-K by probability (same as eval)
        probs_np = probs_old.cpu().numpy()
        sel_idx = sorted(np.argsort(probs_np)[-budget:].tolist())
        
        # 3. Compute reward
        # Extract features numpy
        features_np = features.squeeze(0).cpu().numpy()
        # CAREFUL: features contains anime_attrs if concatenated. 
        # reward_combo needs features for RecErr. 
        # Usually feats are already there.
        
        reward, reward_info = self.compute_reward_enhanced(features_np, anime_attrs, sel_idx)
        self.update_reward_stats(reward)
        norm_reward = self.normalize_reward(reward)
        
        # 4. PPO update
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
) -> Dict[str, float]:
    """
    Run validation on test set, compute metrics, and save visualizations.
    """
    model.eval()
    
    # Create viz dir for this epoch
    viz_dir = os.path.join(save_dir, f"ep{epoch}", "viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    all_metrics = []
    
    # For visualization, pick a few interesting samples (randomly or fixed)
    # We will visualize first 5 scenes
    viz_samples = []
    
    for i, scene_dir in enumerate(tqdm(val_scene_dirs, desc="Validating")):
        try:
            sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=True)
            if sample.anime_attrs is None:
                continue
                
            # Construct features
            feats_full = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
            feats_t = torch.from_numpy(feats_full).float().unsqueeze(0).to(device)
            T = len(sample.feats)
            budget = max(Bmin, min(Bmax, int(T * budget_ratio)))
            
            # Predict
            with torch.no_grad():
                probs, _ = model(feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            
            # Compute Validation Metrics
            # Use same enhanced reward function logic but just for metrics
            _, components = reward_combo_v4(
                feats_all=sample.feats if sample.feats.shape[1] == 512 else feats_full, # Handle dim match
                sel_idx=sel_idx,
                w_rec=1.0, w_fd=1.0, return_components=True
            )
            rec_err = -components.get("rec", 0.0)
            frechet = -components.get("fd", 0.0)
            
            quality = sample.anime_attrs.mean(axis=1)
            ranks = np.argsort(np.argsort(quality))
            percentiles = ranks / max(1, T - 1)
            mpr = float(np.mean(percentiles[sel_idx]))
            
            k10 = max(1, int(T * 0.1))
            top10_idx = set(np.argsort(quality)[-k10:])
            top10 = len(set(sel_idx) & top10_idx) / k10
            
            metrics = {
                "mpr": mpr,
                "top10": top10,
                "RecErr": rec_err,
                "Frechet": frechet
            }
            all_metrics.append(metrics)
            
            # Viz for first 5
            if i < 5:
                viz_path = os.path.join(viz_dir, f"scene_{i:04d}_dashboard.png")
                create_comprehensive_dashboard(
                    attrs_all=sample.anime_attrs,
                    sel_idx=sel_idx,
                    metrics_result={
                        "mean_percentile_rank": mpr,
                        "top_10_coverage": top10,
                        "zscore_improvement": 0.0 # simplified
                    },
                    save_path=viz_path,
                    title=f"Epoch {epoch} - Scene {scene_dir.name}"
                )
                
        except Exception as e:
            print(f"Error validating {scene_dir}: {e}")
            continue

    if not all_metrics:
        return {}
        
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
    
    # Save validation summary
    val_result_path = os.path.join(save_dir, f"ep{epoch}", "val_results.json")
    with open(val_result_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)
        
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="V11 Enhanced Training")
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
    parser.add_argument("--diversity_weight", type=float, default=0.3)
    
    args = parser.parse_args()
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, "logs"))
    
    # Save config
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load Train Data
    scene_dirs = build_epoch_index(args.dataset_root)
    print(f"Found {len(scene_dirs)} training scenes")
    
    # Load Val Data
    val_scene_dirs = build_epoch_index(args.val_root)
    print(f"Found {len(val_scene_dirs)} validation scenes in {args.val_root}")
    
    # Check first scene to determine dim
    sample = load_scene_dir(scene_dirs[0], load_frames=False, load_anime_attrs=True)
    use_anime = sample.anime_attrs is not None
    full_feat_dim = args.feat_dim + (6 if use_anime else 0)
    print(f"Feature dim: {full_feat_dim} (Base {args.feat_dim} + Anime {6 if use_anime else 0})")
    
    model = create_dsn_v8(feat_dim=full_feat_dim, use_pcgrad=False).to(args.device)
    trainer = EnhancedTrainer(
        model, lr=args.lr,
        entropy_coef=args.entropy_coef,
        clip_range=args.clip_range,
        device=args.device,
        diversity_weight=args.diversity_weight,
    )
    
    best_mpr = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # --- TRAINING LOOP ---
        epoch_info = []
        pbar = tqdm(scene_dirs, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for scene_dir in pbar:
            sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=True)
            if sample.anime_attrs is None:
                continue
            
            feats_full = np.concatenate([sample.feats, sample.anime_attrs], axis=1)
            feats_t = torch.from_numpy(feats_full).float().unsqueeze(0)
            
            budget = max(args.Bmin, min(args.Bmax, int(len(sample.feats) * args.budget_ratio)))
            
            info = trainer.train_step(feats_t, sample.anime_attrs, budget)
            epoch_info.append(info)
            
            pbar.set_postfix({
                "loss": f"{info['loss']:.3f}",
                "mpr": f"{info['mpr']:.2f}",
                "rec": f"{info['RecErr']:.2f}"
            })
        
        if not epoch_info:
            print(f"\nEpoch {epoch}: No scenes processed")
            continue
        
        # Train Summary
        avg_info = {k: float(np.mean([x.get(k, 0) for x in epoch_info])) for k in epoch_info[0].keys()}
        for k, v in avg_info.items():
            if not math.isnan(v):
                writer.add_scalar(f"train/{k}", v, epoch)
        
        # --- VALIDATION LOOP ---
        print(f"\nRunning Validation on {len(val_scene_dirs)} scenes...")
        val_metrics = validate_epoch(
            model, val_scene_dirs, args.device, epoch, args.save_dir,
            budget_ratio=args.budget_ratio, Bmin=args.Bmin, Bmax=args.Bmax
        )
        
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        
        mpr = val_metrics.get("mpr", 0.0)
        rec = val_metrics.get("RecErr", 0.0)
        print(f"Epoch {epoch} Summary:")
        print(f"  Train: Loss={avg_info['loss']:.4f}, MPR={avg_info['mpr']:.3f}, Rec={avg_info['RecErr']:.3f}")
        print(f"  Val:   MPR={mpr:.3f}, Rec={rec:.3f}, Top10={val_metrics.get('top10',0):.3f}")
        print(f"  Viz saved to: {os.path.join(args.save_dir, f'ep{epoch}', 'viz')}")
        
        # Save best
        if mpr > best_mpr:
            best_mpr = mpr
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "mpr": mpr,
                "config": vars(args)
            }, os.path.join(args.save_dir, "best.pt"))
            print(f"  ✅ New best Validation MPR: {mpr:.4f}")
        
        # Periodic checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "mpr": mpr,
        }, os.path.join(args.save_dir, f"ep{epoch}", "checkpoint.pt"))
    
    writer.close()
    print(f"\n🎯 Training Complete! Best Val MPR: {best_mpr:.4f}")


if __name__ == "__main__":
    main()
