#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V10 STABLE: VLM-Guided RL Training with NaN Protection

Key features:
1. Simplified reward: Direct quality percentile optimization
2. Numerical stability: Extensive clipping and NaN checks
3. Conservative hyperparameters for stable training
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets import build_epoch_index, load_scene_dir
from src.models.dsn_v8 import DSNMultiTaskV8, create_dsn_v8


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe log that avoids -inf."""
    return torch.log(torch.clamp(x, min=eps))


def check_nan(tensor: torch.Tensor, name: str) -> bool:
    """Check for NaN and return True if found."""
    if torch.isnan(tensor).any():
        print(f"  ⚠️ NaN detected in {name}")
        return True
    return False


class QualityFocusedTrainer:
    """
    Simplified trainer focusing purely on quality percentile maximization.
    No complex multi-task losses - just maximize quality of selected frames.
    """
    
    def __init__(
        self,
        model: DSNMultiTaskV8,
        lr: float = 5e-5,
        clip_range: float = 0.1,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.3,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
    def update_reward_stats(self, reward: float):
        """Update running mean/std for reward normalization."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        if self.reward_count > 1:
            self.reward_std = max(0.1, self.reward_std * 0.99 + abs(delta) * 0.01)
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        return (reward - self.reward_mean) / (self.reward_std + 1e-8)
    
    def compute_quality_reward(
        self,
        anime_attrs: np.ndarray,  # (T, 6)
        vlm_scores: Optional[np.ndarray],  # (T, 8) or None
        sel_idx: List[int],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Simple quality reward based on percentile rank.
        Goal: Select frames with high quality scores.
        """
        T = len(anime_attrs)
        if len(sel_idx) == 0:
            return 0.0, {}
        
        # Use VLM scores if available, else use anime_attrs
        if vlm_scores is not None:
            quality = vlm_scores.mean(axis=1)  # (T,)
        else:
            quality = anime_attrs.mean(axis=1)  # (T,)
        
        # Compute percentile ranks
        ranks = np.argsort(np.argsort(quality))
        percentiles = ranks / max(1, T - 1)  # 0 to 1
        
        # Selected frame percentiles
        sel_percentiles = percentiles[sel_idx]
        mean_percentile = float(np.mean(sel_percentiles))
        
        # Top-10% recall bonus
        k10 = max(1, int(T * 0.1))
        top10_idx = set(np.argsort(quality)[-k10:])
        top10_recall = len(set(sel_idx) & top10_idx) / k10
        
        # Reward: linear combo of mean percentile and top10 recall
        # Scale to roughly [-2, 2] range
        reward = (mean_percentile - 0.5) * 4.0 + top10_recall * 2.0
        
        info = {
            "mean_percentile": mean_percentile,
            "top10_recall": top10_recall,
            "raw_reward": reward,
        }
        
        return reward, info
    
    def train_step(
        self,
        features: torch.Tensor,  # (1, T, D)
        anime_attrs: np.ndarray,
        vlm_scores: Optional[np.ndarray],
        budget: int,
    ) -> Dict[str, float]:
        """Single training step with simplified loss."""
        self.model.train()
        features = features.to(self.device)
        T = features.shape[1]
        
        # 1. Get action probabilities
        with torch.no_grad():
            probs_old, _ = self.model(features)
            probs_old = probs_old.squeeze(0)  # (T,)
            
        # 2. Sample action (select top-K by probability)
        probs_np = probs_old.cpu().numpy()
        probs_safe = np.clip(probs_np, 1e-6, 1.0)
        
        # Greedy selection: top-K by probability
        sel_idx = sorted(np.argsort(probs_safe)[-budget:].tolist())
        
        # 3. Compute reward
        reward, reward_info = self.compute_quality_reward(anime_attrs, vlm_scores, sel_idx)
        self.update_reward_stats(reward)
        norm_reward = self.normalize_reward(reward)
        norm_reward = float(np.clip(norm_reward, -5.0, 5.0))
        
        # 4. Policy gradient update
        self.optimizer.zero_grad()
        
        probs_new, values = self.model(features)
        probs_new = probs_new.squeeze(0)  # (T,)
        values = values.squeeze()
        
        # Log probabilities of selected frames
        old_log_prob = safe_log(probs_old[sel_idx]).sum().detach()
        new_log_prob = safe_log(probs_new[sel_idx]).sum()
        
        # PPO clipped objective
        ratio = torch.exp(new_log_prob - old_log_prob)
        ratio = torch.clamp(ratio, 0.01, 100.0)  # Strong clipping
        
        advantage = torch.tensor([norm_reward], device=self.device, dtype=torch.float32)
        
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        
        # Value loss (simple baseline)
        value_target = torch.tensor([reward], device=self.device, dtype=torch.float32)
        value_loss = F.mse_loss(values.mean().unsqueeze(0), value_target)
        
        # Entropy bonus (encourage exploration early)
        entropy = -(probs_new * safe_log(probs_new)).sum()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # NaN check
        if check_nan(loss, "loss"):
            return {"loss": 0.0, "skipped": 1.0, **reward_info}
        
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "norm_reward": norm_reward,
            **reward_info,
        }


def main():
    parser = argparse.ArgumentParser(description="V10 STABLE Training")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--budget_ratio", type=float, default=0.10)
    parser.add_argument("--Bmin", type=int, default=3)
    parser.add_argument("--Bmax", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.1)
    
    args = parser.parse_args()
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, "logs"))
    
    scene_dirs = build_epoch_index(args.dataset_root)
    full_feat_dim = args.feat_dim + 6  # anime_attrs
    
    model = create_dsn_v8(feat_dim=full_feat_dim, use_pcgrad=False).to(args.device)
    trainer = QualityFocusedTrainer(
        model, lr=args.lr, 
        entropy_coef=args.entropy_coef,
        clip_range=args.clip_range,
        device=args.device
    )
    
    best_mpr = 0.0
    
    for epoch in range(1, args.epochs + 1):
        epoch_info = []
        pbar = tqdm(scene_dirs, desc=f"Epoch {epoch}")
        
        for scene_dir in pbar:
            sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=True, load_vlm_scores=True)
            
            feats_base = sample.feats
            if sample.anime_attrs is None:
                continue
                
            feats_full = np.concatenate([feats_base, sample.anime_attrs], axis=1)
            feats_t = torch.from_numpy(feats_full).float().unsqueeze(0)
            
            budget = max(args.Bmin, min(args.Bmax, int(len(feats_base) * args.budget_ratio)))
            
            info = trainer.train_step(feats_t, sample.anime_attrs, sample.vlm_scores, budget)
            epoch_info.append(info)
            
            pbar.set_postfix({
                "loss": f"{info['loss']:.3f}",
                "mpr": f"{info.get('mean_percentile', 0):.3f}"
            })
        
        # Epoch summary
        avg_info = {k: np.mean([x.get(k, 0) for x in epoch_info]) for k in epoch_info[0].keys()}
        
        for k, v in avg_info.items():
            if isinstance(v, (int, float)) and not math.isnan(v):
                writer.add_scalar(f"train/{k}", v, epoch)
        
        mpr = avg_info.get("mean_percentile", 0)
        print(f"\nEpoch {epoch}: Loss={avg_info['loss']:.4f}, MPR={mpr:.4f}, Top10={avg_info.get('top10_recall', 0):.3f}")
        
        # Save checkpoint
        if mpr > best_mpr:
            best_mpr = mpr
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "mpr": mpr,
                "args": vars(args)
            }, os.path.join(args.save_dir, "best.pt"))
            print(f"  ✅ New best MPR: {mpr:.4f}")
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "mpr": mpr,
        }, os.path.join(args.save_dir, f"ep{epoch}.pt"))
    
    writer.close()
    print(f"\n🎯 Training Complete! Best MPR: {best_mpr:.4f}")


if __name__ == "__main__":
    main()
