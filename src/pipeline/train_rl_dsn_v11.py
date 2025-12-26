#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 Training Pipeline: Quality + Diversity RL

Key features:
1. 2-head architecture (rec + anime) from V9
2. Mixed local + global reward optimization
3. Temporal diversity via DPP and gap penalty
4. Curriculum learning: quality first, diversity later
5. Comprehensive logging with MPR tracking
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets import build_epoch_index, load_scene_dir, SceneSample
from src.models.dsn_v8 import DSNMultiTaskV8, create_dsn_v8, PCGradOptimizer
from src.rl.premium_rewards_v11 import PremiumRewardV11, DiversityConfig
from src.rl.premium_rewards_v9 import ConstraintConfig, DPPConfig
from src.rl.gae import GAEComputer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_mpr(quality: np.ndarray, sel_idx: List[int]) -> float:
    """Compute Mean Percentile Rank."""
    if len(sel_idx) == 0:
        return 0.5
    T = len(quality)
    ranks = np.argsort(np.argsort(quality))
    percentiles = ranks / max(1, T - 1)
    return float(np.mean(percentiles[sel_idx]))


def compute_top10_recall(quality: np.ndarray, sel_idx: List[int]) -> float:
    """Compute Top-10% recall."""
    if len(sel_idx) == 0:
        return 0.0
    T = len(quality)
    k10 = max(1, int(T * 0.1))
    top10_idx = set(np.argsort(quality)[-k10:])
    return len(set(sel_idx) & top10_idx) / k10


class PPOTrainerV11:
    """
    V11 PPO Trainer with 2-head architecture and diversity-aware training.
    """
    
    def __init__(
        self,
        model: DSNMultiTaskV8,
        reward_system: PremiumRewardV11,
        gae_computer: GAEComputer,
        optimizer: torch.optim.Optimizer,
        pcgrad_optimizer: Optional[PCGradOptimizer] = None,
        clip_range: float = 0.1,
        n_ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
        use_pcgrad: bool = True,
        use_dpp: bool = True,
    ):
        self.model = model
        self.reward_system = reward_system
        self.gae_computer = gae_computer
        self.optimizer = optimizer
        self.pcgrad_optimizer = pcgrad_optimizer
        self.clip_range = clip_range
        self.n_ppo_epochs = n_ppo_epochs
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.use_pcgrad = use_pcgrad and pcgrad_optimizer is not None
        self.use_dpp = use_dpp
        
        # Running stats for reward normalization
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
        return float(np.clip((reward - self.reward_mean) / (self.reward_std + 1e-8), -5.0, 5.0))
    
    def compute_per_task_loss(
        self,
        old_log_prob: torch.Tensor,
        new_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        value: torch.Tensor,
        return_target: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PPO loss for a single task."""
        ratio = torch.exp(new_log_prob - old_log_prob)
        ratio = torch.clamp(ratio, 0.01, 100.0)
        
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        
        value_loss = F.mse_loss(value.view(-1), return_target.view(-1))
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        return policy_loss.mean() + self.vf_coef * value_loss - self.entropy_coef * entropy
    
    def train_step(
        self,
        features: torch.Tensor,       # (1, T, D) concatenated features
        anime_attrs: np.ndarray,      # (T, 6)
        rel_positions: Optional[np.ndarray],  # (T,) V11
        budget: int,
        current_epoch: int,
    ) -> Dict[str, float]:
        """Single training step for one scene."""
        self.model.train()
        features = features.to(self.device)
        T = features.shape[1]
        
        # 1. Forward pass (get probabilities)
        with torch.no_grad():
            merged_probs, _ = self.model(features)
            task_outputs = self.model(features, return_all_tasks=True)
            _, values_rec = task_outputs["rec"]
            _, values_anime = task_outputs["anime"]
        
        # 2. Sample action (DPP or greedy)
        feats_np = features.squeeze(0).cpu().numpy()[:, :512]  # Original CLIP feats
        
        if self.use_dpp:
            sel_idx, dpp_info = self.reward_system.select_with_dpp(feats_np, anime_attrs, budget)
            old_log_prob = torch.log(merged_probs.squeeze(0)[sel_idx] + 1e-8).sum()
        else:
            probs_np = merged_probs.squeeze(0).cpu().numpy()
            probs_safe = np.clip(probs_np, 1e-8, 1.0)
            sel_idx = sorted(np.argsort(probs_safe)[-budget:].tolist())
            old_log_prob = torch.log(merged_probs.squeeze(0)[sel_idx] + 1e-8).sum()
            dpp_info = {}
        
        old_log_prob = old_log_prob.detach()
        
        # 3. Compute rewards (V11 with diversity)
        rewards, components = self.reward_system.compute_reward(
            feats_all=feats_np,
            sel_idx=sel_idx,
            anime_attrs=anime_attrs,
            rel_positions=rel_positions,
            current_epoch=current_epoch
        )
        
        # Clip and normalize rewards
        total_reward = float(np.clip(rewards["total"], -20.0, 20.0))
        self.update_reward_stats(total_reward)
        
        reward_rec = float(np.clip(-rewards.get("constraint_penalty", 0), -10.0, 10.0))
        reward_anime = float(np.clip(
            rewards.get("anime", 0) + rewards.get("percentile", 0) + 
            0.5 * rewards.get("diversity", 0),  # Include diversity in anime head
            -10.0, 10.0
        ))
        
        # Compute advantages
        advantage_rec = torch.tensor([self.normalize_reward(reward_rec)], device=self.device, dtype=torch.float32)
        advantage_anime = torch.tensor([self.normalize_reward(reward_anime)], device=self.device, dtype=torch.float32)
        
        # 4. PPO Update
        total_loss = 0.0
        for _ in range(self.n_ppo_epochs):
            if self.use_pcgrad:
                self.pcgrad_optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
            
            merged_probs_new, _, alpha = self.model(features, return_gating=True)
            task_outputs_new = self.model(features, return_all_tasks=True)
            probs_rec_new, values_rec_new = task_outputs_new["rec"]
            probs_anime_new, values_anime_new = task_outputs_new["anime"]
            
            new_log_prob = torch.log(merged_probs_new.squeeze(0)[sel_idx] + 1e-8).sum()
            
            loss_rec = self.compute_per_task_loss(
                old_log_prob, new_log_prob, advantage_rec,
                values_rec_new.mean(),
                torch.tensor([reward_rec], device=self.device, dtype=torch.float32),
                probs_rec_new
            )
            loss_anime = self.compute_per_task_loss(
                old_log_prob, new_log_prob, advantage_anime,
                values_anime_new.mean(),
                torch.tensor([reward_anime], device=self.device, dtype=torch.float32),
                probs_anime_new
            )
            
            # Check for NaN
            if torch.isnan(loss_rec) or torch.isnan(loss_anime):
                continue
            
            if self.use_pcgrad:
                self.pcgrad_optimizer.backward([loss_rec, loss_anime], self.model)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.pcgrad_optimizer.step()
                iter_loss = (loss_rec.item() + loss_anime.item()) / 2
            else:
                loss = alpha.mean() * loss_rec + (1 - alpha.mean()) * loss_anime
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                iter_loss = loss.item()
            
            total_loss += iter_loss
        
        # Compute MPR for logging
        quality = anime_attrs.mean(axis=1) if anime_attrs is not None else feats_np.mean(axis=1)
        mpr = compute_mpr(quality, sel_idx)
        top10 = compute_top10_recall(quality, sel_idx)
        
        train_info = {
            "loss": total_loss / max(1, self.n_ppo_epochs),
            "mpr": mpr,
            "top10_recall": top10,
            "reward_total": total_reward,
            "reward_anime": rewards.get("anime", 0),
            "reward_diversity": rewards.get("diversity", 0),
            "reward_dpp": rewards.get("dpp", 0),
            "constraint_penalty": rewards.get("constraint_penalty", 0),
            "min_gap": components.get("div_min_gap", 0),
            "alpha_mean": float(self.model.get_gating_stats().get("alpha_mean", 0.5)),
        }
        
        return train_info


def main():
    parser = argparse.ArgumentParser(description="V11 RL Training")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--budget_ratio", type=float, default=0.10)
    parser.add_argument("--Bmin", type=int, default=3)
    parser.add_argument("--Bmax", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_pcgrad", type=int, default=1)
    parser.add_argument("--use_dpp", type=int, default=1)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Save config
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    scene_dirs = build_epoch_index(args.dataset_root)
    print(f"Found {len(scene_dirs)} scenes")
    
    # Determine feature dimension
    sample = load_scene_dir(scene_dirs[0], load_frames=False, load_anime_attrs=True)
    use_anime_attrs = sample.anime_attrs is not None
    full_feat_dim = args.feat_dim + (6 if use_anime_attrs else 0)
    
    print(f"Feature dim: {full_feat_dim} (CLIP: {args.feat_dim}, anime: {6 if use_anime_attrs else 0})")
    
    # Create model
    model = create_dsn_v8(feat_dim=full_feat_dim, use_pcgrad=bool(args.use_pcgrad)).to(args.device)
    
    # Create reward system (V11)
    constraint_config = ConstraintConfig(
        rec_err_threshold=0.4,
        coverage_threshold=0.35,
        diversity_threshold=0.2,
        lambda_lr=0.005,
        lambda_max=10.0,
    )
    dpp_config = DPPConfig(beta=1.0, quality_power=1.0, candidate_ratio=0.3)
    diversity_config = DiversityConfig(
        min_gap_ratio=0.3,
        gap_penalty_weight=2.0,
        dpp_diversity_weight=1.0,
    )
    
    reward_system = PremiumRewardV11(
        constraint_config=constraint_config,
        dpp_config=dpp_config,
        diversity_config=diversity_config,
        anime_scale=5.0,
        percentile_scale=3.0,
    )
    
    gae_computer = GAEComputer(gamma=0.99, lam=0.95)
    
    # Create optimizers
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pcgrad_optimizer = PCGradOptimizer(base_optimizer) if args.use_pcgrad else None
    
    trainer = PPOTrainerV11(
        model=model,
        reward_system=reward_system,
        gae_computer=gae_computer,
        optimizer=base_optimizer,
        pcgrad_optimizer=pcgrad_optimizer,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        device=args.device,
        use_pcgrad=bool(args.use_pcgrad),
        use_dpp=bool(args.use_dpp),
    )
    
    best_mpr = 0.0
    
    for epoch in range(1, args.epochs + 1):
        epoch_info = []
        pbar = tqdm(scene_dirs, desc=f"Epoch {epoch}/{args.epochs}")
        
        for scene_dir in pbar:
            # Load scene
            sample = load_scene_dir(
                scene_dir, 
                load_frames=False, 
                load_anime_attrs=True,
                load_vlm_scores=False
            )
            
            feats_base = sample.feats
            if sample.anime_attrs is None:
                continue
            
            # Concatenate features
            feats_full = np.concatenate([feats_base, sample.anime_attrs], axis=1)
            feats_t = torch.from_numpy(feats_full).float().unsqueeze(0)
            
            # Load rel_positions if available (V11)
            rel_pos_path = Path(scene_dir) / "rel_positions.npy"
            rel_positions = np.load(rel_pos_path) if rel_pos_path.exists() else None
            
            # Compute budget
            budget = max(args.Bmin, min(args.Bmax, int(len(feats_base) * args.budget_ratio)))
            
            # Train step
            info = trainer.train_step(feats_t, sample.anime_attrs, rel_positions, budget, epoch)
            epoch_info.append(info)
            
            pbar.set_postfix({
                "loss": f"{info['loss']:.3f}",
                "mpr": f"{info['mpr']:.3f}",
                "div": f"{info['reward_diversity']:.2f}"
            })
        
        # Epoch summary
        avg_info = {k: float(np.mean([x.get(k, 0) for x in epoch_info])) for k in epoch_info[0].keys()}
        
        # Log to tensorboard
        for k, v in avg_info.items():
            if not np.isnan(v):
                writer.add_scalar(f"train/{k}", v, epoch)
        
        print(f"\nEpoch {epoch}: Loss={avg_info['loss']:.4f}, MPR={avg_info['mpr']:.4f}, "
              f"Top10={avg_info['top10_recall']:.3f}, Div={avg_info['reward_diversity']:.3f}")
        
        # Save checkpoint
        if avg_info['mpr'] > best_mpr:
            best_mpr = avg_info['mpr']
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "mpr": avg_info['mpr'],
                "config": vars(args)
            }, os.path.join(args.save_dir, "best.pt"))
            print(f"  ✅ New best MPR: {best_mpr:.4f}")
        
        # Save periodic checkpoints
        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "mpr": avg_info['mpr'],
            }, os.path.join(args.save_dir, f"ep{epoch}.pt"))
    
    writer.close()
    print(f"\n🎯 Training Complete! Best MPR: {best_mpr:.4f}")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
