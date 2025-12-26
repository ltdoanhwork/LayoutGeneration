#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V10 VLM-Guided RL Training

Key improvements over V9:
1. VLM Distillation: Distills knowledge from VLM teacher into CLIP head.
2. Hybrid Reward: Combines CLIP-IQA and VLM-based quality rewards.
3. Adaptive Weighting: Uses VLM guidance early, standalone student later.
"""

from __future__ import annotations
import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from src.datasets import build_epoch_index, load_scene_dir
from src.models.dsn_v8 import DSNMultiTaskV8, create_dsn_v8, create_pcgrad_optimizer, PCGradOptimizer
from src.rl.premium_rewards_v10 import (
    PremiumRewardV10, 
    ConstraintConfig,
    DPPConfig,
    TestTimeScalingConfig,
)
from src.rl.gae import GAEComputer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PPOTrainerV10:
    """
    V10 PPO Trainer with VLM Distillation support.
    """
    
    def __init__(
        self,
        model: DSNMultiTaskV8,
        reward_system: PremiumRewardV10,
        gae_computer: GAEComputer,
        optimizer: torch.optim.Optimizer,
        pcgrad_optimizer: Optional[PCGradOptimizer] = None,
        clip_range: float = 0.2,
        n_ppo_epochs: int = 4,
        entropy_coef: float = 0.02,  # Reduced for stability
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

    def compute_per_task_loss(
        self,
        old_log_prob: torch.Tensor,
        new_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        value: torch.Tensor,
        return_target: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PPO loss for a single task"""
        ratio = torch.exp(new_log_prob - old_log_prob)
        ratio = torch.clamp(ratio, 0.0, 10.0) 
        
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        
        value_loss = F.mse_loss(value.view(-1), return_target.view(-1))
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        total_loss = policy_loss.mean() + self.vf_coef * value_loss - self.entropy_coef * entropy
        return total_loss

    def train_step(
        self,
        features: torch.Tensor,       # (1, T, D) concatenated features
        original_clip: torch.Tensor,  # (1, T, 512) for distillation
        vlm_scores: Optional[np.ndarray],
        anime_attrs: Optional[np.ndarray],
        budget: int,
        current_epoch: int,
    ) -> Dict[str, float]:
        """Single training step for one video with VLM distillation."""
        self.model.train()
        features = features.to(self.device)
        original_clip = original_clip.to(self.device)
        
        # 1. Distillation Step
        distill_loss = 0.0
        if vlm_scores is not None:
            vlm_targets = torch.from_numpy(vlm_scores).float().to(self.device)
            distill_loss = self.reward_system.train_distiller(original_clip.squeeze(0), vlm_targets)

        # 2. RL Forward Pass (Sampling)
        with torch.no_grad():
            merged_probs, _ = self.model(features)
            task_outputs = self.model(features, return_all_tasks=True)
            _, values_rec = task_outputs["rec"]
            _, values_anime = task_outputs["anime"]
        
        # Sample action
        feats_np = original_clip.squeeze(0).cpu().numpy()
        if self.use_dpp:
            sel_idx, dpp_info = self.reward_system.select_with_dpp(feats_np, anime_attrs, budget)
            old_log_prob = torch.log(merged_probs.squeeze(0)[sel_idx] + 1e-8).sum()
        else:
            probs_np = merged_probs.squeeze(0).cpu().numpy()
            probs_safe = np.clip(probs_np, 1e-8, 1.0)
            probs_safe /= probs_safe.sum()
            sel_idx = sorted(np.random.choice(len(probs_safe), size=min(budget, len(probs_safe)), replace=False, p=probs_safe).tolist())
            old_log_prob = torch.log(merged_probs.squeeze(0)[sel_idx] + 1e-8).sum()
            dpp_info = {}

        old_log_prob = old_log_prob.detach()

        # 3. Rewards
        rewards, components = self.reward_system.compute_reward(
            feats_all=feats_np,
            sel_idx=sel_idx,
            anime_attrs=anime_attrs,
            vlm_scores=vlm_scores,
            current_epoch=current_epoch
        )
        
        total_reward = float(np.clip(rewards["total"], -20.0, 20.0))
        reward_rec = float(np.clip(-rewards["constraint_penalty"], -10.0, 10.0))
        reward_anime = float(np.clip(rewards["anime"] + rewards.get("vlm", 0.0), -10.0, 10.0))

        # Advantages - use Python floats to ensure float32 tensors
        advantage_rec = (torch.tensor([reward_rec], device=self.device, dtype=torch.float32) - values_rec.mean().detach()).detach()
        advantage_anime = (torch.tensor([reward_anime], device=self.device, dtype=torch.float32) - values_anime.mean().detach()).detach()
        
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
            
            loss_rec = self.compute_per_task_loss(old_log_prob, new_log_prob, advantage_rec, values_rec_new.mean(), torch.tensor([reward_rec], device=self.device, dtype=torch.float32), probs_rec_new)
            loss_anime = self.compute_per_task_loss(old_log_prob, new_log_prob, advantage_anime, values_anime_new.mean(), torch.tensor([reward_anime], device=self.device, dtype=torch.float32), probs_anime_new)
            
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
        
        train_info = {
            "loss": total_loss / self.n_ppo_epochs,
            "loss_distill": distill_loss,
            "reward_total": total_reward,
            "reward_anime": rewards["anime"],
            "reward_vlm": rewards.get("vlm", 0.0),
            "constraint_penalty": rewards["constraint_penalty"],
        }
        train_info.update(self.model.get_gating_stats())
        train_info.update(self.reward_system.get_lagrangian_state())
        for k, v in components.items():
            if isinstance(v, (int, float, bool)): train_info[f"component_{k}"] = float(v)
        
        return train_info


def log_v10_metrics(writer: SummaryWriter, info: Dict, epoch: int):
    for k, v in info.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(f"train/{k}", v, epoch)

def main():
    ap = argparse.ArgumentParser(description="V10 VLM-Guided RL Training")
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)  # Lower LR for stability
    ap.add_argument("--feat_dim", type=int, default=512)
    ap.add_argument("--budget_ratio", type=float, default=0.06)
    ap.add_argument("--Bmin", type=int, default=3)
    ap.add_argument("--Bmax", type=int, default=15)
    ap.add_argument("--use_pcgrad", type=int, default=1)
    ap.add_argument("--use_dpp", type=int, default=1)
    ap.add_argument("--anime_scale", type=float, default=5.0)
    ap.add_argument("--vlm_scale", type=float, default=4.0)
    ap.add_argument("--vlm_decay_epochs", type=int, default=30)
    ap.add_argument("--distill_lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--val_videos_dir", type=str, default=None)
    ap.add_argument("--val_output_dir", type=str, default=None)
    ap.add_argument("--validate_every", type=int, default=5)
    ap.add_argument("--eval_device", type=str, default="cuda")
    
    args = ap.parse_args()
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, "logs"))
    
    scene_dirs = build_epoch_index(args.dataset_root)
    full_feat_dim = args.feat_dim + 6 # Using anime_attrs as V9 does
    
    model = create_dsn_v8(feat_dim=full_feat_dim, use_pcgrad=bool(args.use_pcgrad)).to(args.device)
    
    reward_system = PremiumRewardV10(
        constraint_config=ConstraintConfig(),
        dpp_config=DPPConfig(),
        vlm_scale=args.vlm_scale,
        vlm_decay_epochs=args.vlm_decay_epochs,
        anime_scale=args.anime_scale,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pcgrad_opt = create_pcgrad_optimizer(model, lr=args.lr) if args.use_pcgrad else None
    trainer = PPOTrainerV10(model, reward_system, GAEComputer(), optimizer, pcgrad_opt, device=args.device)

    for epoch in range(1, args.epochs + 1):
        epoch_info = []
        pbar = tqdm(scene_dirs, desc=f"Epoch {epoch}")
        for scene_dir in pbar:
            sample = load_scene_dir(scene_dir, load_frames=False, load_anime_attrs=True, load_vlm_scores=True)
            feats_base = sample.feats
            extra = [sample.anime_attrs] if sample.anime_attrs is not None else []
            feats_full = np.concatenate([feats_base] + extra, axis=1)
            
            feats_t = torch.from_numpy(feats_full).float().unsqueeze(0)
            clip_t = torch.from_numpy(feats_base).float().unsqueeze(0)
            
            budget = max(args.Bmin, min(args.Bmax, int(len(feats_base) * args.budget_ratio)))
            
            info = trainer.train_step(feats_t, clip_t, sample.vlm_scores, sample.anime_attrs, budget, epoch)
            epoch_info.append(info)
            pbar.set_postfix({"loss": f"{info['loss']:.3f}", "vlm_rew": f"{info['reward_vlm']:.2f}"})
            
        # Logging & Saving
        avg_info = {k: np.mean([x[k] for x in epoch_info if k in x]) for k in epoch_info[0].keys()}
        log_v10_metrics(writer, avg_info, epoch)
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "distiller_state_dict": reward_system.distiller.state_dict(),
            "epoch": epoch,
            "args": vars(args)
        }, os.path.join(args.save_dir, f"dsn_v10_ep{epoch}.pt"))
        
        print(f"\nEpoch {epoch}: Loss={avg_info['loss']:.4f}, VLM_Rew={avg_info['reward_vlm']:.2f}, Distill_Loss={avg_info['loss_distill']:.4f}")

    writer.close()
    print(f"Training Complete. Results in {args.save_dir}")

if __name__ == "__main__":
    main()
