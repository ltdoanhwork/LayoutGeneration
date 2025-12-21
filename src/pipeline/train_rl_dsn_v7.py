#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7 Multi-Task RL Training: Dual-Objective Optimization

Key improvements over V6:
1. Balanced optimization of BOTH RecErr AND Anime quality
2. Enhanced RecErr rewards with diversity and coverage
3. Curriculum learning: RecErr first, then add Anime
4. Comprehensive TensorBoard logging for all metrics
5. Auto-visualization after validation

Usage:
    python -m src.pipeline.train_rl_dsn_v7 \
        --dataset_root data/sakuga_dataset_100_samples \
        --save_dir runs/dsn_v7_dual_objective \
        --epochs 60 \
        --rec_err_scale 3.0 \
        --frechet_scale 2.0 \
        --anime_scale 2.5
"""

from __future__ import annotations
import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports - use correct module paths like V5
from src.datasets import build_epoch_index, load_scene_dir
from src.models.dsn_multitask import DSNMultiTask, create_dsn_multitask
from src.rl.premium_rewards_v5 import PremiumRewardV5, compute_quality_metrics_for_eval
from src.rl.rewards import reward_combo_v4
from src.rl.gae import GAEComputer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PPOTrainerV7:
    """PPO trainer for V7 dual-objective optimization"""
    
    def __init__(
        self,
        model: DSNMultiTask,
        reward_system: PremiumRewardV5,
        gae_computer: GAEComputer,
        optimizer: torch.optim.Optimizer,
        clip_range: float = 0.2,
        n_ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
    ):
        self.model = model
        self.reward_system = reward_system
        self.gae_computer = gae_computer
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.n_ppo_epochs = n_ppo_epochs
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
    
    def sample_action(
        self,
        probs: torch.Tensor,  # (1, T)
        budget: int,
    ) -> Tuple[List[int], torch.Tensor]:
        """Sample K frames from probability distribution"""
        probs_np = probs.squeeze(0).detach().cpu().numpy()
        T = len(probs_np)
        K = min(budget, T)
        
        # Sample without replacement
        probs_safe = np.clip(probs_np, 1e-8, 1.0)
        probs_safe = probs_safe / probs_safe.sum()
        
        try:
            sel_idx = np.random.choice(T, size=K, replace=False, p=probs_safe)
        except:
            sel_idx = np.random.choice(T, size=K, replace=False)
        
        sel_idx = sorted(sel_idx.tolist())
        
        # Compute log probability
        log_probs = torch.log(probs.squeeze(0)[sel_idx] + 1e-8)
        log_prob_sum = log_probs.sum()
        
        return sel_idx, log_prob_sum
    
    def compute_loss(
        self,
        old_log_prob: torch.Tensor,
        new_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss"""
        # Policy loss with clipping
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        # Total loss
        total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy
        
        info = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "ratio": float(ratio.item()),
        }
        
        return total_loss, info
    
    def train_step(
        self,
        features: torch.Tensor,  # (1, T, D)
        anime_attrs: Optional[np.ndarray],  # (T, 6)
        motion: Optional[np.ndarray],  # (T,)
        budget: int,
        current_epoch: int,
    ) -> Dict[str, float]:
        """Single training step for one video"""
        self.model.train()
        features = features.to(self.device)
        
        # Forward pass - get both task outputs (no grad for initial sampling)
        with torch.no_grad():
            task_outputs = self.model(features, return_all_tasks=True)
            probs_rec = task_outputs["rec"][0]  # (1, T)
            values_rec = task_outputs["rec"][1]  # (1, T)
            probs_anime = task_outputs["anime"][0]  # (1, T)
            values_anime = task_outputs["anime"][1]  # (1, T)
            
            # Merge policies
            alpha = 0.5
            merged_probs = alpha * probs_rec + (1 - alpha) * probs_anime
        
        # Sample action (no grad needed)
        sel_idx, old_log_prob = self.sample_action(merged_probs, budget)
        old_log_prob = old_log_prob.detach()  # Detach for PPO
        
        # Compute rewards using V5 reward system
        feats_np = features.squeeze(0).detach().cpu().numpy()
        rewards, components = self.reward_system.compute_reward(
            feats_all=feats_np,
            sel_idx=sel_idx,
            anime_attrs=anime_attrs,
            motion=motion,
            current_epoch=current_epoch,
        )
        
        # Total reward
        total_reward = rewards["total"]
        
        # Detach old values for advantage computation
        old_values_rec = values_rec.detach()
        old_values_anime = values_anime.detach()
        
        # GAE for both tasks
        reward_rec = torch.tensor([rewards["rec"]], device=self.device, dtype=torch.float32)
        reward_anime = torch.tensor([rewards["anime"]], device=self.device, dtype=torch.float32)
        
        # Simplified: single-step episode advantages
        advantage_rec = (reward_rec - old_values_rec.mean()).detach()
        advantage_anime = (reward_anime - old_values_anime.mean()).detach()
        
        returns_rec = reward_rec.detach()
        returns_anime = reward_anime.detach()
        
        # PPO update for both tasks
        total_loss = 0.0
        info_combined = {}
        
        for ppo_iter in range(self.n_ppo_epochs):
            self.optimizer.zero_grad()
            
            # Fresh forward pass (with grad)
            task_outputs = self.model(features, return_all_tasks=True)
            probs_rec_new = task_outputs["rec"][0]
            values_rec_new = task_outputs["rec"][1]
            probs_anime_new = task_outputs["anime"][0]
            values_anime_new = task_outputs["anime"][1]
            
            merged_probs_new = alpha * probs_rec_new + (1 - alpha) * probs_anime_new
            
            # New log prob
            new_log_prob = torch.log(merged_probs_new.squeeze(0)[sel_idx] + 1e-8).sum()
            
            # Merged values and advantages (advantages are detached)
            merged_values = alpha * values_rec_new.mean() + (1 - alpha) * values_anime_new.mean()
            merged_advantage = alpha * advantage_rec + (1 - alpha) * advantage_anime
            merged_returns = alpha * returns_rec + (1 - alpha) * returns_anime
            
            # Compute loss
            loss, info = self.compute_loss(
                old_log_prob=old_log_prob,
                new_log_prob=new_log_prob,
                advantages=merged_advantage,
                values=merged_values.unsqueeze(0),
                returns=merged_returns.unsqueeze(0),
                probs=merged_probs_new,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            info_combined = info
        
        # Compile training info
        train_info = {
            "loss": total_loss / self.n_ppo_epochs,
            "reward_total": total_reward,
            "reward_rec": rewards["rec"],
            "reward_anime": rewards["anime"],
            "n_selected": len(sel_idx),
            **info_combined,
        }
        
        # Add component details
        for k, v in components.items():
            train_info[f"component_{k}"] = v
        
        return train_info


def run_validation(
    checkpoint_path: str,
    val_videos_dir: str,
    val_output_dir: str,
    epoch: int,
    args,
):
    """Run validation using batch_eval.py"""
    epoch_output_dir = os.path.join(val_output_dir, f"ep{epoch}")
    os.makedirs(epoch_output_dir, exist_ok=True)
    
    cmd = [
        "python", "-m", "eval.batch_eval",
        "--videos_dir", val_videos_dir,
        "--output_dir", epoch_output_dir,
        "--checkpoint", checkpoint_path,
        "--device", args.eval_device,
        "--backend", args.eval_backend,
        "--embedder", args.eval_embedder,
        "--enc_hidden", str(args.enc_hidden),
        "--lstm_hidden", str(args.lstm_hidden),
        "--budget_ratio", str(args.budget_ratio),
        "--Bmin", str(args.Bmin),
        "--Bmax", str(args.Bmax),
        "--use_anime_attrs", str(args.use_anime_attrs),
        "--anime_attrs_dim", str(args.anime_attrs_dim),
    ]
    
    if args.eval_with_baselines:
        cmd.append("--with_baselines")
    
    print(f"\n🔍 Running validation for epoch {epoch}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"⚠️  Validation failed: {result.stderr[:500]}")
        return None
    
    # Load and return summary
    summary_path = os.path.join(epoch_output_dir, "summary_results.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser(description="V7 Dual-Objective RL Training")
    
    # Data
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    
    # Model
    ap.add_argument("--feat_dim", type=int, default=512)
    ap.add_argument("--enc_hidden", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--use_anime_attrs", type=int, default=1)
    ap.add_argument("--anime_attrs_dim", type=int, default=6)
    ap.add_argument("--use_raft_motion", type=int, default=1)
    ap.add_argument("--motion_dim", type=int, default=128)
    
    # V7 Reward System
    ap.add_argument("--rec_err_scale", type=float, default=3.0,
                    help="Scale for RecErr rewards")
    ap.add_argument("--frechet_scale", type=float, default=2.0,
                    help="Scale for Frechet distance rewards")
    ap.add_argument("--diversity_weight", type=float, default=1.0)
    ap.add_argument("--coverage_weight", type=float, default=1.0)
    ap.add_argument("--anime_scale", type=float, default=2.5,
                    help="Scale for anime quality rewards")
    ap.add_argument("--top_k_ratio", type=float, default=0.1)
    ap.add_argument("--use_curriculum", type=int, default=1)
    
    # PPO
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--n_ppo_epochs", type=int, default=4)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    
    # Budget
    ap.add_argument("--budget_ratio", type=float, default=0.06)
    ap.add_argument("--Bmin", type=int, default=3)
    ap.add_argument("--Bmax", type=int, default=15)
    
    # Device
    ap.add_argument("--device", type=str, default="cuda")
    
    # Validation
    ap.add_argument("--val_videos_dir", type=str, default=None)
    ap.add_argument("--val_output_dir", type=str, default=None)
    ap.add_argument("--validate_every", type=int, default=5)
    ap.add_argument("--eval_backend", type=str, default="transnetv2")
    ap.add_argument("--eval_embedder", type=str, default="clip_vitb32")
    ap.add_argument("--eval_device", type=str, default="cuda")
    ap.add_argument("--eval_with_baselines", action="store_true")
    
    # Logging
    ap.add_argument("--log_dir", type=str, default=None)
    ap.add_argument("--save_visualizations", type=int, default=1)
    
    args = ap.parse_args()
    
    # Setup
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    log_dir = args.log_dir or os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    if args.val_output_dir:
        os.makedirs(args.val_output_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(os.path.join(args.save_dir, "config_v7.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("=" * 60)
    print("V7 Dual-Objective Training")
    print("=" * 60)
    print(f"RecErr scale: {args.rec_err_scale}")
    print(f"Frechet scale: {args.frechet_scale}")
    print(f"Anime scale: {args.anime_scale}")
    print(f"Curriculum: {bool(args.use_curriculum)}")
    print("=" * 60)
    
    # Dataset - use build_epoch_index like V5
    scene_dirs = build_epoch_index(args.dataset_root)
    print(f"Dataset: {len(scene_dirs)} scenes")
    
    # Model
    feat_dim = args.feat_dim
    if args.use_anime_attrs:
        feat_dim += args.anime_attrs_dim
    if args.use_raft_motion:
        feat_dim += args.motion_dim
    
    model = create_dsn_multitask(
        feat_dim=feat_dim,
        hidden_dim=args.enc_hidden,
        lstm_hidden=args.lstm_hidden,
    ).to(args.device)
    
    print(f"Model: DSNMultiTask (feat_dim={feat_dim})")
    
    # V7 Reward System
    reward_system = PremiumRewardV5(
        rec_err_scale=args.rec_err_scale,
        frechet_scale=args.frechet_scale,
        diversity_weight=args.diversity_weight,
        coverage_weight=args.coverage_weight,
        anime_scale=args.anime_scale,
        top_k_ratio=args.top_k_ratio,
        use_curriculum=bool(args.use_curriculum),
        total_epochs=args.epochs,
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # GAE Computer
    gae_computer = GAEComputer(gamma=0.99, lam=0.95)
    
    # Trainer
    trainer = PPOTrainerV7(
        model=model,
        reward_system=reward_system,
        gae_computer=gae_computer,
        optimizer=optimizer,
        clip_range=args.clip_range,
        n_ppo_epochs=args.n_ppo_epochs,
        entropy_coef=args.entropy_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )
    
    # Training loop
    global_step = 0
    best_rec_err = float("inf")
    best_anime_improvement = float("-inf")
    
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        epoch_rewards_rec = []
        epoch_rewards_anime = []
        epoch_rewards_total = []
        
        pbar = tqdm(scene_dirs, desc=f"Epoch {epoch}/{args.epochs}")
        
        for scene_dir in pbar:
            # Load scene data using proper API
            sample = load_scene_dir(
                scene_dir,
                load_frames=False,  # Don't need frames for training
                load_motion=bool(args.use_raft_motion),
                load_anime_attrs=bool(args.use_anime_attrs),
            )
            
            # Get features
            feats = sample.feats  # (T, D_base)
            
            # Concatenate extra features
            extra_feats = []
            if args.use_anime_attrs and sample.anime_attrs is not None:
                extra_feats.append(sample.anime_attrs)
            if args.use_raft_motion and sample.motion is not None:
                motion = sample.motion
                if motion.ndim == 1:
                    motion = np.repeat(motion[:, None], args.motion_dim, axis=1)
                extra_feats.append(motion[:len(feats)])
            
            if extra_feats:
                feats = np.concatenate([feats] + extra_feats, axis=1)
            
            feats_tensor = torch.from_numpy(feats).float().unsqueeze(0)
            
            # Get anime attrs and motion for rewards
            anime_attrs = sample.anime_attrs
            motion = sample.motion
            if motion is not None and motion.ndim > 1:
                motion = motion.mean(axis=1)  # Reduce to (T,)
            
            # Compute budget
            T = len(feats)
            budget = max(args.Bmin, min(args.Bmax, int(T * args.budget_ratio)))
            
            # Train step
            train_info = trainer.train_step(
                features=feats_tensor,
                anime_attrs=anime_attrs,
                motion=motion,
                budget=budget,
                current_epoch=epoch,
            )
            
            epoch_losses.append(train_info["loss"])
            epoch_rewards_rec.append(train_info["reward_rec"])
            epoch_rewards_anime.append(train_info["reward_anime"])
            epoch_rewards_total.append(train_info["reward_total"])
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{np.mean(epoch_losses):.4f}",
                "R_rec": f"{np.mean(epoch_rewards_rec):.3f}",
                "R_anime": f"{np.mean(epoch_rewards_anime):.3f}",
            })
            
            global_step += 1
        
        # Epoch logging
        writer.add_scalar("train/loss", np.mean(epoch_losses), epoch)
        writer.add_scalar("train/reward_rec", np.mean(epoch_rewards_rec), epoch)
        writer.add_scalar("train/reward_anime", np.mean(epoch_rewards_anime), epoch)
        writer.add_scalar("train/reward_total", np.mean(epoch_rewards_total), epoch)
        
        # Log tracker summary
        tracker_summary = reward_system.get_tracker_summary()
        for k, v in tracker_summary.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        
        # Log curriculum weights
        weights = reward_system.get_curriculum_weights(epoch)
        writer.add_scalar("train/weight_rec", weights["rec"], epoch)
        writer.add_scalar("train/weight_anime", weights["anime"], epoch)
        
        print(f"\nEpoch {epoch}: Loss={np.mean(epoch_losses):.4f}, "
              f"R_rec={np.mean(epoch_rewards_rec):.3f}, "
              f"R_anime={np.mean(epoch_rewards_anime):.3f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"dsn_v7_ep{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }, checkpoint_path)
        
        # Validation
        if args.val_videos_dir and epoch % args.validate_every == 0:
            val_summary = run_validation(
                checkpoint_path=checkpoint_path,
                val_videos_dir=args.val_videos_dir,
                val_output_dir=args.val_output_dir,
                epoch=epoch,
                args=args,
            )
            
            if val_summary:
                agg = val_summary.get("aggregate_metrics", {})
                anime_quality = val_summary.get("anime_quality_metrics", {})
                
                # Log validation metrics
                for k, v in agg.items():
                    if v is not None:
                        writer.add_scalar(f"val/{k}", v, epoch)
                
                for k, v in anime_quality.items():
                    if v is not None:
                        writer.add_scalar(f"val/{k}", v, epoch)
                
                # Track best
                rec_err = agg.get("RecErr_mean")
                quality_imp = anime_quality.get("Quality_Improvement_mean")
                
                if rec_err is not None and rec_err < best_rec_err:
                    best_rec_err = rec_err
                    torch.save(model.state_dict(), 
                              os.path.join(args.save_dir, "best_rec_err.pt"))
                    print(f"✅ New best RecErr: {rec_err:.4f}")
                
                if quality_imp is not None and quality_imp > best_anime_improvement:
                    best_anime_improvement = quality_imp
                    torch.save(model.state_dict(),
                              os.path.join(args.save_dir, "best_anime.pt"))
                    print(f"✅ New best Anime Improvement: {quality_imp:.4f}")
                
                print(f"📊 Val: RecErr={rec_err}, QualityImp={quality_imp}")
            
            # Generate visualizations
            if args.save_visualizations:
                viz_cmd = [
                    "python", "-m", "eval.visualize_validation",
                    "--val_output_dir", args.val_output_dir,
                    "--output_dir", os.path.join(args.save_dir, "plots"),
                    "--save_images",
                ]
                subprocess.run(viz_cmd, capture_output=True)
    
    writer.close()
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best RecErr: {best_rec_err:.4f}")
    print(f"Best Anime Improvement: {best_anime_improvement:.4f}")
    print(f"Checkpoints: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
