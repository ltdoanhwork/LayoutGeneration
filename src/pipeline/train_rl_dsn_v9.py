#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V9 Quality-Focused RL Training

Key improvements over V8:
1. INCREASED quality weighting (anime_scale: 3.0 → 5.0)
2. NEW: Percentile-based reward for selecting high-percentile frames
3. REMOVED: Motion features (simpler, faster)
4. REDUCED: Constraint penalty (focus on quality)

Goal: Maximize Mean Percentile Rank and Top-K Coverage

Usage:
    python -m src.pipeline.train_rl_dsn_v9 \\
        --dataset_root data/sakuga_dataset_100_samples \\
        --save_dir runs/dsn_v9_quality_focused \\
        --epochs 60 \\
        --use_pcgrad 1 \\
        --use_dpp 1
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
# V9: Use premium_rewards_v9 instead of v8
from src.rl.premium_rewards_v9 import (
    PremiumRewardV9, 
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


class PPOTrainerV9:
    """
    V9 PPO Trainer:
    - Quality-focused optimization
    - NO motion features (simpler and faster)
    - PCGrad for conflict-free gradients
    - DPP selection support
    """
    
    def __init__(
        self,
        model: DSNMultiTaskV8,
        reward_system: PremiumRewardV9,
        gae_computer: GAEComputer,
        optimizer: torch.optim.Optimizer,
        pcgrad_optimizer: Optional[PCGradOptimizer] = None,
        clip_range: float = 0.2,
        n_ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
        use_pcgrad: bool = True,
        use_dpp: bool = True,
        use_tts: bool = False,  # Test-time scaling at inference only
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
        self.use_tts = use_tts
    
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
    
    def sample_with_dpp(
        self,
        probs: torch.Tensor,
        features: torch.Tensor,
        anime_attrs: Optional[np.ndarray],
        budget: int,
    ) -> Tuple[List[int], Dict]:
        """Sample using DPP two-stage selection"""
        feats_np = features.squeeze(0).cpu().numpy()
        
        # Use DPP selection from reward system
        sel_idx, info = self.reward_system.select_with_dpp(
            feats_np, anime_attrs, budget
        )
        
        return sel_idx, info
    
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
        # Policy loss with clipping
        ratio = torch.exp(new_log_prob - old_log_prob)
        # Numerical stability: clamp ratio to avoid huge gradients
        ratio = torch.clamp(ratio, 0.0, 10.0) 
        
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        
        # Value loss - FIX: ensure same shape to avoid broadcasting error
        # Use view(-1) to make them both flattened tensors
        value_loss = F.mse_loss(value.view(-1), return_target.view(-1))
        
        # Entropy bonus
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        # Total loss
        total_loss = policy_loss.mean() + self.vf_coef * value_loss - self.entropy_coef * entropy
        
        return total_loss
    
    def train_step(
        self,
        features: torch.Tensor,  # (1, T, D)
        anime_attrs: Optional[np.ndarray],
        motion: Optional[np.ndarray],
        budget: int,
        current_epoch: int,
    ) -> Dict[str, float]:
        """Single training step for one video"""
        self.model.train()
        features = features.to(self.device)
        
        # Forward pass with gating
        with torch.no_grad():
            merged_probs, merged_values, alpha = self.model(
                features, return_gating=True
            )
            
            # Get task-specific outputs for Lagrangian
            task_outputs = self.model(features, return_all_tasks=True)
            probs_rec, values_rec = task_outputs["rec"]
            probs_anime, values_anime = task_outputs["anime"]
        
        # Sample action
        if self.use_dpp and anime_attrs is not None:
            # DPP selection (better diversity)
            sel_idx, dpp_info = self.sample_with_dpp(
                merged_probs, features, anime_attrs, budget
            )
            # Compute log prob for selected indices
            old_log_prob = torch.log(merged_probs.squeeze(0)[sel_idx] + 1e-8).sum()
        else:
            sel_idx, old_log_prob = self.sample_action(merged_probs, budget)
            dpp_info = {}
        
        old_log_prob = old_log_prob.detach()
        
        # Compute rewards using V8 reward system
        feats_np = features.squeeze(0).detach().cpu().numpy()
        rewards, components = self.reward_system.compute_reward(
            feats_all=feats_np,
            sel_idx=sel_idx,
            anime_attrs=anime_attrs,
            motion=motion,
            current_epoch=current_epoch,
            update_lagrangian=True,  # Update lambda multipliers
        )
        
        total_reward = rewards["total"]
        
        # V9: CLIP REWARDS to prevent explosion
        total_reward = np.clip(total_reward, -20.0, 20.0)
        
        # Separate rewards for tasks
        reward_rec = -rewards["constraint_penalty"]
        reward_anime = rewards["anime"] + rewards.get("percentile", 0) # V9: both count towards anime task
        
        # Clip individual rewards too
        reward_rec = np.clip(reward_rec, -10.0, 10.0)
        reward_anime = np.clip(reward_anime, -10.0, 10.0)
        
        # Compute advantages (simplified single-step)
        reward_rec_t = torch.tensor([reward_rec], device=self.device, dtype=torch.float32)
        reward_anime_t = torch.tensor([reward_anime], device=self.device, dtype=torch.float32)
        
        # Advantage normalization/clamping
        advantage_rec = (reward_rec_t - values_rec.mean().detach()).detach()
        advantage_anime = (reward_anime_t - values_anime.mean().detach()).detach()
        
        # Clamp advantages
        advantage_rec = torch.clamp(advantage_rec, -5.0, 5.0)
        advantage_anime = torch.clamp(advantage_anime, -5.0, 5.0)
        
        returns_rec = reward_rec_t.detach()
        returns_anime = reward_anime_t.detach()
        
        # PPO update
        total_loss = 0.0
        info_combined = {}
        
        for ppo_iter in range(self.n_ppo_epochs):
            if self.use_pcgrad:
                self.pcgrad_optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
            
            # Fresh forward pass
            merged_probs_new, merged_values_new, alpha_new = self.model(
                features, return_gating=True
            )
            task_outputs_new = self.model(features, return_all_tasks=True)
            probs_rec_new, values_rec_new = task_outputs_new["rec"]
            probs_anime_new, values_anime_new = task_outputs_new["anime"]
            
            # New log probs
            new_log_prob = torch.log(merged_probs_new.squeeze(0)[sel_idx] + 1e-8).sum()
            
            if self.use_pcgrad:
                # Compute per-task losses
                loss_rec = self.compute_per_task_loss(
                    old_log_prob, new_log_prob, advantage_rec,
                    values_rec_new.mean(), returns_rec, probs_rec_new
                )
                loss_anime = self.compute_per_task_loss(
                    old_log_prob, new_log_prob, advantage_anime,
                    values_anime_new.mean(), returns_anime, probs_anime_new
                )
                
                # PCGrad backward
                self.pcgrad_optimizer.backward([loss_rec, loss_anime], self.model)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.pcgrad_optimizer.step()
                
                iter_loss = (loss_rec.item() + loss_anime.item()) / 2
            else:
                # Standard merged loss
                merged_advantage = alpha_new.mean() * advantage_rec + (1 - alpha_new.mean()) * advantage_anime
                merged_returns = alpha_new.mean() * returns_rec + (1 - alpha_new.mean()) * returns_anime
                
                loss = self.compute_per_task_loss(
                    old_log_prob, new_log_prob, merged_advantage,
                    merged_values_new.mean(), merged_returns, merged_probs_new
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                iter_loss = loss.item()
            
            total_loss += iter_loss
        
        # Compile training info
        train_info = {
            "loss": total_loss / self.n_ppo_epochs,
            "reward_total": total_reward,
            "reward_anime": rewards["anime"],
            "constraint_penalty": rewards["constraint_penalty"],
            "n_selected": len(sel_idx),
        }
        
        # Add gating stats
        gating_stats = self.model.get_gating_stats()
        train_info.update(gating_stats)
        
        # Add Lagrangian state
        lagrangian_state = self.reward_system.get_lagrangian_state()
        train_info.update(lagrangian_state)
        
        # Add component details
        for k, v in components.items():
            if isinstance(v, (int, float, bool)):
                train_info[f"component_{k}"] = float(v)
        
        # Add DPP info if available
        for k, v in dpp_info.items():
            train_info[f"dpp_{k}"] = v
        
        return train_info


def run_validation(
    checkpoint_path: str,
    val_videos_dir: str,
    val_output_dir: str,
    epoch: int,
    args,
) -> Optional[Dict]:
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
        "--use_raft_motion", str(args.use_raft_motion),
        "--motion_dim", str(args.motion_dim),
        "--feat_dim", str(args.feat_dim),  # Explicitly pass feat_dim
        "--min_scene_len", str(args.min_scene_len),
    ]
    
    if args.eval_with_baselines:
        cmd.append("--with_baselines")
    
    if args.eval_max_videos:
        cmd.extend(["--max_videos", str(args.eval_max_videos)])
    
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


def log_v8_metrics(writer: SummaryWriter, train_info: Dict, epoch: int, prefix: str = "train"):
    """Log V8-specific metrics to TensorBoard"""
    
    # Core metrics
    core_metrics = ["loss", "reward_total", "reward_anime", "constraint_penalty"]
    for m in core_metrics:
        if m in train_info:
            writer.add_scalar(f"{prefix}/{m}", train_info[m], epoch)
    
    # Lagrangian multipliers
    lambda_metrics = ["lambda_rec", "lambda_cov", "lambda_div"]
    for m in lambda_metrics:
        if m in train_info:
            writer.add_scalar(f"{prefix}/lagrangian/{m}", train_info[m], epoch)
    
    # Gating statistics
    gating_metrics = ["gating_mean", "gating_std", "gating_min", "gating_max", "gating_rec_dominant"]
    for m in gating_metrics:
        if m in train_info:
            writer.add_scalar(f"{prefix}/gating/{m}", train_info[m], epoch)
    
    # Constraint values
    constraint_metrics = ["rec_err", "coverage_gap", "diversity"]
    for m in constraint_metrics:
        key = f"component_{m}"
        if key in train_info:
            writer.add_scalar(f"{prefix}/constraints/{m}", train_info[key], epoch)
    
    # DPP metrics
    dpp_metrics = ["dpp_diversity", "dpp_selected_mean_quality", "dpp_n_candidates"]
    for m in dpp_metrics:
        if m in train_info:
            writer.add_scalar(f"{prefix}/{m}", train_info[m], epoch)
    
    # Quantile metrics
    quantile_metrics = ["mean_percentile", "top_1_percent_ratio", "top_10_percent_ratio"]
    for m in quantile_metrics:
        key = f"component_anime_{m}"
        if key in train_info:
            writer.add_scalar(f"{prefix}/quantile/{m}", train_info[key], epoch)


def main():
    ap = argparse.ArgumentParser(description="V8 Constrained MORL Training")
    
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
    ap.add_argument("--use_raft_motion", type=int, default=0)  # V9: DISABLED by default
    ap.add_argument("--motion_dim", type=int, default=128)
    ap.add_argument("--gating_hidden", type=int, default=64)
    
    # V8 Features
    ap.add_argument("--use_pcgrad", type=int, default=1, help="Use PCGrad gradient surgery")
    ap.add_argument("--use_dpp", type=int, default=1, help="Use DPP for selection")
    ap.add_argument("--use_tts", type=int, default=1, help="Use test-time scaling at inference")
    
    # Constraint thresholds
    ap.add_argument("--rec_err_threshold", type=float, default=0.35)
    ap.add_argument("--coverage_threshold", type=float, default=0.3)
    ap.add_argument("--diversity_threshold", type=float, default=0.15)  # V9.1: relaxed (0.25 -> 0.15)
    ap.add_argument("--lambda_lr", type=float, default=0.01)
    
    # DPP config
    ap.add_argument("--dpp_beta", type=float, default=1.0)
    ap.add_argument("--dpp_candidate_ratio", type=float, default=0.3)
    
    # Test-time scaling config
    ap.add_argument("--tts_n_samples", type=int, default=8)
    ap.add_argument("--tts_temperature", type=float, default=1.2)
    
    # Reward scales - V9: INCREASED quality focus
    ap.add_argument("--anime_scale", type=float, default=5.0)      # V8: 3.0 → V9: 5.0
    ap.add_argument("--quantile_scale", type=float, default=3.0)   # V8: 2.0 → V9: 3.0
    ap.add_argument("--percentile_scale", type=float, default=2.0) # NEW in V9
    ap.add_argument("--use_curriculum", type=int, default=1)
    
    # PPO
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--n_ppo_epochs", type=int, default=4)
    ap.add_argument("--entropy_coef", type=float, default=0.08)  # V9.2: further increased for exploration
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
    ap.add_argument("--eval_device", type=str, default="cpu")  # V9.1: CPU for stability
    ap.add_argument("--eval_with_baselines", action="store_true")
    ap.add_argument("--eval_max_videos", type=int, default=None)
    ap.add_argument("--min_scene_len", type=int, default=48)
    
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
    config["timestamp"] = datetime.now().isoformat()
    config["version"] = "V9"
    with open(os.path.join(args.save_dir, "config_v9.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("=" * 70)
    print("V9 Quality-Focused RL Training")
    print("=" * 70)
    print(f"Features: PCGrad={bool(args.use_pcgrad)}, DPP={bool(args.use_dpp)}, Motion={bool(args.use_raft_motion)}")
    print(f"Reward scales: anime={args.anime_scale}, quantile={args.quantile_scale}, percentile={args.percentile_scale}")
    print(f"Goal: Maximize Mean Percentile Rank and Top-K Coverage")
    print("=" * 70)
    
    # Dataset
    scene_dirs = build_epoch_index(args.dataset_root)
    print(f"Dataset: {len(scene_dirs)} scenes")
    
    # Model
    feat_dim = args.feat_dim
    if args.use_anime_attrs:
        feat_dim += args.anime_attrs_dim
    if args.use_raft_motion:
        feat_dim += args.motion_dim
    
    model = create_dsn_v8(
        feat_dim=feat_dim,
        hidden_dim=args.enc_hidden,
        lstm_hidden=args.lstm_hidden,
        use_pcgrad=bool(args.use_pcgrad),
        gating_hidden=args.gating_hidden,
    ).to(args.device)
    
    print(f"Model: DSNMultiTaskV8 (feat_dim={feat_dim})")
    
    # V9 Reward System (with percentile-based reward)
    reward_system = PremiumRewardV9(
        constraint_config=ConstraintConfig(
            rec_err_threshold=args.rec_err_threshold,
            coverage_threshold=args.coverage_threshold,
            diversity_threshold=args.diversity_threshold,
            lambda_lr=args.lambda_lr,
        ),
        dpp_config=DPPConfig(
            beta=args.dpp_beta,
            candidate_ratio=args.dpp_candidate_ratio,
        ),
        tts_config=TestTimeScalingConfig(
            n_samples=args.tts_n_samples,
            temperature=args.tts_temperature,
        ),
        anime_scale=args.anime_scale,
        quantile_scale=args.quantile_scale,
        percentile_scale=args.percentile_scale,  # NEW in V9
        use_curriculum=bool(args.use_curriculum),
        total_epochs=args.epochs,
    )
    
    # Optimizers
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    if args.use_pcgrad:
        pcgrad_optimizer = create_pcgrad_optimizer(model, lr=args.lr)
    else:
        pcgrad_optimizer = None
    
    # GAE Computer
    gae_computer = GAEComputer(gamma=0.99, lam=0.95)
    
    # Trainer - V9
    trainer = PPOTrainerV9(
        model=model,
        reward_system=reward_system,
        gae_computer=gae_computer,
        optimizer=base_optimizer,
        pcgrad_optimizer=pcgrad_optimizer,
        clip_range=args.clip_range,
        n_ppo_epochs=args.n_ppo_epochs,
        entropy_coef=args.entropy_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        use_pcgrad=bool(args.use_pcgrad),
        use_dpp=bool(args.use_dpp),
        use_tts=bool(args.use_tts),
    )
    
    # Training loop
    global_step = 0
    best_anime_score = float("-inf")
    best_constraint_satisfaction = 0.0
    
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = {
            "loss": [],
            "reward_total": [],
            "reward_anime": [],
            "constraint_penalty": [],
            "gating_mean": [],
            "lambda_rec": [],
            "lambda_cov": [],
            "lambda_div": [],
        }
        
        pbar = tqdm(scene_dirs, desc=f"Epoch {epoch}/{args.epochs}")
        
        for scene_dir in pbar:
            # Load scene data
            sample = load_scene_dir(
                scene_dir,
                load_frames=False,
                load_motion=bool(args.use_raft_motion),
                load_anime_attrs=bool(args.use_anime_attrs),
            )
            
            # Get features
            feats = sample.feats
            
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
                motion = motion.mean(axis=1)
            
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
            
            # Collect metrics
            for key in epoch_metrics.keys():
                if key in train_info:
                    epoch_metrics[key].append(train_info[key])
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{np.mean(epoch_metrics['loss']):.4f}",
                "anime": f"{np.mean(epoch_metrics['reward_anime']):.3f}",
                "α": f"{np.mean(epoch_metrics['gating_mean']):.2f}" if epoch_metrics['gating_mean'] else "N/A",
            })
            
            global_step += 1
        
        # Epoch logging
        avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
        
        # Log to TensorBoard
        for k, v in avg_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        
        # Log tracker summary
        tracker_summary = reward_system.tracker.get_summary()
        for k, v in tracker_summary.items():
            writer.add_scalar(f"train/tracker/{k}", v, epoch)
        
        print(f"\nEpoch {epoch}: Loss={avg_metrics['loss']:.4f}, "
              f"Anime={avg_metrics['reward_anime']:.3f}, "
              f"Penalty={avg_metrics['constraint_penalty']:.3f}, "
              f"α_mean={avg_metrics['gating_mean']:.3f}")
        print(f"  λ: rec={avg_metrics['lambda_rec']:.3f}, "
              f"cov={avg_metrics['lambda_cov']:.3f}, "
              f"div={avg_metrics['lambda_div']:.3f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"dsn_v9_ep{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": base_optimizer.state_dict(),
            "lagrangian_state": reward_system.lagrangian.state_dict(),
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
                        writer.add_scalar(f"val/anime/{k}", v, epoch)
                
                # Track best
                quality_imp = anime_quality.get("Quality_Improvement_mean", 0)
                rec_err = agg.get("RecErr_mean", 1.0)
                
                # Constraint satisfaction rate
                constraint_sat = 1.0 if rec_err <= args.rec_err_threshold else 0.0
                
                if quality_imp > best_anime_score:
                    best_anime_score = quality_imp
                    torch.save(model.state_dict(), 
                              os.path.join(args.save_dir, "best_anime.pt"))
                    print(f"✅ New best Anime Score: {quality_imp:.4f}")
                
                if constraint_sat > best_constraint_satisfaction or (
                    constraint_sat == best_constraint_satisfaction and quality_imp > best_anime_score
                ):
                    best_constraint_satisfaction = constraint_sat
                    torch.save(model.state_dict(),
                              os.path.join(args.save_dir, "best_constrained.pt"))
                    print(f"✅ New best constrained model (RecErr={rec_err:.4f}, Anime={quality_imp:.4f})")
                
                print(f"📊 Val: RecErr={rec_err:.4f}, Quality={quality_imp:.4f}")
            
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
    print("\n" + "=" * 70)
    print("V9 Training Complete!")
    print(f"Best Anime Score: {best_anime_score:.4f}")
    print(f"Best Constraint Satisfaction: {best_constraint_satisfaction:.1%}")
    print(f"Checkpoints: {args.save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
