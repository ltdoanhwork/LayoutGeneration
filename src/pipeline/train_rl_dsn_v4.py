#!/usr/bin/env python3
"""
V4 PPO Training Script for Anime DSN with Actor-Critic.

This script implements the Version 4 enhancement of the anime video summarization
training pipeline with state-of-the-art RL techniques:

Key Improvements over V3:
- PPO (Proximal Policy Optimization) instead of REINFORCE
- GAE (Generalized Advantage Estimation) for low-variance advantage estimates
- Actor-Critic architecture with learned value baseline
- Reward ensemble for stable CLIP-IQA targets
- Multiple PPO epochs per batch for sample efficiency

Usage:
    bash scripts/rl/train_dsn_v4_anime_ppo.sh

Author: V4 Enhancement
Date: 2025-12-06
"""

from __future__ import annotations
import os, json, argparse, subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import core components
from src.datasets import build_epoch_index, load_scene_dir
from src.models.dsn_advanced import DSNAdvanced, DSNConfig
from src.rl.rewards import reward_combo_v4 as reward_combo
from src.rl.premium_rewards_v3 import PremiumAnimeRewardV3
from src.rl.ppo_core import PPOCore, PPOConfig, compute_log_prob_bernoulli
from src.rl.gae import GAEComputer
from src.rl.experience_buffer import ExperienceBuffer
from src.rl.reward_ensemble import RewardEnsemble


# ---------------- utils ---------------- #
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def as_device(d: str) -> torch.device:
    if d.startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.device(d)
        except Exception:
            return torch.device("cuda")
    return torch.device("cpu")


def bernoulli_sample(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample from Bernoulli distribution."""
    m = torch.distributions.Bernoulli(probs)
    actions = m.sample()
    log_probs = m.log_prob(actions)
    entropy = m.entropy()
    return actions, log_probs, entropy


# --------------- PPO Update --------------- #
def ppo_update(
    model: DSNAdvanced,
    optimizer: optim.Optimizer,
    buffer: ExperienceBuffer,
    ppo: PPOCore,
    n_ppo_epochs: int = 4,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
    max_grad_norm: float = 0.5
) -> Dict[str, float]:
    """
    Perform PPO update using collected experiences.
    
    Args:
        model: DSN Actor-Critic model
        optimizer: Model optimizer
        buffer: Experience buffer with collected rollouts
        ppo: PPO core algorithm
        n_ppo_epochs: Number of PPO epochs per update
        batch_size: Mini-batch size
        device: Compute device
        max_grad_norm: Gradient clipping threshold
        
    Returns:
        Dict of training metrics
    """
    # Compute advantages if not already done
    buffer.compute_advantages()
    
    # Collect metrics
    all_actor_losses = []
    all_critic_losses = []
    all_entropy = []
    all_kl = []
    all_clip_frac = []
    
    for ppo_epoch in range(n_ppo_epochs):
        # Check for early stopping based on KL
        if ppo.should_stop_early():
            break
        
        for batch in buffer.get_batches(batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            
            # Process each scene in batch
            new_log_probs_list = []
            new_values_list = []
            new_probs_list = []
            
            for features in batch.features:
                features = features.unsqueeze(0).to(device)  # (1, T, D)
                
                # Forward pass with value
                probs, values = model(features, return_value=True)
                
                # Compute new log probs
                new_probs_list.append(probs.squeeze(0))
                new_values_list.append(values.squeeze(0))
            
            # Concatenate and compute log probs
            new_probs = torch.cat(new_probs_list)
            new_values = torch.cat(new_values_list)
            new_log_probs = compute_log_prob_bernoulli(new_probs, batch.actions)
            
            # PPO Actor Loss
            actor_loss, metrics = ppo.compute_actor_loss(
                batch.old_log_probs.to(device),
                new_log_probs,
                batch.advantages.to(device)
            )
            
            # Critic Loss (Value function)
            critic_loss = ppo.compute_critic_loss(
                new_values,
                batch.old_values.to(device),
                batch.returns.to(device)
            )
            
            # Entropy bonus
            entropy = ppo.compute_entropy_bonus(new_probs)
            
            # Total loss
            total_loss = (
                actor_loss +
                ppo.config.vf_coef * critic_loss -
                ppo.config.entropy_coef * entropy
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # Track metrics
            all_actor_losses.append(actor_loss.item())
            all_critic_losses.append(critic_loss.item())
            all_entropy.append(entropy.item())
            all_kl.append(metrics["approx_kl"])
            all_clip_frac.append(metrics["clip_fraction"])
    
    return {
        "actor_loss": np.mean(all_actor_losses),
        "critic_loss": np.mean(all_critic_losses),
        "entropy": np.mean(all_entropy),
        "approx_kl": np.mean(all_kl),
        "clip_fraction": np.mean(all_clip_frac),
        "ppo_epochs_completed": ppo_epoch + 1
    }


# --------------- Training --------------- #
def main():
    ap = argparse.ArgumentParser(description="V4 PPO Anime DSN Training")
    
    # Data
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)

    # Multi-Video Specific
    ap.add_argument("--multi_video", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--video_list", type=str, default=None)

    # Device
    ap.add_argument("--device", type=str, default="cuda")

    # Model
    ap.add_argument("--model_type", type=str, default="advanced")
    ap.add_argument("--feat_dim", type=int, default=512)
    ap.add_argument("--enc_hidden", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    
    # Advanced model hyperparameters
    ap.add_argument("--num_attn_heads", type=int, default=4)
    ap.add_argument("--num_attn_layers", type=int, default=2)
    ap.add_argument("--num_scales", type=int, default=3)
    ap.add_argument("--use_cache", type=int, default=1)
    ap.add_argument("--cache_size", type=int, default=1000)
    ap.add_argument("--pos_encoding_type", type=str, default="sinusoidal")
    ap.add_argument("--use_lstm_in_advanced", type=int, default=1)
    
    # RAFT Motion features
    ap.add_argument("--use_raft_motion", type=int, default=1)
    ap.add_argument("--motion_dim", type=int, default=128)
    ap.add_argument("--motion_fusion_type", type=str, default="cross_attention")

    # Anime-CLIP-IQA
    ap.add_argument("--use_anime_attrs", type=int, default=1)
    ap.add_argument("--anime_attrs_dim", type=int, default=6)
    
    # V4 PPO Configuration
    ap.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    ap.add_argument("--target_kl", type=float, default=0.01, help="Target KL for early stopping")
    ap.add_argument("--n_ppo_epochs", type=int, default=4, help="PPO update epochs")
    ap.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    ap.add_argument("--value_hidden_dim", type=int, default=256, help="Value head hidden dim")
    
    # V4 GAE Configuration
    ap.add_argument("--gamma", type=float, default=0.99, help="GAE discount factor")
    ap.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    
    # V4 Reward Ensemble
    ap.add_argument("--use_reward_ensemble", type=int, default=1, help="Use reward ensemble")
    ap.add_argument("--n_ensemble", type=int, default=3, help="Number of ensemble members")
    
    # Premium Reward Config (from V3)
    ap.add_argument("--w_anime_look", type=float, default=2.5)
    ap.add_argument("--w_anime_sakuga", type=float, default=2.5)
    ap.add_argument("--w_anime_story", type=float, default=1.2)
    ap.add_argument("--w_temporal", type=float, default=0.5)
    
    ap.add_argument("--percentile_threshold", type=float, default=0.75)
    ap.add_argument("--contrastive_margin", type=float, default=0.15)
    ap.add_argument("--hard_negative_margin", type=float, default=0.05)
    ap.add_argument("--use_curriculum", type=int, default=1)

    # RL
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--budget_ratio", type=float, default=0.06)
    ap.add_argument("--budget_penalty", type=float, default=0.05)
    ap.add_argument("--Bmin", type=int, default=3)
    ap.add_argument("--Bmax", type=int, default=15)

    # Standard Reward weights
    ap.add_argument("--w_div", type=float, default=0.5)
    ap.add_argument("--w_rep", type=float, default=0.5)
    ap.add_argument("--w_rec", type=float, default=0.0)  # Reconstruction reward
    ap.add_argument("--w_fd", type=float, default=0.0)   # Frechet distance reward
    ap.add_argument("--w_probsep", type=float, default=0.1)

    # Optim
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)

    # Logging & validation
    ap.add_argument("--log_dir", type=str, default="runs/dsn_anime_v4/logs")
    ap.add_argument("--val_videos_dir", type=str, default=None)
    ap.add_argument("--val_output_dir", type=str, default=None)
    ap.add_argument("--validate_every", type=int, default=2)
    ap.add_argument("--eval_embedder", type=str, default="clip_vitb32")
    ap.add_argument("--eval_backend", type=str, default="transnetv2")
    ap.add_argument("--eval_sample_stride", type=int, default=5)
    ap.add_argument("--eval_resize_w", type=int, default=320)
    ap.add_argument("--eval_resize_h", type=int, default=180)
    ap.add_argument("--eval_device", type=str, default=None)
    ap.add_argument("--eval_with_baselines", action="store_true")
    ap.add_argument("--eval_max_videos", type=int, default=None)

    args = ap.parse_args()

    device = as_device(args.device)
    eval_device = args.eval_device if args.eval_device else args.device

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(args.log_dir)))

    print("=" * 60)
    print("V4 PPO Training for Anime DSN")
    print("=" * 60)
    print(f"[V4] PPO Config: clip={args.clip_range}, kl_target={args.target_kl}, epochs={args.n_ppo_epochs}")
    print(f"[V4] GAE Config: gamma={args.gamma}, lambda={args.gae_lambda}")
    print(f"[V4] Reward Ensemble: {bool(args.use_reward_ensemble)} (n={args.n_ensemble})")

    # Initialize Premium Reward System
    premium_reward = PremiumAnimeRewardV3(
        percentile_threshold=args.percentile_threshold,
        contrastive_margin=args.contrastive_margin,
        hard_negative_margin=args.hard_negative_margin,
        temporal_weight=args.w_temporal,
        use_curriculum=bool(args.use_curriculum),
        use_quality_calibration=True,
        total_epochs=args.epochs
    )

    # Initialize Reward Ensemble (V4)
    reward_ensemble = None
    if args.use_reward_ensemble:
        reward_ensemble = RewardEnsemble(n_models=args.n_ensemble, device=str(device))
        print(f"[V4] Reward ensemble initialized with {args.n_ensemble} models")

    # Initialize PPO
    ppo_config = PPOConfig(
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        n_ppo_epochs=args.n_ppo_epochs,
        entropy_coef=args.entropy_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm
    )
    ppo = PPOCore(ppo_config)

    # Initialize GAE
    gae_computer = GAEComputer(gamma=args.gamma, lam=args.gae_lambda, normalize=True)

    # Model Setup with Actor-Critic
    print("\n[Model] Using Actor-Critic DSN with PPO")
    effective_feat_dim = args.feat_dim
    if args.use_anime_attrs:
        effective_feat_dim += args.anime_attrs_dim
        print(f"[Model] Adjusted feat_dim to {effective_feat_dim}")

    config = DSNConfig(
        feat_dim=effective_feat_dim,
        hidden_dim=args.enc_hidden,
        lstm_hidden=args.lstm_hidden,
        num_attn_heads=args.num_attn_heads,
        num_attn_layers=args.num_attn_layers,
        num_scales=args.num_scales,
        use_cache=bool(args.use_cache),
        cache_size=args.cache_size,
        pos_encoding_type=args.pos_encoding_type,
        use_lstm=bool(args.use_lstm_in_advanced),
        dropout=args.dropout,
        use_motion=bool(args.use_raft_motion),
        motion_dim=args.motion_dim,
        motion_fusion_type=args.motion_fusion_type,
        # V4: Enable Actor-Critic
        use_actor_critic=True,
        value_hidden_dim=args.value_hidden_dim
    )
    model = DSNAdvanced(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"  Config: {config}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Index scenes
    all_scene_dirs = build_epoch_index(args.dataset_root)
    if args.video_list:
        target_videos = args.video_list.split(',')
        all_scene_dirs = [p for p in all_scene_dirs if any(v in str(p) for v in target_videos)]
    
    if not all_scene_dirs:
        print("No scenes found.")
        return

    print(f"Found {len(all_scene_dirs)} scenes for training.")

    # Log hyperparameters
    hparams_dict = vars(args)
    hparams_dict["algorithm"] = "PPO"
    hparams_dict["architecture"] = "Actor-Critic"
    writer.add_text('hyperparameters', json.dumps(hparams_dict, indent=2), 0)

    global_step = 0
    best_metric = None

    # Training loop
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="V4 PPO Training", position=0)
    
    for epoch in epoch_pbar:
        np.random.shuffle(all_scene_dirs)
        
        # Initialize experience buffer for this epoch
        buffer = ExperienceBuffer(
            gamma=args.gamma,
            lam=args.gae_lambda,
            normalize_advantages=True,
            device=str(device)
        )
        
        # Accumulators
        ep_rewards: List[float] = []
        ep_components: Dict[str, List[float]] = {}  # Store all components
        look_scores_sel = []
        sakuga_scores_sel = []
        
        # Get curriculum stage
        curriculum_stage = premium_reward.get_curriculum_stage(epoch)
        curriculum_weights = premium_reward.get_curriculum_weights(epoch)
        tqdm.write(f"\n[Epoch {epoch}] Stage: {curriculum_stage}")

        # === ROLLOUT PHASE === #
        # Collect experiences with current policy
        rollout_pbar = tqdm(all_scene_dirs, desc=f"Epoch {epoch} Rollout", leave=False, position=1)
        
        for scene_dir in rollout_pbar:
            # Load scene data
            load_motion = bool(args.use_raft_motion)
            sample = load_scene_dir(scene_dir, load_frames=True, load_motion=load_motion, load_anime_attrs=True)
            
            # Feature concatenation
            feats_clip = sample.feats.astype(np.float32)
            if sample.anime_attrs is not None:
                attrs = sample.anime_attrs.astype(np.float32)
                T = min(len(feats_clip), len(attrs))
                feats = np.concatenate([feats_clip[:T], attrs[:T]], axis=1)
            else:
                feats = feats_clip
            
            feats = l2_normalize(feats, axis=1)
            frames = sample.frames
            T, D = feats.shape
            if T < 2:
                continue

            # Motion
            motion_feats_np = sample.motion
            
            # Budget
            B_target = int(np.clip(int(np.ceil(args.budget_ratio * T)), args.Bmin, args.Bmax))

            # To torch
            x = torch.from_numpy(feats).unsqueeze(0).to(device)
            motion_feats = None
            if motion_feats_np is not None:
                motion_feats = torch.from_numpy(motion_feats_np.astype(np.float32)).unsqueeze(0).to(device)

            # Forward pass (Actor-Critic)
            with torch.no_grad():
                probs, values = model(x, scene_id=str(scene_dir), motion_feats=motion_feats, return_value=True)
                probs = torch.clamp(probs, 1e-6, 1 - 1e-6)

            # Sample actions
            actions, log_probs_t, entropy_t = bernoulli_sample(probs)
            acts = actions.squeeze(0)
            log_probs = log_probs_t.squeeze(0)
            
            sel_idx = (acts == 1).nonzero(as_tuple=False).squeeze(-1).cpu().numpy().tolist()

            # === Compute Rewards === #
            R_base, components_base = reward_combo(
                feats_all=feats,
                sel_idx=sel_idx,
                frames_all=frames,
                w_div=args.w_div,
                w_rep=args.w_rep,
                w_rec=args.w_rec,
                w_fd=args.w_fd,
                w_probsep=args.w_probsep,
                probs=probs.detach().cpu().numpy().squeeze(0),
                return_components=True
            )
            
            # Premium Anime Components
            attrs_for_reward = sample.anime_attrs if sample.anime_attrs is not None else np.zeros((T, 6))
            anime_rewards = premium_reward.compute_reward(attrs_for_reward, sel_idx, current_epoch=epoch)
            R_anime = anime_rewards["total"]
            
            # Ensemble uncertainty bonus (V4)
            if reward_ensemble is not None and sample.anime_attrs is not None:
                ensemble_reward = reward_ensemble.compute_quality_reward(
                    sample.anime_attrs, sel_idx, percentile_threshold=args.percentile_threshold
                )
                uncertainty_penalty = ensemble_reward.get("uncertainty_penalty", 0.0)
                R_anime += uncertainty_penalty * 0.1  # Small weight
            
            R = R_base + R_anime

            # Budget penalty
            if B_target > 0:
                over = max(0, len(sel_idx) - B_target)
                under = max(0, B_target - len(sel_idx))
                R -= args.budget_penalty * (over + 0.5 * under)

            # Store experience in buffer
            buffer.add(
                scene_id=str(scene_dir),
                features=feats,
                actions=acts.cpu().numpy(),
                log_probs=log_probs.cpu().numpy(),
                reward=R,
                values=values.squeeze(0).cpu().numpy(),
                probs=probs.squeeze(0).cpu().numpy()
            )
            
            # Track metrics
            # Track metrics
            ep_rewards.append(R)
            
            # Aggregate all components for logging
            # components_base keys: div, rep, rec, fd, ms, motion, anime_look, anime_sakuga, anime_story, probsep
            # Add total anime reward breakdown if desired
            for k, v in components_base.items():
                if k not in ep_components: ep_components[k] = []
                # Scale by weight for visualization of effective contribution
                weight = 0.0
                if k == 'div': weight = args.w_div
                elif k == 'rep': weight = args.w_rep
                elif k == 'rec': weight = args.w_rec
                elif k == 'fd': weight = args.w_fd
                elif k == 'probsep': weight = args.w_probsep
                elif k.startswith('anime_'): weight = 1.0 # already weighted in R_anime or not part of base
                else: weight = 1.0
                ep_components[k].append(v * weight) # Weighted component
            
            # Also track unweighted anime components from premium_reward
            for k, v in anime_rewards.items():
                if k != 'total':
                    k_full = f"raw_anime_{k}"
                    if k_full not in ep_components: ep_components[k_full] = []
                    ep_components[k_full].append(v)
            
            if sample.anime_attrs is not None and len(sel_idx) > 0:
                look = (sample.anime_attrs[:, 0] + sample.anime_attrs[:, 1] + sample.anime_attrs[:, 2]) / 3.0
                look_scores_sel.extend(look[sel_idx].tolist())
                sakuga_scores_sel.extend(sample.anime_attrs[:, 3][sel_idx].tolist())

        # === PPO UPDATE PHASE === #
        tqdm.write(f"[Epoch {epoch}] Collected {len(buffer)} scenes, running PPO update...")
        
        ppo.reset_tracking()
        ppo_metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            ppo=ppo,
            n_ppo_epochs=args.n_ppo_epochs,
            batch_size=args.batch_size,
            device=device,
            max_grad_norm=args.max_grad_norm
        )
        
        global_step += 1
        
        # Epoch Summary
        meanR = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        stdR = float(np.std(ep_rewards)) if ep_rewards else 0.0
        mean_look = np.mean(look_scores_sel) if look_scores_sel else 0.0
        mean_sakuga = np.mean(sakuga_scores_sel) if sakuga_scores_sel else 0.0
        
        tqdm.write(f"[Epoch {epoch}] R={meanR:.4f}±{stdR:.4f} | "
                  f"Look={mean_look:.3f} | Sakuga={mean_sakuga:.3f} | "
                  f"KL={ppo_metrics['approx_kl']:.4f}")
        
        # TensorBoard logging
        writer.add_scalar("train/mean_reward", meanR, epoch)
        writer.add_scalar("train/std_reward", stdR, epoch)
        writer.add_scalar("aesthetic/mean_look_selected", mean_look, epoch)
        writer.add_scalar("aesthetic/mean_sakuga_selected", mean_sakuga, epoch)
        
        # Log all individual reward components
        for comp_name, comp_vals in ep_components.items():
             if comp_vals:
                 writer.add_scalar(f"components/{comp_name}", np.mean(comp_vals), epoch)
        
        # V4 PPO metrics
        writer.add_scalar("ppo/actor_loss", ppo_metrics["actor_loss"], epoch)
        writer.add_scalar("ppo/critic_loss", ppo_metrics["critic_loss"], epoch)
        writer.add_scalar("ppo/entropy", ppo_metrics["entropy"], epoch)
        writer.add_scalar("ppo/approx_kl", ppo_metrics["approx_kl"], epoch)
        writer.add_scalar("ppo/clip_fraction", ppo_metrics["clip_fraction"], epoch)
        writer.add_scalar("ppo/epochs_completed", ppo_metrics["ppo_epochs_completed"], epoch)
        
        # Curriculum
        for key, val in curriculum_weights.items():
            writer.add_scalar(f"curriculum/{key}_weight", val, epoch)

        # Save Checkpoint
        ckpt_path = save_dir / f"dsn_checkpoint_ep{epoch}.pt"
        torch.save({
            "model": model.state_dict(),
            "config": config,
            "model_type": "advanced_actor_critic",
            "epoch": epoch,
            "version": "v4",
            "ppo_config": vars(ppo_config)
        }, ckpt_path)

        # Validate
        if args.val_videos_dir and args.val_output_dir and (epoch % args.validate_every == 0):
            val_out = Path(args.val_output_dir) / f"ep{epoch}"
            val_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python", "-m", "eval.batch_eval",
                "--videos_dir", args.val_videos_dir,
                "--output_dir", str(val_out),
                "--checkpoint", str(ckpt_path),
                "--device", eval_device,
                "--feat_dim", str(args.feat_dim),
                "--enc_hidden", str(args.enc_hidden),
                "--lstm_hidden", str(args.lstm_hidden),
                "--budget_ratio", str(args.budget_ratio),
                "--Bmin", str(args.Bmin),
                "--Bmax", str(args.Bmax),
                "--sample_stride", str(args.eval_sample_stride),
                "--resize_w", str(args.eval_resize_w),
                "--resize_h", str(args.eval_resize_h),
                "--embedder", args.eval_embedder,
                "--backend", args.eval_backend,
                "--eval_device", eval_device,
                "--use_anime_attrs", str(args.use_anime_attrs),
                "--anime_attrs_dim", str(args.anime_attrs_dim),
            ]
            
            if args.eval_with_baselines:
                cmd.append("--with_baselines")
            if args.eval_max_videos is not None:
                cmd += ["--max_videos", str(args.eval_max_videos)]
            
            tqdm.write("[Validate] Running batch_eval...")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                tqdm.write(f"[Validate][Error] {r.stderr[:500]}")
            else:
                summary_path = val_out / "summary_results.json"
                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    agg = s.get("aggregate_metrics", {})
                    
                    for k, v in agg.items():
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            writer.add_scalar(f"val/{k}", float(v), epoch)
                    
                    rec_mean = agg.get("RecErr_mean", None)
                    if isinstance(rec_mean, (int, float)) and not np.isnan(rec_mean):
                        if (best_metric is None) or (rec_mean < best_metric):
                            best_metric = rec_mean
                            best_ckpt = save_dir / "dsn_checkpoint_best.pt"
                            torch.save({
                                "model": model.state_dict(),
                                "config": config,
                                "model_type": "advanced_actor_critic",
                                "epoch": epoch,
                                "best_metric": best_metric,
                                "version": "v4"
                            }, best_ckpt)
                            tqdm.write(f"  ✅ New best RecErr: {best_metric:.4f}")
                        
                        # Run visualization
                        viz_out = val_out / "plots"
                        viz_cmd = [
                            "python", "-m", "eval.visualize_validation",
                            "--val_output_dir", str(val_out.parent), # Point to parent of epX (val_runs)
                            "--output_dir", str(viz_out),
                            "--epoch", str(epoch)
                        ]
                        subprocess.run(viz_cmd, capture_output=True) # Fail silently if viz fails, it's optional

    # End training
    epoch_pbar.close()
    writer.close()
    
    # Save final checkpoint
    final_ckpt = save_dir / "dsn_checkpoint_final.pt"
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "model_type": "advanced_actor_critic",
        "epoch": args.epochs,
        "version": "v4"
    }, final_ckpt)
    
    print("\n" + "=" * 60)
    if best_metric is not None:
        print(f"✅ Best checkpoint (RecErr={best_metric:.4f}) saved to {save_dir}/dsn_checkpoint_best.pt")
    print(f"🎉 V4 PPO Training Complete! Results in {save_dir}")
    print(f"📊 View TensorBoard: tensorboard --logdir {args.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
