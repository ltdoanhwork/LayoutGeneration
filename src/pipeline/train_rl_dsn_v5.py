#!/usr/bin/env python3
"""
V5 Multi-Task RL Training Script

This script implements Multi-Task RL for anime video summarization:
- Task 1 (RecErr): Optimize reconstruction error + Frechet distance
- Task 2 (Anime): Optimize anime quality metrics (sharpness, sakuga, etc.)

Each task has its own policy and value head, sharing a common backbone.
Losses are computed separately and summed - no weight tuning needed!

Usage:
    python -m src.pipeline.train_rl_dsn_v5 --dataset_root data/sakuga_dataset_100_samples ...
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
from src.models.dsn_advanced import DSNConfig
from src.models.dsn_multitask import DSNMultiTask, MultiTaskConfig
from src.rl.rewards import reward_combo_v4 as reward_combo
from src.rl.premium_rewards_v3 import PremiumAnimeRewardV3
from src.rl.ppo_core import PPOCore, PPOConfig, compute_log_prob_bernoulli
from src.rl.gae import GAEComputer


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


# ---------------- Multi-Task PPO Update ---------------- #
def multitask_ppo_update(
    model: DSNMultiTask,
    optimizer: optim.Optimizer,
    experiences: List[Dict],
    ppo_config: PPOConfig,
    n_ppo_epochs: int = 4,
    device: torch.device = torch.device("cuda"),
    max_grad_norm: float = 0.5
) -> Dict[str, float]:
    """
    Perform Multi-Task PPO update.
    
    Each experience contains:
    - features, actions, old_log_probs_{rec,anime}, old_values_{rec,anime}
    - rewards_{rec,anime}, advantages_{rec,anime}, returns_{rec,anime}
    """
    all_metrics = {
        "rec_actor_loss": [], "rec_critic_loss": [],
        "anime_actor_loss": [], "anime_critic_loss": [],
        "entropy": [], "approx_kl": []
    }
    
    for _ in range(n_ppo_epochs):
        np.random.shuffle(experiences)
        
        for exp in experiences:
            optimizer.zero_grad()
            
            features = torch.from_numpy(exp["features"]).unsqueeze(0).to(device)
            actions = torch.from_numpy(exp["actions"]).to(device)
            
            # Forward all tasks
            task_outputs = model(features, return_all_tasks=True)
            
            total_loss = 0.0
            
            for task in ["rec", "anime"]:
                probs, logits, values = task_outputs[task]
                probs = probs.squeeze(0)
                values = values.squeeze(0)
                
                # Clamp probs
                probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
                
                # Compute new log probs
                new_log_probs = compute_log_prob_bernoulli(probs, actions)
                
                # Get old values
                old_log_probs = torch.from_numpy(exp[f"old_log_probs_{task}"]).to(device)
                old_values = torch.from_numpy(exp[f"old_values_{task}"]).to(device)
                advantages = torch.from_numpy(exp[f"advantages_{task}"]).to(device)
                returns = torch.from_numpy(exp[f"returns_{task}"]).to(device)
                
                # PPO Actor Loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                clip_adv = torch.clamp(ratio, 1 - ppo_config.clip_range, 1 + ppo_config.clip_range) * advantages
                actor_loss = -torch.min(ratio * advantages, clip_adv).mean()
                
                # Critic Loss
                values_clipped = old_values + torch.clamp(values - old_values, -ppo_config.clip_range, ppo_config.clip_range)
                critic_loss1 = (values - returns) ** 2
                critic_loss2 = (values_clipped - returns) ** 2
                critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean()
                
                total_loss += actor_loss + ppo_config.vf_coef * critic_loss
                
                all_metrics[f"{task}_actor_loss"].append(actor_loss.item())
                all_metrics[f"{task}_critic_loss"].append(critic_loss.item())
            
            # Entropy bonus (from merged policy)
            merged_probs = model(features).squeeze(0)
            merged_probs = torch.clamp(merged_probs, 1e-6, 1 - 1e-6)
            entropy = -(merged_probs * torch.log(merged_probs) + (1 - merged_probs) * torch.log(1 - merged_probs)).mean()
            total_loss -= ppo_config.entropy_coef * entropy
            
            all_metrics["entropy"].append(entropy.item())
            
            # Approximate KL
            with torch.no_grad():
                old_log_probs_merged = 0.5 * (
                    torch.from_numpy(exp["old_log_probs_rec"]).to(device) +
                    torch.from_numpy(exp["old_log_probs_anime"]).to(device)
                )
                new_log_probs_merged = compute_log_prob_bernoulli(merged_probs, actions)
                approx_kl = (old_log_probs_merged - new_log_probs_merged).mean().item()
                all_metrics["approx_kl"].append(abs(approx_kl))
            
            # Backward
            total_loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
    
    return {k: np.mean(v) for k, v in all_metrics.items()}


# ---------------- Main Training Loop ---------------- #
def main():
    ap = argparse.ArgumentParser(description="V5 Multi-Task RL Training")
    
    # Data
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=60)

    # Multi-Video
    ap.add_argument("--multi_video", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--video_list", type=str, default=None)

    # Device
    ap.add_argument("--device", type=str, default="cuda")

    # Model
    ap.add_argument("--feat_dim", type=int, default=512)
    ap.add_argument("--enc_hidden", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num_attn_heads", type=int, default=4)
    ap.add_argument("--num_attn_layers", type=int, default=2)
    ap.add_argument("--use_cache", type=int, default=1)
    ap.add_argument("--use_lstm_in_advanced", type=int, default=1)
    
    # Motion & Anime
    ap.add_argument("--use_raft_motion", type=int, default=1)
    ap.add_argument("--motion_dim", type=int, default=128)
    ap.add_argument("--use_anime_attrs", type=int, default=1)
    ap.add_argument("--anime_attrs_dim", type=int, default=6)
    
    # PPO
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--target_kl", type=float, default=0.02)
    ap.add_argument("--n_ppo_epochs", type=int, default=6)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--entropy_coef", type=float, default=0.02)
    ap.add_argument("--value_hidden_dim", type=int, default=128)
    
    # GAE
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    
    # RL Rewards
    ap.add_argument("--budget_ratio", type=float, default=0.06)
    ap.add_argument("--budget_penalty", type=float, default=0.05)
    ap.add_argument("--Bmin", type=int, default=3)
    ap.add_argument("--Bmax", type=int, default=15)
    
    # Premium Anime Rewards
    ap.add_argument("--percentile_threshold", type=float, default=0.80)
    ap.add_argument("--use_curriculum", type=int, default=1)

    # Optim
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)

    # Logging
    ap.add_argument("--log_dir", type=str, default="runs/dsn_v5/logs")
    ap.add_argument("--val_videos_dir", type=str, default=None)
    ap.add_argument("--val_output_dir", type=str, default=None)
    ap.add_argument("--validate_every", type=int, default=5)
    ap.add_argument("--eval_embedder", type=str, default="clip_vitb32")
    ap.add_argument("--eval_backend", type=str, default="transnetv2")
    ap.add_argument("--eval_device", type=str, default=None)
    ap.add_argument("--eval_with_baselines", action="store_true")

    args = ap.parse_args()

    device = as_device(args.device)
    eval_device = args.eval_device if args.eval_device else args.device

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(args.log_dir)))

    print("=" * 60)
    print("V5 Multi-Task RL Training")
    print("=" * 60)
    print(f"[V5] Tasks: RecErr + Anime (separate heads)")
    print(f"[V5] PPO: clip={args.clip_range}, epochs={args.n_ppo_epochs}")

    # Premium Reward System
    premium_reward = PremiumAnimeRewardV3(
        percentile_threshold=args.percentile_threshold,
        use_curriculum=bool(args.use_curriculum),
        total_epochs=args.epochs
    )

    # PPO Config
    ppo_config = PPOConfig(
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        n_ppo_epochs=args.n_ppo_epochs,
        entropy_coef=args.entropy_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm
    )

    # GAE
    gae_computer = GAEComputer(gamma=args.gamma, lam=args.gae_lambda, normalize=True)

    # Model Setup
    print("\n[Model] Creating Multi-Task DSN")
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
        use_cache=bool(args.use_cache),
        use_lstm=bool(args.use_lstm_in_advanced),
        dropout=args.dropout,
        use_motion=bool(args.use_raft_motion),
        motion_dim=args.motion_dim,
        use_actor_critic=True,
        value_hidden_dim=args.value_hidden_dim
    )
    
    model = DSNMultiTask(config).to(device)
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
    hparams_dict["algorithm"] = "Multi-Task PPO"
    hparams_dict["architecture"] = "DSNMultiTask"
    writer.add_text('hyperparameters', json.dumps(hparams_dict, indent=2), 0)

    best_metric = None

    # Training loop
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="V5 Multi-Task Training", position=0)
    
    for epoch in epoch_pbar:
        np.random.shuffle(all_scene_dirs)
        
        # Collect experiences
        experiences = []
        ep_rewards_rec = []
        ep_rewards_anime = []
        
        curriculum_stage = premium_reward.get_curriculum_stage(epoch)
        tqdm.write(f"\n[Epoch {epoch}] Stage: {curriculum_stage}")

        rollout_pbar = tqdm(all_scene_dirs, desc=f"Epoch {epoch} Rollout", leave=False, position=1)
        
        for scene_dir in rollout_pbar:
            # Load scene
            load_motion = bool(args.use_raft_motion)
            sample = load_scene_dir(scene_dir, load_frames=True, load_motion=load_motion, load_anime_attrs=True)
            
            # Features
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

            motion_feats_np = sample.motion
            B_target = int(np.clip(int(np.ceil(args.budget_ratio * T)), args.Bmin, args.Bmax))

            # To torch
            x = torch.from_numpy(feats).unsqueeze(0).to(device)
            motion_feats = None
            if motion_feats_np is not None:
                motion_feats = torch.from_numpy(motion_feats_np.astype(np.float32)).unsqueeze(0).to(device)

            # Forward all tasks
            with torch.no_grad():
                task_outputs = model(x, scene_id=str(scene_dir), motion_feats=motion_feats, return_all_tasks=True)
            
            # Get merged policy for action sampling
            probs_rec = task_outputs["rec"][0].squeeze(0).clamp(1e-6, 1 - 1e-6)
            probs_anime = task_outputs["anime"][0].squeeze(0).clamp(1e-6, 1 - 1e-6)
            
            # Merge for sampling
            alpha = torch.sigmoid(model.merge_weight).item()
            merged_probs = alpha * probs_rec + (1 - alpha) * probs_anime
            
            # Sample actions
            actions, log_probs_merged, _ = bernoulli_sample(merged_probs)
            sel_idx = (actions == 1).nonzero(as_tuple=False).squeeze(-1).cpu().numpy().tolist()

            # Compute SEPARATE rewards for each task
            # Task 1: RecErr reward
            R_rec, comp_rec = reward_combo(
                feats_all=feats,
                sel_idx=sel_idx,
                frames_all=frames,
                w_div=0.5,
                w_rep=1.0,
                w_rec=3.0,  # Focus
                w_fd=2.0,   # Focus
                w_probsep=0.1,
                probs=merged_probs.detach().cpu().numpy(),
                return_components=True
            )
            
            # Task 2: Anime reward
            attrs_for_reward = sample.anime_attrs if sample.anime_attrs is not None else np.zeros((T, 6))
            anime_rewards = premium_reward.compute_reward(attrs_for_reward, sel_idx, current_epoch=epoch)
            R_anime = anime_rewards["total"]
            
            # Add diversity for anime task too
            R_anime += 0.5 * comp_rec.get("div", 0.0)
            
            # Budget penalty (shared)
            budget_penalty = 0.0
            if B_target > 0:
                over = max(0, len(sel_idx) - B_target)
                under = max(0, B_target - len(sel_idx))
                budget_penalty = args.budget_penalty * (over + 0.5 * under)
            
            R_rec -= budget_penalty
            R_anime -= budget_penalty
            
            ep_rewards_rec.append(R_rec)
            ep_rewards_anime.append(R_anime)

            # Store experience
            exp = {
                "features": feats,
                "actions": actions.cpu().numpy(),
                "old_log_probs_rec": compute_log_prob_bernoulli(probs_rec, actions).cpu().numpy(),
                "old_log_probs_anime": compute_log_prob_bernoulli(probs_anime, actions).cpu().numpy(),
                "old_values_rec": task_outputs["rec"][2].squeeze(0).cpu().numpy(),
                "old_values_anime": task_outputs["anime"][2].squeeze(0).cpu().numpy(),
            }
            
            # Compute GAE for each task
            for task, R in [("rec", R_rec), ("anime", R_anime)]:
                values = exp[f"old_values_{task}"]
                # Simple reward assignment (per-scene reward to all frames)
                rewards = np.full_like(values, R / len(values))
                
                # GAE - convert to torch tensors
                rewards_t = torch.from_numpy(rewards).float()
                values_t = torch.from_numpy(values).float()
                
                advantages, returns = gae_computer.compute(
                    rewards=rewards_t,
                    values=values_t
                )
                exp[f"advantages_{task}"] = advantages.numpy() if isinstance(advantages, torch.Tensor) else advantages
                exp[f"returns_{task}"] = returns.numpy() if isinstance(returns, torch.Tensor) else returns
            
            experiences.append(exp)

        # PPO Update
        tqdm.write(f"[Epoch {epoch}] Collected {len(experiences)} scenes, running Multi-Task PPO...")
        
        ppo_metrics = multitask_ppo_update(
            model=model,
            optimizer=optimizer,
            experiences=experiences,
            ppo_config=ppo_config,
            n_ppo_epochs=args.n_ppo_epochs,
            device=device,
            max_grad_norm=args.max_grad_norm
        )
        
        # Epoch Summary
        mean_rec = np.mean(ep_rewards_rec) if ep_rewards_rec else 0.0
        mean_anime = np.mean(ep_rewards_anime) if ep_rewards_anime else 0.0
        merge_weight = torch.sigmoid(model.merge_weight).item()
        
        tqdm.write(f"[Epoch {epoch}] R_rec={mean_rec:.4f} | R_anime={mean_anime:.4f} | α={merge_weight:.3f}")
        
        # TensorBoard logging
        writer.add_scalar("train/reward_rec", mean_rec, epoch)
        writer.add_scalar("train/reward_anime", mean_anime, epoch)
        writer.add_scalar("train/merge_weight_alpha", merge_weight, epoch)
        
        for k, v in ppo_metrics.items():
            writer.add_scalar(f"ppo/{k}", v, epoch)

        # Save Checkpoint
        ckpt_path = save_dir / f"dsn_checkpoint_ep{epoch}.pt"
        torch.save({
            "model": model.state_dict(),
            "config": config,
            "model_type": "multitask_v5",
            "epoch": epoch,
            "version": "v5",
            "merge_weight": merge_weight
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
                "--embedder", args.eval_embedder,
                "--backend", args.eval_backend,
                "--use_anime_attrs", str(args.use_anime_attrs),
                "--min_scene_len", "48",
                "--model_dir", "./src/models/TransNetV2",
                "--prob_threshold", "0.5",
                "--scene_device", eval_device,
                "--sample_stride", "4",  # Match fps=6 on 24fps video
                "--resize_w", "0",
                "--resize_h", "0",
            ]
            
            if args.eval_with_baselines:
                cmd.append("--with_baselines")
            
            tqdm.write("[Validate] Running batch_eval...")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                tqdm.write(f"[Validate][Error] {r.stderr[:500]}")
            else:
                summary_path = val_out / "summary_results.json"
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        s = json.load(f)
                    agg = s.get("aggregate_metrics", {})
                    
                    for k, v in agg.items():
                        if v is not None:
                            writer.add_scalar(f"val/{k}", float(v), epoch)
                    
                    rec_mean = agg.get("RecErr_mean")
                    if rec_mean is not None:
                        if best_metric is None or rec_mean < best_metric:
                            best_metric = rec_mean
                            best_ckpt = save_dir / "dsn_checkpoint_best.pt"
                            torch.save({
                                "model": model.state_dict(),
                                "config": config,
                                "model_type": "multitask_v5",
                                "epoch": epoch,
                                "best_metric": best_metric,
                                "version": "v5"
                            }, best_ckpt)
                            tqdm.write(f"  ✅ New best RecErr: {best_metric:.4f}")

    # Finished
    epoch_pbar.close()
    writer.close()
    
    final_ckpt = save_dir / "dsn_checkpoint_final.pt"
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "model_type": "multitask_v5",
        "epoch": args.epochs,
        "version": "v5"
    }, final_ckpt)
    
    print("\n" + "=" * 60)
    if best_metric is not None:
        print(f"✅ Best checkpoint (RecErr={best_metric:.4f}) saved")
    print(f"🎉 V5 Multi-Task Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
