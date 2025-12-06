#!/usr/bin/env python3
"""
Premium Multi-Video RL Training Script for Anime DSN.

This script is a specialized version of the multi-video training pipeline designed 
specifically to maximize Anime-CLIP-IQA scores ("Look", "Sakuga", "Story").

It features:
1. **Premium Reward System**: Uses percentile-based and contrastive rewards.
2. **Curriculum Learning**: Gradually ramps up aesthetic requirements.
3. **Style-Aware Logging**: Detailed tracking of aesthetic metrics.
4. **Gradient Accumulation**: For robust multi-video updates.

Usage:
    python -m src.pipeline.train_rl_dsn_multi_anime_premium ...
"""

from __future__ import annotations
import os, json, argparse, subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import core components
from src.datasets import build_epoch_index, load_scene_dir
from src.models.dsn import EncoderFC, DSNPolicy
from src.models.dsn_advanced import DSNAdvanced, DSNConfig
from src.rl.rewards import reward_combo
from src.rl.premium_rewards import PremiumAnimeReward

# ---------------- utils ---------------- #
def l2_normalize(x: np.ndarray, axis: int=-1, eps: float=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def as_device(d: str) -> torch.device:
    if d.startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.device(d)
        except Exception:
            return torch.device("cuda")
    return torch.device("cpu")

def bernoulli_sample(probs: torch.Tensor):
    m = torch.distributions.Bernoulli(probs)
    a = m.sample()
    return a, m.log_prob(a), m.entropy()

# --------------- training --------------- #
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)

    # Multi-Video Specific
    ap.add_argument("--multi_video", type=int, default=1, help="Enable multi-video training mode")
    ap.add_argument("--batch_size", type=int, default=4, help="Number of videos to accumulate gradients over")
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
    
    # Premium Reward Config
    ap.add_argument("--w_anime_look",   type=float, default=1.0)
    ap.add_argument("--w_anime_sakuga", type=float, default=1.0)
    ap.add_argument("--w_anime_story",  type=float, default=0.5)
    ap.add_argument("--percentile_threshold", type=float, default=0.75, help="Top percentile to reward")
    ap.add_argument("--use_curriculum", type=int, default=1, help="Ramp up weights over time")

    # RL
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--baseline_momentum", type=float, default=0.9)
    ap.add_argument("--budget_ratio", type=float, default=0.06)
    ap.add_argument("--budget_penalty", type=float, default=0.05)
    ap.add_argument("--Bmin", type=int, default=3)
    ap.add_argument("--Bmax", type=int, default=15)

    # Standard Reward weights (kept low to prioritize anime aesthetic)
    ap.add_argument("--w_div", type=float, default=0.5)
    ap.add_argument("--w_rep", type=float, default=0.5)
    ap.add_argument("--w_rec", type=float, default=0.0)
    ap.add_argument("--w_fd",  type=float, default=0.0)
    ap.add_argument("--w_ms",  type=float, default=0.0)
    ap.add_argument("--w_motion", type=float, default=0.0)
    ap.add_argument("--w_probsep", type=float, default=0.1)

    # Optim
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging & validation
    ap.add_argument("--log_dir", type=str, default="runs/dsn_anime_premium")
    ap.add_argument("--val_videos_dir", type=str, default=None)
    ap.add_argument("--val_output_dir", type=str, default=None)
    ap.add_argument("--validate_every", type=int, default=1)
    ap.add_argument("--eval_embedder", type=str, default="clip_vitb32")
    ap.add_argument("--eval_backend", type=str, default="pyscenedetect")
    ap.add_argument("--eval_sample_stride", type=int, default=5)
    ap.add_argument("--eval_resize_w", type=int, default=320)
    ap.add_argument("--eval_resize_h", type=int, default=180)
    ap.add_argument("--eval_device", type=str, default=None)
    ap.add_argument("--eval_with_baselines", action="store_true")
    
    # Eval extra args
    ap.add_argument("--eval_threshold", type=float, default=None)
    ap.add_argument("--eval_model_dir", type=str, default=None)
    ap.add_argument("--eval_weights_path", type=str, default=None)
    ap.add_argument("--eval_prob_threshold", type=float, default=None)
    ap.add_argument("--eval_scene_device", type=str, default=None)
    ap.add_argument("--eval_max_videos", type=int, default=None)
    ap.add_argument("--eval_num_workers", type=int, default=None)

    args = ap.parse_args()

    device = as_device(args.device)
    eval_device = args.eval_device if args.eval_device else args.device

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(args.log_dir)))

    # Initialize Premium Reward System
    premium_reward = PremiumAnimeReward(
        percentile_threshold=args.percentile_threshold,
        use_curriculum=bool(args.use_curriculum),
        total_epochs=args.epochs
    )

    # Model Setup
    print("[Model] Using advanced DSN (Attention + Multi-Scale)")
    # Adjust feat_dim if using anime attributes
    effective_feat_dim = args.feat_dim
    if args.use_anime_attrs:
        effective_feat_dim += args.anime_attrs_dim
        print(f"[Model] Adjusted feat_dim to {effective_feat_dim} (Base: {args.feat_dim} + Anime: {args.anime_attrs_dim})")

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
        motion_fusion_type=args.motion_fusion_type
    )
    model = DSNAdvanced(config).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"  Config: {config}")

    # Baseline for REINFORCE variance reduction
    baseline: Optional[float] = None
    beta = args.baseline_momentum

    # Index scenes
    all_scene_dirs = build_epoch_index(args.dataset_root)
    if args.video_list:
        target_videos = args.video_list.split(',')
        all_scene_dirs = [p for p in all_scene_dirs if any(v in str(p) for v in target_videos)]
    
    if not all_scene_dirs:
        print("No scenes found.")
        return

    print(f"Found {len(all_scene_dirs)} scenes for training.")
    print(f"Batch size: {args.batch_size}")

    # Log hyperparameters
    hparams_dict = vars(args)
    writer.add_text('hyperparameters', json.dumps(hparams_dict, indent=2), 0)

    global_step = 0
    best_metric = None

    # Epoch progress bar
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training", position=0)
    
    for epoch in epoch_pbar:
        np.random.shuffle(all_scene_dirs)
        ep_rewards: List[float] = []
        
        # Accumulators
        sel_count_sum = 0
        frame_count_sum = 0
        entropy_sum = 0.0
        
        # Aesthetic Accumulators
        look_scores_sel = []
        sakuga_scores_sel = []
        story_scores_sel = []

        # Process in batches
        num_batches = (len(all_scene_dirs) + args.batch_size - 1) // args.batch_size
        batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}", leave=False, position=1)
        
        # Update Curriculum Weights
        curr_w_look = premium_reward.get_curriculum_weight(epoch, args.w_anime_look)
        curr_w_sakuga = premium_reward.get_curriculum_weight(epoch, args.w_anime_sakuga)
        curr_w_story = premium_reward.get_curriculum_weight(epoch, args.w_anime_story)
        
        writer.add_scalar('curriculum/w_look', curr_w_look, epoch)
        writer.add_scalar('curriculum/w_sakuga', curr_w_sakuga, epoch)
        
        for batch_idx in batch_pbar:
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(all_scene_dirs))
            batch_scenes = all_scene_dirs[start_idx:end_idx]
            
            opt.zero_grad()
            batch_loss = 0.0
            valid_scenes_in_batch = 0
            
            for scene_dir in batch_scenes:
                # Load scene data
                load_motion = bool(args.use_raft_motion)
                # Always load anime attrs for premium training
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
                if T < 2: continue

                # Motion
                motion_feats_np = sample.motion
                
                # Budget
                B_target = int(np.clip(int(np.ceil(args.budget_ratio * T)), args.Bmin, args.Bmax))

                # To torch
                x = torch.from_numpy(feats).unsqueeze(0).to(device)
                motion_feats = None
                if motion_feats_np is not None:
                    motion_feats = torch.from_numpy(motion_feats_np.astype(np.float32)).unsqueeze(0).to(device)

                # Forward pass
                scene_id = str(scene_dir).replace('/', '_')
                probs = model(x, scene_id=scene_id, motion_feats=motion_feats)
                probs = torch.clamp(probs, 1e-6, 1-1e-6)

                # Sample actions
                actions, logp_t, ent_t = bernoulli_sample(probs)
                acts = actions.squeeze(0)
                log_probs = logp_t.sum(dim=1)
                entropy = ent_t.sum(dim=1)

                sel_idx = (acts == 1).nonzero(as_tuple=False).squeeze(-1).cpu().numpy().tolist()

                # --- Compute Premium Reward ---
                # 1. Standard components (Diversity, Rep, etc.)
                R_base = reward_combo(
                    feats_all=feats,
                    sel_idx=sel_idx,
                    frames_all=frames,
                    w_div=args.w_div, w_rep=args.w_rep, 
                    w_probsep=args.w_probsep,
                    probs=probs.detach().cpu().numpy().squeeze(0)
                )
                
                # 2. Premium Anime Components
                anime_terms = premium_reward.compute_reward(
                    sample.anime_attrs, sel_idx, current_epoch=epoch
                )
                
                R_anime = (curr_w_look * anime_terms["look"] + 
                           curr_w_sakuga * anime_terms["sakuga"] + 
                           curr_w_story * anime_terms["story"])
                
                R = R_base + R_anime

                # Budget penalty
                if B_target > 0:
                    over = max(0, len(sel_idx) - B_target)
                    under = max(0, B_target - len(sel_idx))
                    R -= args.budget_penalty * (over + 0.5 * under)

                R_t = torch.tensor(R, dtype=torch.float32, device=device)

                # Moving baseline
                if baseline is None:
                    baseline = R
                else:
                    baseline = beta * baseline + (1 - beta) * R
                b_t = torch.tensor(baseline, dtype=torch.float32, device=device)

                advantage = R_t - b_t
                loss = - advantage * log_probs - args.entropy_coef * entropy
                loss = loss.mean()
                
                loss = loss / args.batch_size
                loss.backward()
                
                batch_loss += loss.item()
                valid_scenes_in_batch += 1
                
                # Stats
                ep_rewards.append(R)
                sel_count_sum += len(sel_idx)
                frame_count_sum += T
                entropy_sum += float(entropy.item())
                
                # Track raw aesthetic scores of selected frames
                if sample.anime_attrs is not None and len(sel_idx) > 0:
                    # Look = (Sharp+Color+Bright)/3
                    look = (sample.anime_attrs[:, 0] + sample.anime_attrs[:, 1] + sample.anime_attrs[:, 2]) / 3.0
                    look_scores_sel.extend(look[sel_idx].tolist())
                    sakuga_scores_sel.extend(sample.anime_attrs[:, 3][sel_idx].tolist())

            if valid_scenes_in_batch > 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                global_step += 1
                batch_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'R': f'{np.mean(ep_rewards[-valid_scenes_in_batch:]):.3f}'})

        # Epoch Summary
        meanR = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        sel_ratio = (sel_count_sum / frame_count_sum) if frame_count_sum > 0 else 0.0
        mean_entropy = entropy_sum / max(1, len(all_scene_dirs))
        
        mean_look = np.mean(look_scores_sel) if look_scores_sel else 0.0
        mean_sakuga = np.mean(sakuga_scores_sel) if sakuga_scores_sel else 0.0
        
        tqdm.write(f"[Epoch {epoch}] meanR={meanR:.4f} | Look={mean_look:.3f} | Sakuga={mean_sakuga:.3f}")
        
        writer.add_scalar("train/mean_reward", meanR, epoch)
        writer.add_scalar("train/sel_ratio", sel_ratio, epoch)
        writer.add_scalar("aesthetic/mean_look_selected", mean_look, epoch)
        writer.add_scalar("aesthetic/mean_sakuga_selected", mean_sakuga, epoch)
        
        if look_scores_sel:
            writer.add_histogram("aesthetic/look_dist_selected", np.array(look_scores_sel), epoch)
        if sakuga_scores_sel:
            writer.add_histogram("aesthetic/sakuga_dist_selected", np.array(sakuga_scores_sel), epoch)

        # Save Checkpoint
        ckpt_path = save_dir / f"dsn_checkpoint_ep{epoch}.pt"
        torch.save({
            "model": model.state_dict(),
            "config": config,
            "model_type": "advanced"
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
            # Add other eval args...
            if args.eval_threshold is not None: cmd += ["--threshold", str(args.eval_threshold)]
            
            tqdm.write("[Validate] Running batch_eval...")
            subprocess.run(cmd, capture_output=True, text=True)

    writer.close()
    print("Premium Training done.")

if __name__ == "__main__":
    main()
