#!/usr/bin/env python3
"""
Multi-Video RL Training Script for DSN.

This script implements multi-video training using Gradient Accumulation.
It processes a batch of videos sequentially, accumulating gradients, and then performs a single optimizer step.
This avoids padding issues with the multi-scale DSN model while providing the benefits of batch training.

Usage:
    python -m src.pipeline.train_rl_dsn_multi --dataset_root ... --multi_video 1 --batch_size 4 ...
"""

from __future__ import annotations
import os, json, argparse, subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

# Import core components from existing pipeline
from src.datasets import build_epoch_index, load_scene_dir
from src.vision_flow import compute_flow_magnitude_robust
from src.models.dsn import EncoderFC, DSNPolicy
from src.models.dsn_advanced import DSNAdvanced, DSNConfig
from src.rl.rewards import reward_combo

# Import new replay buffer
from src.rl.multi_video_replay import MultiVideoReplayBuffer, Episode

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
    ap.add_argument("--epochs", type=int, default=5)

    # Multi-Video Specific
    ap.add_argument("--multi_video", type=int, default=0, help="Enable multi-video training mode")
    ap.add_argument("--batch_size", type=int, default=4, help="Number of videos to accumulate gradients over")
    ap.add_argument("--video_list", type=str, default=None, help="Optional comma-separated list of video IDs/paths")
    ap.add_argument("--sampling_strategy", type=str, default="random_uniform", choices=["random_uniform", "round_robin"])

    # Device
    ap.add_argument("--device", type=str, default="cuda")

    # Model
    ap.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "advanced"])
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
    ap.add_argument("--use_raft_motion", type=int, default=0)
    ap.add_argument("--motion_dim", type=int, default=128)
    ap.add_argument("--motion_fusion_type", type=str, default="cross_attention")

    # Anime-CLIP-IQA
    ap.add_argument("--use_anime_attrs", type=int, default=0)
    ap.add_argument("--anime_attrs_dim", type=int, default=0)
    
    # Anime-CLIP-IQA reward weights
    ap.add_argument("--use_anime_reward", type=int, default=0)
    ap.add_argument("--w_anime_look",   type=float, default=0.0)
    ap.add_argument("--w_anime_sakuga", type=float, default=0.0)
    ap.add_argument("--w_anime_story",  type=float, default=0.0)

    # RL
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--baseline_momentum", type=float, default=0.9)
    ap.add_argument("--budget_ratio", type=float, default=0.06)
    ap.add_argument("--budget_penalty", type=float, default=0.05)
    ap.add_argument("--Bmin", type=int, default=3)
    ap.add_argument("--Bmax", type=int, default=15)

    # Reward weights
    ap.add_argument("--w_div", type=float, default=1.0)
    ap.add_argument("--w_rep", type=float, default=1.0)
    ap.add_argument("--w_rec", type=float, default=0.5)
    ap.add_argument("--w_fd",  type=float, default=0.2)
    ap.add_argument("--w_ms",  type=float, default=0.2)
    ap.add_argument("--w_motion", type=float, default=0.2)
    ap.add_argument("--w_probsep", type=float, default=0.0)
    ap.add_argument("--ms_swd_scales", type=int, default=3)
    ap.add_argument("--ms_swd_dirs",   type=int, default=16)

    # Options
    ap.add_argument("--use_motion", type=int, default=1)
    ap.add_argument("--use_lpips_div", type=int, default=0)
    ap.add_argument("--lpips_net", type=str, default="alex")
    ap.add_argument("--lpips_device", type=str, default="cuda")

    # Optim
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging & validation
    ap.add_argument("--log_dir", type=str, default="runs/dsn_rl_multi")
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

    args = ap.parse_args()

    device = as_device(args.device)
    eval_device = args.eval_device if args.eval_device else args.device

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(args.log_dir)))

    # Model Setup
    if args.model_type == "baseline":
        print("[Model] Using baseline DSN (BiLSTM)")
        enc = EncoderFC(args.feat_dim, args.enc_hidden).to(device)
        pol = DSNPolicy(args.enc_hidden, args.lstm_hidden, dropout=args.dropout).to(device)
        opt = optim.Adam(
            list(enc.parameters()) + list(pol.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
        model = None
    elif args.model_type == "advanced":
        print("[Model] Using advanced DSN (Attention + Multi-Scale)")
        config = DSNConfig(
            feat_dim=args.feat_dim,
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
        enc = None
        pol = None
        print(f"  Config: {config}")
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Baseline for REINFORCE variance reduction
    baseline: Optional[float] = None
    beta = args.baseline_momentum

    # Index scenes
    all_scene_dirs = build_epoch_index(args.dataset_root)
    if args.video_list:
        # Filter scenes based on video list if provided
        target_videos = args.video_list.split(',')
        all_scene_dirs = [p for p in all_scene_dirs if any(v in str(p) for v in target_videos)]
    
    if not all_scene_dirs:
        print("No scenes found.")
        return

    print(f"Found {len(all_scene_dirs)} scenes for training.")
    print(f"Multi-video mode: {bool(args.multi_video)}")
    print(f"Batch size (gradient accumulation): {args.batch_size}")

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
        
        # Accumulators for TB
        sel_count_sum = 0
        frame_count_sum = 0
        entropy_sum = 0.0
        mean_prob_sum = 0.0
        
        # Gradient accumulators
        grad_norms_before_clip = []
        grad_norms_after_clip = []

        # Process in batches (Gradient Accumulation)
        num_batches = (len(all_scene_dirs) + args.batch_size - 1) // args.batch_size
        
        batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}", leave=False, position=1)
        
        for batch_idx in batch_pbar:
            # Get batch of scenes
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(all_scene_dirs))
            batch_scenes = all_scene_dirs[start_idx:end_idx]
            
            opt.zero_grad()
            batch_loss = 0.0
            valid_scenes_in_batch = 0
            
            # Forward pass and loss computation for each scene in batch
            for scene_dir in batch_scenes:
                # Load scene data
                load_motion = bool(args.use_raft_motion) and args.model_type == "advanced"
                load_anime_attrs = (bool(args.use_anime_attrs) or bool(args.use_anime_reward) or 
                                  args.w_anime_look > 0 or args.w_anime_sakuga > 0 or args.w_anime_story > 0)
                sample = load_scene_dir(scene_dir, load_frames=True, load_motion=load_motion, load_anime_attrs=load_anime_attrs)
                
                # Feature concatenation (Track A)
                feats_clip = sample.feats.astype(np.float32)
                if args.use_anime_attrs and (sample.anime_attrs is not None):
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

                # Motion features
                motion_feats_np = sample.motion
                motion = None
                if args.use_motion and (frames is not None) and (len(frames) > 1):
                    try:
                        motion = compute_flow_magnitude_robust(frames)
                    except Exception:
                        motion = None

                # Budget
                B_target = int(np.clip(int(np.ceil(args.budget_ratio * T)), args.Bmin, args.Bmax))

                # To torch
                x = torch.from_numpy(feats).unsqueeze(0).to(device)
                motion_feats = None
                if motion_feats_np is not None:
                    motion_feats = torch.from_numpy(motion_feats_np.astype(np.float32)).unsqueeze(0).to(device)

                # Forward pass
                if args.model_type == "baseline":
                    h = enc(x)
                    probs = pol(h)
                else:
                    scene_id = str(scene_dir).replace('/', '_')
                    probs = model(x, scene_id=scene_id, motion_feats=motion_feats)
                
                probs = torch.clamp(probs, 1e-6, 1-1e-6)

                # Sample actions
                actions, logp_t, ent_t = bernoulli_sample(probs)
                acts = actions.squeeze(0)
                log_probs = logp_t.sum(dim=1)
                entropy = ent_t.sum(dim=1)

                sel_idx = (acts == 1).nonzero(as_tuple=False).squeeze(-1).cpu().numpy().tolist()

                # Compute reward
                R = reward_combo(
                    feats_all=feats,
                    sel_idx=sel_idx,
                    frames_all=frames,
                    motion=motion,
                    w_div=args.w_div, w_rep=args.w_rep, w_rec=args.w_rec,
                    w_fd=args.w_fd, w_ms=args.w_ms, w_motion=args.w_motion,
                    w_anime_look=args.w_anime_look, 
                    w_anime_sakuga=args.w_anime_sakuga, 
                    w_anime_story=args.w_anime_story,
                    w_probsep=args.w_probsep,
                    anime_scores=sample.anime_attrs if args.use_anime_reward or (args.w_anime_look > 0 or args.w_anime_sakuga > 0 or args.w_anime_story > 0) else None,
                    probs=probs.detach().cpu().numpy().squeeze(0),
                    ms_swd_scales=args.ms_swd_scales, ms_swd_dirs=args.ms_swd_dirs,
                    use_lpips_div=bool(args.use_lpips_div),
                    lpips_net=args.lpips_net,
                    lpips_device=args.lpips_device,
                )

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
                
                # Normalize loss by batch size for gradient accumulation
                loss = loss / args.batch_size
                loss.backward()
                
                batch_loss += loss.item()
                valid_scenes_in_batch += 1
                
                # Stats
                ep_rewards.append(R)
                sel_count_sum += len(sel_idx)
                frame_count_sum += T
                entropy_sum += float(entropy.item())
                mean_prob_sum += float(probs.mean().item())

            if valid_scenes_in_batch > 0:
                # Gradient monitoring & clipping
                if args.model_type == "baseline":
                    params = list(enc.parameters()) + list(pol.parameters())
                else:
                    params = list(model.parameters())
                
                # Norm before clip
                grad_norm_before = 0.0
                for p in params:
                    if p.grad is not None:
                        grad_norm_before += p.grad.data.norm(2).item() ** 2
                grad_norm_before = grad_norm_before ** 0.5
                grad_norms_before_clip.append(grad_norm_before)
                
                # Clip
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                
                # Norm after clip
                grad_norm_after = 0.0
                for p in params:
                    if p.grad is not None:
                        grad_norm_after += p.grad.data.norm(2).item() ** 2
                grad_norm_after = grad_norm_after ** 0.5
                grad_norms_after_clip.append(grad_norm_after)

                opt.step()
                global_step += 1
                
                batch_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'R': f'{np.mean(ep_rewards[-valid_scenes_in_batch:]):.3f}'})

        # Epoch Summary
        meanR = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        sel_ratio = (sel_count_sum / frame_count_sum) if frame_count_sum > 0 else 0.0
        mean_entropy = entropy_sum / max(1, len(all_scene_dirs))
        
        tqdm.write(f"[Epoch {epoch}] meanR={meanR:.4f} | sel_ratio={sel_ratio:.4f}")
        
        writer.add_scalar("train/mean_reward", meanR, epoch)
        writer.add_scalar("train/sel_ratio", sel_ratio, epoch)
        
        # Gradient stats
        if grad_norms_before_clip:
            writer.add_scalar('gradients/norm_before_clip_mean', np.mean(grad_norms_before_clip), epoch)
            writer.add_scalar('gradients/clip_ratio', np.mean(grad_norms_after_clip) / (np.mean(grad_norms_before_clip) + 1e-8), epoch)

        # Save Checkpoint
        ckpt_path = save_dir / f"dsn_checkpoint_ep{epoch}.pt"
        if args.model_type == "baseline":
            torch.save({"encoder": enc.state_dict(), "policy": pol.state_dict()}, ckpt_path)
        else:
            torch.save({
                "model": model.state_dict(),
                "config": config,
                "model_type": "advanced"
            }, ckpt_path)

    writer.close()
    print("Training done.")

if __name__ == "__main__":
    main()
