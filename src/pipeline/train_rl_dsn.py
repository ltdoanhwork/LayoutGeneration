#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse, subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets import build_epoch_index, load_scene_dir
from src.vision_flow import compute_flow_magnitude_robust
from src.models.dsn import EncoderFC, DSNPolicy
from src.models.dsn_advanced import DSNAdvanced, DSNConfig
from src.rl.rewards import reward_combo

# ---------------- utils ---------------- #
def l2_normalize(x: np.ndarray, axis: int=-1, eps: float=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def as_device(d: str) -> torch.device:
    """
    Accepts: 'cuda', 'cuda:0', 'cpu'. Falls back to CPU if CUDA not available.
    """
    if d.startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.device(d)
        except Exception:
            return torch.device("cuda")
    return torch.device("cpu")

def bernoulli_sample(probs: torch.Tensor):
    """
    probs: (B,T) in (0,1)
    Returns:
      actions: (B,T) in {0,1}
      logp    : (B,T)
      entropy : (B,T)
    """
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

    # Device
    ap.add_argument("--device", type=str, default="cuda")

    # Model
    ap.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "advanced"],
                    help="Model type: baseline (BiLSTM) or advanced (attention+multi-scale)")
    ap.add_argument("--feat_dim", type=int, default=512)
    ap.add_argument("--enc_hidden", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    
    # Advanced model hyperparameters (only used if model_type=advanced)
    ap.add_argument("--num_attn_heads", type=int, default=4,
                    help="[Advanced] Number of attention heads")
    ap.add_argument("--num_attn_layers", type=int, default=2,
                    help="[Advanced] Number of attention layers")
    ap.add_argument("--num_scales", type=int, default=3,
                    help="[Advanced] Number of temporal scales (1, 2, 3, or 4)")
    ap.add_argument("--use_cache", type=int, default=1,
                    help="[Advanced] Enable feature caching (0 or 1)")
    ap.add_argument("--cache_size", type=int, default=1000,
                    help="[Advanced] Max cache size (number of scenes)")
    ap.add_argument("--pos_encoding_type", type=str, default="sinusoidal",
                    choices=["sinusoidal", "learned"],
                    help="[Advanced] Positional encoding type")
    ap.add_argument("--use_lstm_in_advanced", type=int, default=1,
                    help="[Advanced] Use LSTM in advanced model (0 or 1, for ablation)")
    
    # RAFT Motion features
    ap.add_argument("--use_raft_motion", type=int, default=0,
                    help="Use precomputed RAFT motion features (0 or 1)")
    ap.add_argument("--motion_dim", type=int, default=128,
                    help="Motion feature dimension")
    ap.add_argument("--motion_fusion_type", type=str, default="cross_attention",
                    choices=["cross_attention", "simple"],
                    help="Motion fusion type")

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
    ap.add_argument("--log_dir", type=str, default="runs/dsn_rl")
    ap.add_argument("--val_videos_dir", type=str, default=None,
                    help="Folder of raw videos for validation batch eval")
    ap.add_argument("--val_output_dir", type=str, default=None,
                    help="Where to write validation outputs")
    ap.add_argument("--validate_every", type=int, default=1)
    # Embedder for eval (SHOULD match prepare: default clip_vitb32)
    ap.add_argument("--eval_embedder", type=str, default="clip_vitb32")
    # Scene backend for eval (keep in sync with prepare)
    ap.add_argument("--eval_backend", type=str, default="pyscenedetect")
    ap.add_argument("--eval_threshold", type=float, default=None)
    ap.add_argument("--eval_model_dir", type=str, default=None)
    ap.add_argument("--eval_weights_path", type=str, default=None)
    ap.add_argument("--eval_prob_threshold", type=float, default=None)
    ap.add_argument("--eval_scene_device", type=str, default=None)

    # Eval sampling/resize
    ap.add_argument("--eval_sample_stride", type=int, default=5)
    ap.add_argument("--eval_resize_w", type=int, default=320)
    ap.add_argument("--eval_resize_h", type=int, default=180)
    ap.add_argument("--eval_device", type=str, default=None,
                    help="device for eval.run_dsn_pipeline; default = --device")
    ap.add_argument("--eval_with_baselines", action="store_true",
                    help="Evaluate uniform/mid/motion baselines too")
    ap.add_argument("--eval_max_videos", type=int, default=None)
    ap.add_argument("--eval_num_workers", type=int, default=None)

    args = ap.parse_args()

    device = as_device(args.device)
    eval_device = args.eval_device if args.eval_device else args.device

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(args.log_dir)))

    # Model
    if args.model_type == "baseline":
        print("[Model] Using baseline DSN (BiLSTM)")
        enc = EncoderFC(args.feat_dim, args.enc_hidden).to(device)
        pol = DSNPolicy(args.enc_hidden, args.lstm_hidden, dropout=args.dropout).to(device)
        opt = optim.Adam(
            list(enc.parameters()) + list(pol.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
        model = None  # Use separate enc/pol
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
            # RAFT motion
            use_motion=bool(args.use_raft_motion),
            motion_dim=args.motion_dim,
            motion_fusion_type=args.motion_fusion_type
        )
        model = DSNAdvanced(config).to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        enc = None  # Not used in advanced mode
        pol = None
        print(f"  Config: {config}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Baseline for REINFORCE variance reduction
    baseline: Optional[float] = None
    beta = args.baseline_momentum

    # Index scenes
    scene_dirs = build_epoch_index(args.dataset_root)
    if not scene_dirs:
        print("No scenes found.")
        return

    global_step = 0
    best_metric = None
    best_ckpt_path = None

    # Log hyperparameters to tensorboard
    hparams_dict = {
        'lr': args.lr,
        'entropy_coef': args.entropy_coef,
        'budget_ratio': args.budget_ratio,
        'budget_penalty': args.budget_penalty,
        'w_div': args.w_div,
        'w_rep': args.w_rep,
        'w_rec': args.w_rec,
        'w_fd': args.w_fd,
        'w_ms': args.w_ms,
        'w_motion': args.w_motion,
        'feat_dim': args.feat_dim,
        'enc_hidden': args.enc_hidden,
        'lstm_hidden': args.lstm_hidden,
    }
    writer.add_text('hyperparameters', 
                    '\n'.join([f'{k}: {v}' for k, v in hparams_dict.items()]), 0)

    # Epoch progress bar
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training", position=0)
    
    for epoch in epoch_pbar:
        np.random.shuffle(scene_dirs)
        ep_rewards: List[float] = []

        # Accumulators for TB
        sel_count_sum = 0
        frame_count_sum = 0
        entropy_sum = 0.0
        mean_prob_sum = 0.0
        budget_gap_sum = 0.0
        
        # Collect data for visualizations
        all_probs = []  # For histogram
        all_rewards = []  # For distribution
        sample_frames_selected = []  # For image visualization
        sample_frames_rejected = []

        # Scene progress bar
        scene_pbar = tqdm(scene_dirs, desc=f"Epoch {epoch}", leave=False, position=1)
        
        for scene_dir in scene_pbar:
            # Load scene data with optional RAFT motion
            load_motion = bool(args.use_raft_motion) and args.model_type == "advanced"
            sample = load_scene_dir(scene_dir, load_frames=True, load_motion=load_motion)
            feats = l2_normalize(sample.feats.astype(np.float32), axis=1)
            frames = sample.frames
            T, D = feats.shape
            if T < 2:
                continue

            # RAFT motion features (precomputed)
            motion_feats_np = sample.motion  # (T, D_m) or None
            
            # Old motion for reward computation (if enabled)
            motion = None
            if args.use_motion and (frames is not None) and (len(frames) > 1):
                try:
                    motion = compute_flow_magnitude_robust(frames)
                except Exception:
                    # flow optional – fallback silent
                    motion = None

            # budget
            B_target = int(np.clip(int(np.ceil(args.budget_ratio * T)), args.Bmin, args.Bmax))

            # to torch
            x = torch.from_numpy(feats).unsqueeze(0).to(device)  # (1,T,D)
            
            # RAFT motion to torch (if available)
            motion_feats = None
            if motion_feats_np is not None:
                motion_feats = torch.from_numpy(motion_feats_np.astype(np.float32)).unsqueeze(0).to(device)  # (1,T,D_m)
            
            # Forward pass (different for baseline vs advanced)
            if args.model_type == "baseline":
                h = enc(x)                    # (1,T,H)
                probs = pol(h)                # (1,T) in (0,1)
            else:  # advanced
                # Extract scene_id for caching
                scene_id = str(scene_dir).replace('/', '_')
                probs = model(x, scene_id=scene_id, motion_feats=motion_feats)  # (1,T)
            
            probs = torch.clamp(probs, 1e-6, 1-1e-6)

            # sample actions + stats
            actions, logp_t, ent_t = bernoulli_sample(probs)     # (1,T)
            acts = actions.squeeze(0)                            # (T,)
            log_probs = logp_t.sum(dim=1)                        # (1,)
            entropy = ent_t.sum(dim=1)                           # (1,)

            sel_idx = (acts == 1).nonzero(as_tuple=False).squeeze(-1).cpu().numpy().tolist()

            # compute reward
            R = reward_combo(
                feats_all=feats,
                sel_idx=sel_idx,
                frames_all=frames,
                motion=motion,
                w_div=args.w_div, w_rep=args.w_rep, w_rec=args.w_rec,
                w_fd=args.w_fd, w_ms=args.w_ms, w_motion=args.w_motion,
                ms_swd_scales=args.ms_swd_scales, ms_swd_dirs=args.ms_swd_dirs,
                use_lpips_div=bool(args.use_lpips_div),
                lpips_net=args.lpips_net,
                lpips_device=args.lpips_device,
            )

            # budget penalty
            if B_target > 0:
                over = max(0, len(sel_idx) - B_target)
                under = max(0, B_target - len(sel_idx))
                R -= args.budget_penalty * (over + 0.5 * under)

            R_t = torch.tensor(R, dtype=torch.float32, device=device)

            # moving baseline
            if baseline is None:
                baseline = R
            else:
                baseline = beta * baseline + (1 - beta) * R
            b_t = torch.tensor(baseline, dtype=torch.float32, device=device)

            advantage = R_t - b_t  # scalar
            loss = - advantage * log_probs - args.entropy_coef * entropy  # (1,)
            loss = loss.mean()

            opt.zero_grad()
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                if args.model_type == "baseline":
                    torch.nn.utils.clip_grad_norm_(
                        list(enc.parameters()) + list(pol.parameters()),
                        args.max_grad_norm
                    )
                else:  # advanced
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm
                    )
            opt.step()

            # logging accumulators
            ep_rewards.append(R)
            sel_count_sum += len(sel_idx)
            frame_count_sum += T
            entropy_sum += float(entropy.item())
            mean_prob_sum += float(probs.mean().item())
            budget_gap_sum += abs(len(sel_idx) - B_target)
            
            # Collect for visualizations
            all_probs.append(probs.squeeze(0).detach().cpu().numpy())
            all_rewards.append(R)
            
            # Collect sample frames for visualization (first scene of epoch)
            if len(sample_frames_selected) == 0 and frames is not None and len(sel_idx) > 0:
                # Get up to 8 selected frames
                for idx in sel_idx[:8]:
                    if idx < len(frames):
                        sample_frames_selected.append(frames[idx])
                # Get up to 8 rejected frames
                rejected_idx = [i for i in range(T) if i not in sel_idx]
                for idx in rejected_idx[:8]:
                    if idx < len(frames):
                        sample_frames_rejected.append(frames[idx])

            # Update scene progress bar with current stats
            scene_pbar.set_postfix({
                'R': f'{R:.3f}',
                'sel': f'{len(sel_idx)}/{T}',
                'prob': f'{probs.mean().item():.3f}'
            })
            
            # Log per-step metrics (every 10 steps to avoid overhead)
            if global_step % 10 == 0:
                writer.add_scalar('step/reward', R, global_step)
                writer.add_scalar('step/selection_count', len(sel_idx), global_step)
                writer.add_scalar('step/mean_prob', probs.mean().item(), global_step)
                writer.add_scalar('step/entropy', entropy.item(), global_step)

            global_step += 1

        # epoch summary
        meanR = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        sel_ratio = (sel_count_sum / frame_count_sum) if frame_count_sum > 0 else 0.0
        mean_entropy = entropy_sum / max(1, len(scene_dirs))
        mean_prob = mean_prob_sum / max(1, len(scene_dirs))
        mean_budget_gap = budget_gap_sum / max(1, len(scene_dirs))

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'meanR': f'{meanR:.4f}',
            'sel_ratio': f'{sel_ratio:.4f}',
            'entropy': f'{mean_entropy:.4f}'
        })

        tqdm.write(f"[Epoch {epoch}] "
                   f"meanR={meanR:.4f} | sel_ratio={sel_ratio:.4f} | "
                   f"entropy={mean_entropy:.4f} | mean_prob={mean_prob:.4f} | "
                   f"budget_gap={mean_budget_gap:.3f}")

        writer.add_scalar("train/mean_reward", meanR, epoch)
        writer.add_scalar("train/sel_ratio", sel_ratio, epoch)
        writer.add_scalar("train/entropy", mean_entropy, epoch)
        writer.add_scalar("train/mean_prob", mean_prob, epoch)
        writer.add_scalar("train/budget_gap", mean_budget_gap, epoch)
        
        # Add reward statistics
        if ep_rewards:
            writer.add_scalar("train/reward_std", float(np.std(ep_rewards)), epoch)
            writer.add_scalar("train/reward_min", float(np.min(ep_rewards)), epoch)
            writer.add_scalar("train/reward_max", float(np.max(ep_rewards)), epoch)
        
        # Add probability histogram
        if all_probs:
            all_probs_concat = np.concatenate(all_probs)
            writer.add_histogram('train/prob_distribution', all_probs_concat, epoch)
            writer.add_scalar('train/prob_std', float(np.std(all_probs_concat)), epoch)
        
        # Add reward distribution
        if all_rewards:
            writer.add_histogram('train/reward_distribution', np.array(all_rewards), epoch)
        
        # Visualize sample frames (selected vs rejected)
        if sample_frames_selected:
            # Create grid of selected frames
            selected_grid = []
            for frm in sample_frames_selected[:8]:
                # Resize to small size for visualization
                frm_small = cv2.resize(frm, (160, 90))
                # Convert BGR to RGB
                frm_rgb = cv2.cvtColor(frm_small, cv2.COLOR_BGR2RGB)
                selected_grid.append(frm_rgb)
            
            if selected_grid:
                # Stack into grid (2 rows x 4 cols)
                grid_tensor = torch.from_numpy(np.array(selected_grid)).permute(0, 3, 1, 2).float() / 255.0
                writer.add_images('frames/selected_keyframes', grid_tensor, epoch, dataformats='NCHW')
        
        if sample_frames_rejected:
            rejected_grid = []
            for frm in sample_frames_rejected[:8]:
                frm_small = cv2.resize(frm, (160, 90))
                frm_rgb = cv2.cvtColor(frm_small, cv2.COLOR_BGR2RGB)
                rejected_grid.append(frm_rgb)
            
            if rejected_grid:
                grid_tensor = torch.from_numpy(np.array(rejected_grid)).permute(0, 3, 1, 2).float() / 255.0
                writer.add_images('frames/rejected_frames', grid_tensor, epoch, dataformats='NCHW')
        
        # Log cache statistics for advanced model
        if args.model_type == "advanced" and args.use_cache:
            cache_stats = model.get_cache_stats()
            if cache_stats:
                writer.add_scalar('cache/hit_rate', cache_stats['hit_rate'], epoch)
                writer.add_scalar('cache/hits', cache_stats['hits'], epoch)
                writer.add_scalar('cache/misses', cache_stats['misses'], epoch)
                writer.add_scalar('cache/size', cache_stats['cache_size'], epoch)
                tqdm.write(f"  Cache: hit_rate={cache_stats['hit_rate']:.2%} "
                          f"({cache_stats['hits']}/{cache_stats['total_queries']} hits)")

        # save ckpt per epoch
        ckpt_path = save_dir / f"dsn_checkpoint_ep{epoch}.pt"
        if args.model_type == "baseline":
            torch.save({"encoder": enc.state_dict(), "policy": pol.state_dict()}, ckpt_path)
        else:  # advanced
            torch.save({
                "model": model.state_dict(),
                "config": config,
                "model_type": "advanced"
            }, ckpt_path)

        # validate per epoch
        if args.val_videos_dir and args.val_output_dir and (epoch % args.validate_every == 0):
            val_out = Path(args.val_output_dir) / f"ep{epoch}"
            val_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python", "-m", "eval.batch_eval",
                "--videos_dir", args.val_videos_dir,
                "--output_dir", str(val_out),
                "--checkpoint", str(ckpt_path),
                "--device", eval_device,  # Add device for DSN inference
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
                "--eval_device", eval_device,  # Device for evaluation metrics
            ]
            # Detector-specific
            if args.eval_threshold is not None:
                cmd += ["--threshold", str(args.eval_threshold)]
            if args.eval_model_dir:
                cmd += ["--model_dir", args.eval_model_dir]
            if args.eval_weights_path:
                cmd += ["--weights_path", args.eval_weights_path]
            if args.eval_prob_threshold is not None:
                cmd += ["--prob_threshold", str(args.eval_prob_threshold)]
            if args.eval_scene_device:
                cmd += ["--scene_device", args.eval_scene_device]
            # Baseline switch
            if args.eval_with_baselines:
                cmd.append("--with_baselines")
            # Optional batch controls
            if args.eval_max_videos is not None:
                cmd += ["--max_videos", str(args.eval_max_videos)]
            if args.eval_num_workers is not None:
                cmd += ["--num_workers", str(args.eval_num_workers)]

            tqdm.write("[Validate] Running batch_eval...")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                tqdm.write(f"[Validate][Error] {r.stderr}")
            else:
                summary_path = val_out / "summary_results.json"
                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    agg = s.get("aggregate_metrics", {})
                    # log a few reliable metrics
                    for k, v in agg.items():
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            writer.add_scalar(f"val/{k}", float(v), epoch)
                    # track best (example: minimize RecErr_mean; tie-break by Frechet_mean)
                    rec_mean = agg.get("RecErr_mean", None)
                    if isinstance(rec_mean, (int, float)) and not np.isnan(rec_mean):
                        if (best_metric is None) or (rec_mean < best_metric):
                            best_metric = rec_mean
                            best_ckpt_path = str(ckpt_path)

    # end epochs
    epoch_pbar.close()
    writer.close()
    
    # Save final checkpoint
    if args.model_type == "baseline":
        torch.save({"encoder": enc.state_dict(), "policy": pol.state_dict()}, save_dir / "dsn_checkpoint.pt")
    else:  # advanced
        torch.save({
            "model": model.state_dict(),
            "config": config,
            "model_type": "advanced"
        }, save_dir / "dsn_checkpoint.pt")
    
    if best_ckpt_path:
        tqdm.write(f"\n✅ Best checkpoint by RecErr_mean: {best_ckpt_path}")
    tqdm.write("\n🎉 Training done.")

if __name__ == "__main__":
    main()


"""
python -m src.pipeline.train_rl_dsn \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/dsn_baseline_v1_use_motion \
  --epochs 20 \
  --device cuda:0 \
  --feat_dim 512 --enc_hidden 256 --lstm_hidden 128 \
  --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
  --w_div 1.0 --w_rep 1.0 --w_rec 0.5 --w_fd 0.2 --w_ms 0.2 --w_motion 0.2 \
  --use_motion 1 --ms_swd_scales 3 --ms_swd_dirs 16 \
  --val_videos_dir /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga \
  --val_output_dir outputs/val_runs/base_v1 \
  --validate_every 1 \
  --eval_embedder clip_vitb32 \
  --eval_backend pyscenedetect --eval_sample_stride 5 --eval_resize_w 320 --eval_resize_h 180 \
  --eval_with_baselines
"""