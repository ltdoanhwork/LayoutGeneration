#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Distribution Visualizer

Load a checkpoint, run inference on video(s), and visualize the
quality distribution of selected keyframes.

This script provides:
1. Load DSN checkpoint (V8 or other versions)
2. Run keyframe selection on input video(s)
3. Compute distribution-aware metrics
4. Generate comprehensive visualizations

Usage:
    python -m eval.inference_distribution \
        --checkpoint /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v7_dual_objective/dsn_v7_ep75.pt \
        --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test/70025.mp4 \
        --output_dir outputs/distribution_viz

    # For multiple videos:
    python -m eval.inference_distribution \
        --checkpoint runs/dsn_v8/best.pt \
        --videos_dir data/samples/Sakuga \
        --output_dir runs/dsn_v8/distribution_viz

"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import cv2
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.distribution_metrics import (
    DistributionAwareMetrics,
    compute_distribution_metrics_for_eval,
    ATTR_NAMES,
)
from eval.visualize_distribution import (
    create_comprehensive_dashboard,
    create_quality_histogram,
    create_selection_timeline,
    visualize_aggregate_distribution,
    HAS_MATPLOTLIB,
)


def load_checkpoint_and_model(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load DSN checkpoint and instantiate model.
    
    Supports V8 and earlier checkpoint formats.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Determine checkpoint format
    if "config" in ckpt:
        config = ckpt["config"]
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        # Try to infer config from state dict
        config = {}
    else:
        # Assume it's a direct state dict
        state_dict = ckpt
        config = {}
    
    # Determine model version from config or checkpoint keys
    version = config.get("version", "unknown")
    
    # Get model parameters from config
    feat_dim = config.get("feat_dim", 512)
    enc_hidden = config.get("enc_hidden", 256)
    lstm_hidden = config.get("lstm_hidden", 128)
    use_anime_attrs = config.get("use_anime_attrs", 0)
    anime_attrs_dim = config.get("anime_attrs_dim", 6)
    use_raft_motion = config.get("use_raft_motion", 0)
    motion_dim = config.get("motion_dim", 128)
    
    # IMPORTANT: Detect actual feat_dim from checkpoint weights
    # This handles cases where config doesn't properly store use_anime_attrs/use_raft_motion
    actual_feat_dim = None
    if isinstance(state_dict, dict):
        # V8 models have input_proj.0.weight with shape [hidden_dim, feat_dim]
        if "input_proj.0.weight" in state_dict:
            actual_feat_dim = state_dict["input_proj.0.weight"].shape[1]
        # Multitask models may have encoder.fc1.weight
        elif "encoder.fc1.weight" in state_dict:
            actual_feat_dim = state_dict["encoder.fc1.weight"].shape[1]
    
    # Calculate expected feat_dim from config
    expected_feat_dim = feat_dim
    if use_anime_attrs:
        expected_feat_dim += anime_attrs_dim
    if use_raft_motion:
        expected_feat_dim += motion_dim
    
    # Use actual feat_dim from weights if available and different from expected
    if actual_feat_dim is not None and actual_feat_dim != expected_feat_dim:
        print(f"  ⚠️ Config says feat_dim={expected_feat_dim}, but checkpoint weights show feat_dim={actual_feat_dim}")
        total_feat_dim = actual_feat_dim
        
        # Infer what features were used based on feat_dim
        if actual_feat_dim == 646:  # 512 + 6 + 128
            use_anime_attrs = 1
            use_raft_motion = 1
        elif actual_feat_dim == 518:  # 512 + 6
            use_anime_attrs = 1
            use_raft_motion = 0
        elif actual_feat_dim == 640:  # 512 + 128
            use_anime_attrs = 0
            use_raft_motion = 1
        else:
            # Just use what we found
            pass
    else:
        total_feat_dim = expected_feat_dim
    
    # Store computed info in config for later use
    config["_total_feat_dim"] = total_feat_dim
    config["use_anime_attrs"] = use_anime_attrs
    config["anime_attrs_dim"] = anime_attrs_dim
    config["use_raft_motion"] = use_raft_motion
    config["motion_dim"] = motion_dim
    
    print(f"  Config: use_anime_attrs={use_anime_attrs}, use_raft_motion={use_raft_motion}, total_feat_dim={total_feat_dim}")
    
    # Try V8 model first
    try:
        from src.models.dsn_v8 import create_dsn_v8
        model = create_dsn_v8(
            feat_dim=total_feat_dim,
            hidden_dim=enc_hidden,
            lstm_hidden=lstm_hidden,
            use_pcgrad=False,  # Not needed for inference
        )
        # Try to load
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded as V8 model (feat_dim={total_feat_dim})")
    except Exception as e:
        print(f"  V8 load failed: {e}")
        # Fall back to multitask model
        try:
            from src.models.dsn_multitask import create_dsn_multitask
            model = create_dsn_multitask(
                feat_dim=total_feat_dim,
                hidden_dim=enc_hidden,
                lstm_hidden=lstm_hidden,
            )
            model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded as multitask DSN model (feat_dim={total_feat_dim})")
        except Exception as e2:
            print(f"  Multitask load also failed: {e2}")
            raise ValueError(f"Could not load checkpoint: {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    return model, config


def compute_anime_iqa_attributes(
    frames: List[np.ndarray],
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute Anime-CLIP-IQA attributes for frames.
    
    Args:
        frames: List of BGR images
        device: Device for computation
        
    Returns:
        (T, 6) array of IQA scores
    """
    try:
        import clip
        import torch.nn.functional as F
        
        # Load CLIP model
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        
        # Define prompts
        quality_prompts = [
            ("sharp detailed anime frame", "blurry low-quality anime frame"),  # sharpness
            ("vibrant colorful anime scene", "dull faded anime scene"),  # colorfulness
            ("well-lit bright anime scene", "dark poorly-lit anime scene"),  # brightness
            ("dynamic sakuga animation key frame", "static simple animation frame"),  # sakuga
            ("cinematic dramatic anime shot", "plain simple anime shot"),  # cinematic
            ("expressive emotional anime character", "neutral expressionless character"),  # expression
        ]
        
        # Tokenize prompts
        text_features_pos = []
        text_features_neg = []
        for pos, neg in quality_prompts:
            pos_tok = clip.tokenize([pos]).to(device)
            neg_tok = clip.tokenize([neg]).to(device)
            with torch.no_grad():
                text_features_pos.append(clip_model.encode_text(pos_tok))
                text_features_neg.append(clip_model.encode_text(neg_tok))
        
        text_features_pos = torch.cat(text_features_pos, dim=0)  # (6, 512)
        text_features_neg = torch.cat(text_features_neg, dim=0)  # (6, 512)
        text_features_pos = F.normalize(text_features_pos, dim=-1)
        text_features_neg = F.normalize(text_features_neg, dim=-1)
        
        # Process frames
        from PIL import Image
        import torchvision.transforms as T
        
        attrs = []
        batch_size = 32
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            images = []
            for frame in batch_frames:
                # Convert BGR to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                images.append(preprocess(pil_img))
            
            images = torch.stack(images).to(device)
            
            with torch.no_grad():
                img_features = clip_model.encode_image(images)  # (B, 512)
                img_features = F.normalize(img_features, dim=-1)
                
                # Compute scores for each attribute
                batch_attrs = []
                for j in range(6):
                    pos_sim = (img_features @ text_features_pos[j:j+1].T).squeeze(-1)
                    neg_sim = (img_features @ text_features_neg[j:j+1].T).squeeze(-1)
                    # Softmax-normalized score
                    score = F.softmax(torch.stack([neg_sim, pos_sim], dim=-1) * 100, dim=-1)[:, 1]
                    batch_attrs.append(score.cpu().numpy())
                
                batch_attrs = np.stack(batch_attrs, axis=1)  # (B, 6)
                attrs.append(batch_attrs)
        
        return np.concatenate(attrs, axis=0)
        
    except ImportError as e:
        print(f"Warning: CLIP not available ({e}). Using random attributes.")
        return np.random.rand(len(frames), 6).astype(np.float32)


def extract_clip_features(
    frames: List[np.ndarray],
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract CLIP features from frames.
    
    Args:
        frames: List of BGR images
        device: Device for computation
        
    Returns:
        (T, 512) feature array
    """
    try:
        import clip
        import torch.nn.functional as F
        from PIL import Image
        
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        
        features = []
        batch_size = 32
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            images = []
            for frame in batch_frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                images.append(preprocess(pil_img))
            
            images = torch.stack(images).to(device)
            
            with torch.no_grad():
                feats = clip_model.encode_image(images)
                feats = F.normalize(feats, dim=-1)
                features.append(feats.cpu().numpy())
        
        return np.concatenate(features, axis=0)
        
    except ImportError as e:
        print(f"Warning: CLIP not available ({e}). Using random features.")
        return np.random.randn(len(frames), 512).astype(np.float32)


def compute_motion_features(
    frames: List[np.ndarray],
    motion_dim: int = 128,
) -> np.ndarray:
    """
    Compute simple motion features using optical flow.
    
    Uses Farneback optical flow and extracts motion statistics as features.
    This is a simplified version - V7/V8 models use RAFT which is more accurate.
    
    Args:
        frames: List of BGR images
        motion_dim: Output feature dimension
        
    Returns:
        (T, motion_dim) motion feature array
    """
    T = len(frames)
    if T < 2:
        return np.zeros((T, motion_dim), dtype=np.float32)
    
    motion_feats = []
    
    # Convert to grayscale
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    
    for i in range(T):
        if i == 0:
            # First frame: use flow to next frame
            j = 1
        elif i == T - 1:
            # Last frame: use flow from previous frame
            j = T - 2
        else:
            # Middle frames: average of prev and next
            j = i + 1
        
        # Compute optical flow
        prev_gray = gray_frames[min(i, j)]
        next_gray = gray_frames[max(i, j)]
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Extract motion statistics
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Feature extraction: histogram of motion magnitude and angle
        hist_mag, _ = np.histogram(mag.flatten(), bins=32, range=(0, 50))
        hist_ang, _ = np.histogram(ang.flatten(), bins=32, range=(0, 2*np.pi))
        
        # Additional statistics
        stats = np.array([
            np.mean(mag), np.std(mag), np.max(mag),
            np.mean(flow[..., 0]), np.std(flow[..., 0]),
            np.mean(flow[..., 1]), np.std(flow[..., 1]),
        ])
        
        # Combine histogram and stats
        feat = np.concatenate([
            hist_mag.astype(np.float32) / (np.sum(hist_mag) + 1e-6),
            hist_ang.astype(np.float32) / (np.sum(hist_ang) + 1e-6),
            stats.astype(np.float32) / (np.abs(stats).max() + 1e-6)
        ])
        
        # Pad or truncate to motion_dim
        if len(feat) < motion_dim:
            feat = np.pad(feat, (0, motion_dim - len(feat)))
        else:
            feat = feat[:motion_dim]
        
        motion_feats.append(feat)
    
    return np.stack(motion_feats, axis=0).astype(np.float32)


def run_inference(
    model: torch.nn.Module,
    features: np.ndarray,
    anime_attrs: Optional[np.ndarray] = None,
    motion: Optional[np.ndarray] = None,
    budget_ratio: float = 0.06,
    Bmin: int = 3,
    Bmax: int = 15,
    device: str = "cuda",
) -> Tuple[List[int], np.ndarray]:
    """
    Run model inference to select keyframes.
    
    Args:
        model: DSN model
        features: (T, D) feature array
        anime_attrs: Optional (T, 6) anime attributes
        motion: Optional (T,) motion features
        budget_ratio: Selection budget ratio
        Bmin, Bmax: Budget bounds
        device: Device
        
    Returns:
        sel_idx: Selected frame indices
        probs: (T,) selection probabilities
    """
    T = len(features)
    budget = max(Bmin, min(Bmax, int(T * budget_ratio)))
    
    # Concatenate features
    feat_list = [features]
    if anime_attrs is not None:
        feat_list.append(anime_attrs)
    if motion is not None:
        if motion.ndim == 1:
            motion = motion[:, np.newaxis]
        feat_list.append(motion[:T])
    
    concat_features = np.concatenate(feat_list, axis=1)
    
    # Convert to tensor
    feat_tensor = torch.from_numpy(concat_features).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Try different forward patterns
        try:
            # V8 model
            output = model(feat_tensor)
            if isinstance(output, tuple):
                probs = output[0].squeeze(0).cpu().numpy()
            else:
                probs = output.squeeze(0).cpu().numpy()
        except Exception:
            # Fallback
            probs = model.forward(feat_tensor)
            if isinstance(probs, tuple):
                probs = probs[0]
            probs = probs.squeeze(0).cpu().numpy()
    
    # Handle potential 2D output
    if probs.ndim == 2:
        probs = probs[:, -1]  # Take last column
    
    # Select top-K by probability
    probs = np.clip(probs, 0, 1)
    sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
    
    return sel_idx, probs


def process_video(
    video_path: str,
    model: torch.nn.Module,
    config: Dict[str, Any],
    output_dir: str,
    device: str = "cuda",
    sample_stride: int = 5,
    resize_w: int = 320,
    resize_h: int = 180,
    budget_ratio: float = 0.06,
    Bmin: int = 3,
    Bmax: int = 15,
) -> Dict[str, Any]:
    """
    Process a single video: extract features, run inference, compute metrics, visualize.
    
    Args:
        video_path: Path to video file
        model: DSN model
        config: Model config
        output_dir: Output directory
        device: Device
        sample_stride: Frame sampling stride
        resize_w, resize_h: Resize dimensions
        budget_ratio, Bmin, Bmax: Budget settings
        
    Returns:
        Dict with metrics and paths
    """
    video_id = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    print(f"\n📹 Processing: {video_id}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Cannot open video: {video_path}")
        return {"error": "Cannot open video"}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample frames
    frames = []
    frame_indices = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_stride == 0:
            # Resize
            frame = cv2.resize(frame, (resize_w, resize_h))
            frames.append(frame)
            frame_indices.append(frame_idx)
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        print(f"  ❌ No frames extracted")
        return {"error": "No frames"}
    
    print(f"  📊 Extracted {len(frames)} frames (stride={sample_stride})")
    
    # Extract CLIP features
    print(f"  🔮 Extracting CLIP features...")
    features = extract_clip_features(frames, device)
    
    # Compute anime attributes (always computed for distribution analysis)
    print(f"  🎨 Computing anime IQA attributes...")
    anime_attrs = compute_anime_iqa_attributes(frames, device)
    
    # Compute motion features if needed by model config
    motion_feats = None
    use_raft_motion = config.get("use_raft_motion", 0)
    motion_dim = config.get("motion_dim", 128)
    
    if use_raft_motion:
        print(f"  🎬 Computing motion features (dim={motion_dim})...")
        motion_feats = compute_motion_features(frames, motion_dim)
    
    # Run inference
    print(f"  🤖 Running model inference...")
    sel_idx, probs = run_inference(
        model, features, anime_attrs, motion_feats,
        budget_ratio=budget_ratio, Bmin=Bmin, Bmax=Bmax, device=device
    )
    
    print(f"  ✅ Selected {len(sel_idx)} keyframes")
    
    # Compute distribution metrics
    print(f"  📈 Computing distribution metrics...")
    metrics = compute_distribution_metrics_for_eval(anime_attrs, sel_idx)
    
    # Get visualization data
    metrics_computer = DistributionAwareMetrics()
    viz_data = metrics_computer.get_selection_distribution_data(anime_attrs, sel_idx)
    
    # Save distribution data
    dist_data = {
        "video_id": video_id,
        "video_path": str(video_path),
        "total_frames": total_frames,
        "sampled_frames": len(frames),
        "sample_stride": sample_stride,
        "selected_frames": len(sel_idx),
        "frame_indices_selected": sel_idx,
        "probs": probs.tolist(),
        "attrs_all": anime_attrs.tolist(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    dist_json_path = os.path.join(video_output_dir, f"{video_id}_distribution.json")
    with open(dist_json_path, 'w') as f:
        json.dump(dist_data, f, indent=2)
    
    # Generate visualizations
    if HAS_MATPLOTLIB:
        print(f"  📊 Generating visualizations...")
        
        # Comprehensive dashboard
        dashboard_path = os.path.join(video_output_dir, f"{video_id}_dashboard.png")
        create_comprehensive_dashboard(
            anime_attrs, sel_idx, metrics,
            save_path=dashboard_path,
            title=f"Distribution Analysis: {video_id}"
        )
        
        # Quality histogram
        quality = metrics_computer.compute_aggregate_quality(anime_attrs)
        hist_path = os.path.join(video_output_dir, f"{video_id}_histogram.png")
        create_quality_histogram(
            quality, quality[sel_idx], sel_idx,
            save_path=hist_path,
            title=f"Quality Distribution: {video_id}"
        )
        
        # Timeline
        timeline_path = os.path.join(video_output_dir, f"{video_id}_timeline.png")
        create_selection_timeline(
            quality, sel_idx,
            save_path=timeline_path,
            title=f"Selection Timeline: {video_id}"
        )
        
        print(f"  ✅ Saved visualizations to {video_output_dir}")
    
    # Print key metrics
    print(f"\n  📊 Key Metrics:")
    print(f"     Mean Percentile Rank: {metrics['mean_percentile_rank']:.3f}")
    print(f"     Z-Score Improvement:  {metrics['zscore_improvement']:.3f}")
    print(f"     Top-10% Coverage:     {metrics['top_10_coverage']:.1%}")
    print(f"     Above P90 Ratio:      {metrics['above_p90_ratio']:.1%}")
    
    return {
        "video_id": video_id,
        "metrics": metrics,
        "n_selected": len(sel_idx),
        "n_frames": len(frames),
        "output_dir": video_output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference and visualize distribution"
    )
    
    # Input options
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to DSN checkpoint")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to single video file")
    parser.add_argument("--videos_dir", type=str, default=None,
                       help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    
    # Processing options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample_stride", type=int, default=5)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--resize_h", type=int, default=180)
    parser.add_argument("--budget_ratio", type=float, default=0.06)
    parser.add_argument("--Bmin", type=int, default=3)
    parser.add_argument("--Bmax", type=int, default=15)
    parser.add_argument("--max_videos", type=int, default=None,
                       help="Maximum number of videos to process")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.video is None and args.videos_dir is None:
        print("Error: Must specify either --video or --videos_dir")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, config = load_checkpoint_and_model(args.checkpoint, args.device)
    
    # Collect videos to process
    if args.video:
        video_paths = [args.video]
    else:
        exts = [".mp4", ".mkv", ".avi", ".mov"]
        video_paths = []
        for ext in exts:
            video_paths.extend(Path(args.videos_dir).glob(f"*{ext}"))
        video_paths = sorted([str(p) for p in video_paths])
    
    if args.max_videos:
        video_paths = video_paths[:args.max_videos]
    
    print(f"\n{'='*60}")
    print(f"Inference Distribution Visualization")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Videos: {len(video_paths)}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")
    
    # Process videos
    results = []
    for video_path in video_paths:
        try:
            result = process_video(
                video_path, model, config, args.output_dir, args.device,
                args.sample_stride, args.resize_w, args.resize_h,
                args.budget_ratio, args.Bmin, args.Bmax
            )
            if "error" not in result:
                results.append(result)
        except Exception as e:
            print(f"  ❌ Error processing {video_path}: {e}")
    
    # Create aggregate visualization
    if len(results) > 1 and HAS_MATPLOTLIB:
        print(f"\n📊 Creating aggregate visualization...")
        
        # Load all distribution data for aggregate viz
        all_dist_data = []
        for r in results:
            dist_file = os.path.join(r["output_dir"], f"{r['video_id']}_distribution.json")
            if os.path.exists(dist_file):
                with open(dist_file, 'r') as f:
                    all_dist_data.append(json.load(f))
        
        if all_dist_data:
            visualize_aggregate_distribution(
                all_dist_data, args.output_dir, "aggregate"
            )
    
    # Save summary
    summary = {
        "checkpoint": args.checkpoint,
        "n_videos": len(results),
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "aggregate_metrics": {},
    }
    
    if results:
        # Compute aggregate metrics
        mean_percentiles = [r["metrics"]["mean_percentile_rank"] for r in results]
        zscores = [r["metrics"]["zscore_improvement"] for r in results]
        top10_covs = [r["metrics"]["top_10_coverage"] for r in results]
        
        summary["aggregate_metrics"] = {
            "mean_percentile_rank_mean": float(np.mean(mean_percentiles)),
            "mean_percentile_rank_std": float(np.std(mean_percentiles)),
            "zscore_improvement_mean": float(np.mean(zscores)),
            "zscore_improvement_std": float(np.std(zscores)),
            "top_10_coverage_mean": float(np.mean(top10_covs)),
            "top_10_coverage_std": float(np.std(top10_covs)),
        }
    
    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Complete!")
    print(f"{'='*60}")
    print(f"Processed: {len(results)} videos")
    if summary["aggregate_metrics"]:
        agg = summary["aggregate_metrics"]
        print(f"\n📊 Aggregate Metrics:")
        print(f"   Mean Percentile Rank: {agg['mean_percentile_rank_mean']:.3f} ± {agg['mean_percentile_rank_std']:.3f}")
        print(f"   Z-Score Improvement:  {agg['zscore_improvement_mean']:.3f} ± {agg['zscore_improvement_std']:.3f}")
        print(f"   Top-10% Coverage:     {agg['top_10_coverage_mean']:.1%} ± {agg['top_10_coverage_std']:.1%}")
    print(f"\n📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
Usage:
    python -m eval.inference_distribution \
        --checkpoint /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v7_dual_objective/dsn_v7_ep75.pt \
        --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test/70025.mp4 \
        --output_dir outputs/distribution_viz

    python -m eval.inference_distribution \
        --checkpoint /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v8_constrained/best_anime.pt \
        --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test/70025.mp4 \
        --output_dir runs/dsn_v8_constrained/distribution_viz

    # For multiple videos:
    python -m eval.inference_distribution \
        --checkpoint runs/dsn_v8/best.pt \
        --videos_dir data/samples/Sakuga \
        --output_dir runs/dsn_v8/distribution_viz

    python -m eval.inference_distribution \
    --checkpoint /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v8_constrained/best_anime.pt \
    --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test/70025.mp4 \
    --output_dir runs/dsn_v8_constrained/distribution_viz \
    --budget_ratio 0.10 \
    --Bmin 5 \
    --Bmax 20


    python -m eval.inference_distribution \
    --checkpoint /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v8_constrained/best_anime.pt \
    --videos_dir data/samples/Sakuga \
    --output_dir runs/dsn_v8_constrained/distribution_viz \
    --budget_ratio 0.10 \
    --Bmin 5 \
    --Bmax 20
"""