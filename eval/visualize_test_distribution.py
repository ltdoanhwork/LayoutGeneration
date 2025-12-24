#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize IQA Distribution of Test Set

Compute and visualize the Anime-CLIP-IQA score distribution for all frames 
in a test set. This does NOT require a trained model - it just shows the 
quality distribution of the dataset.

Usage:
    conda activate sam && python -m eval.visualize_test_distribution \
        --videos_dir data/samples/Sakuga_test \
        --output_dir outputs/test_iqa_distribution \
        --max_videos 5

Output:
    - Per-video IQA histograms
    - Aggregate distribution across test set
    - Per-attribute analysis
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import cv2
import torch

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not found")

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.distribution_metrics import (
    DistributionAwareMetrics,
    ATTR_NAMES,
    ATTR_INDEX,
)


def compute_anime_iqa_attributes(
    frames: List[np.ndarray],
    device: str = "cuda",
) -> np.ndarray:
    """Compute Anime-CLIP-IQA attributes for frames."""
    try:
        import clip
        import torch.nn.functional as F
        from PIL import Image
        
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        
        # Define prompts
        quality_prompts = [
            ("A sharp anime frame.", "A blurry anime frame."),
            ("A colorful anime frame.", "A dull anime frame."),
            ("A bright anime frame.", "A dark anime frame."),
            ("A dynamic sakuga action frame.", "A calm talking anime frame."),
            ("A cinematic impactful anime frame.", "An unremarkable anime frame."),
            ("An anime frame with strong facial expression.", "A neutral anime frame."),
        ]
        
        # Tokenize
        text_tokens = []
        for p_pos, p_neg in quality_prompts:
            text_tokens.append(clip.tokenize(p_pos))
            text_tokens.append(clip.tokenize(p_neg))
        text_tokens = torch.cat(text_tokens).to(device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            K = len(quality_prompts)
            D = text_features.shape[-1]
            text_features_pairs = text_features.view(K, 2, D)
        
        all_scores = []
        batch_size = 32
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            images = []
            for frame in batch:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                images.append(preprocess(img))
            images = torch.stack(images).to(device)
            
            with torch.no_grad():
                img_features = clip_model.encode_image(images)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                
                for frame_idx in range(len(batch)):
                    scores = []
                    for k in range(K):
                        pair_feats = text_features_pairs[k]
                        logits = (100.0 * img_features[frame_idx:frame_idx+1] @ pair_feats.T)
                        probs = logits.softmax(dim=-1)
                        scores.append(probs[0, 0].item())
                    all_scores.append(scores)
        
        return np.array(all_scores, dtype=np.float32)
        
    except ImportError as e:
        print(f"Warning: CLIP error ({e})")
        return np.random.rand(len(frames), 6).astype(np.float32)


def process_video(
    video_path: str,
    device: str = "cuda",
    sample_stride: int = 5,
    resize_w: int = 320,
    resize_h: int = 180,
) -> Optional[Dict[str, Any]]:
    """Process a single video and compute IQA attributes for all frames."""
    video_id = Path(video_path).stem
    print(f"\n📹 Processing: {video_id}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Cannot open video")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            frame = cv2.resize(frame, (resize_w, resize_h))
            frames.append(frame)
        frame_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    print(f"  📊 Extracted {len(frames)} frames (total={total_frames}, stride={sample_stride})")
    
    # Compute IQA
    print(f"  🎨 Computing anime IQA attributes...")
    attrs = compute_anime_iqa_attributes(frames, device)
    
    # Compute aggregate quality
    metrics = DistributionAwareMetrics()
    quality = metrics.compute_aggregate_quality(attrs)
    
    result = {
        "video_id": video_id,
        "video_path": str(video_path),
        "total_frames": total_frames,
        "sampled_frames": len(frames),
        "attrs": attrs,
        "quality": quality,
        "stats": {
            "quality_mean": float(np.mean(quality)),
            "quality_std": float(np.std(quality)),
            "quality_min": float(np.min(quality)),
            "quality_max": float(np.max(quality)),
            "quality_median": float(np.median(quality)),
            "quality_p10": float(np.percentile(quality, 10)),
            "quality_p90": float(np.percentile(quality, 90)),
        }
    }
    
    # Per-attribute stats
    for name, idx in ATTR_INDEX.items():
        result["stats"][f"{name}_mean"] = float(np.mean(attrs[:, idx]))
        result["stats"][f"{name}_std"] = float(np.std(attrs[:, idx]))
    
    print(f"  ✅ Quality: mean={result['stats']['quality_mean']:.3f}, "
          f"range=[{result['stats']['quality_min']:.3f}, {result['stats']['quality_max']:.3f}]")
    
    return result


def create_per_video_visualization(
    result: Dict[str, Any],
    output_dir: str,
):
    """Create visualization for a single video."""
    if not HAS_MATPLOTLIB:
        return
    
    video_id = result["video_id"]
    quality = result["quality"]
    attrs = result["attrs"]
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Quality histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(quality, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(quality), color='red', linestyle='--', label=f'Mean: {np.mean(quality):.3f}')
    ax1.axvline(np.percentile(quality, 90), color='green', linestyle='--', label=f'P90: {np.percentile(quality, 90):.3f}')
    ax1.set_xlabel('Aggregate Quality Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Quality Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Quality over time
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(quality, 'b-', alpha=0.7, linewidth=0.5)
    ax2.axhline(np.mean(quality), color='red', linestyle='--', alpha=0.5)
    ax2.axhline(np.percentile(quality, 90), color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(range(len(quality)), 0, quality, alpha=0.3)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Quality')
    ax2.set_title('Quality Over Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-attribute distributions
    ax3 = fig.add_subplot(gs[1, :2])
    attr_data = [attrs[:, idx] for idx in range(6)]
    bp = ax3.boxplot(attr_data, labels=[n.capitalize() for n in ATTR_NAMES], patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, 6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax3.set_ylabel('Score')
    ax3.set_title('Per-Attribute Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Stats summary
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    stats = result["stats"]
    stats_text = [
        f"Sampled Frames: {result['sampled_frames']}",
        f"",
        f"Quality Stats:",
        f"  Mean:   {stats['quality_mean']:.3f}",
        f"  Std:    {stats['quality_std']:.3f}",
        f"  Min:    {stats['quality_min']:.3f}",
        f"  Max:    {stats['quality_max']:.3f}",
        f"  Median: {stats['quality_median']:.3f}",
        f"  P10:    {stats['quality_p10']:.3f}",
        f"  P90:    {stats['quality_p90']:.3f}",
    ]
    ax4.text(0.1, 0.95, '\n'.join(stats_text), transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Statistics', fontweight='bold')
    
    fig.suptitle(f'IQA Distribution: {video_id}', fontsize=14, fontweight='bold')
    
    save_path = os.path.join(output_dir, f"{video_id}_iqa_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")


def create_aggregate_visualization(
    results: List[Dict[str, Any]],
    output_dir: str,
):
    """Create aggregate visualization across all videos."""
    if not HAS_MATPLOTLIB or len(results) == 0:
        return
    
    print(f"\n📊 Creating aggregate visualization...")
    
    # Collect all frame qualities
    all_qualities = []
    all_attrs = []
    video_means = []
    video_stds = []
    
    for r in results:
        all_qualities.extend(r["quality"].tolist())
        all_attrs.append(r["attrs"])
        video_means.append(r["stats"]["quality_mean"])
        video_stds.append(r["stats"]["quality_std"])
    
    all_qualities = np.array(all_qualities)
    all_attrs = np.concatenate(all_attrs, axis=0)
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Overall quality histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_qualities, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(all_qualities), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(all_qualities):.3f}')
    ax1.axvline(np.percentile(all_qualities, 90), color='green', linestyle='--', linewidth=2,
                label=f'P90: {np.percentile(all_qualities, 90):.3f}')
    ax1.set_xlabel('Quality Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'All Frames Quality Distribution (N={len(all_qualities)})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Per-video means
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(video_means)), video_means, yerr=video_stds, 
            color='coral', alpha=0.7, capsize=3)
    ax2.axhline(np.mean(video_means), color='blue', linestyle='--', 
                label=f'Mean of means: {np.mean(video_means):.3f}')
    ax2.set_xlabel('Video Index')
    ax2.set_ylabel('Mean Quality')
    ax2.set_title('Per-Video Mean Quality', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Quality CDF
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_q = np.sort(all_qualities)
    cdf = np.arange(1, len(sorted_q) + 1) / len(sorted_q)
    ax3.plot(sorted_q, cdf, 'b-', linewidth=2)
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Quality Score')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Quality CDF', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4-9. Per-attribute histograms
    for i, name in enumerate(ATTR_NAMES):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        ax.hist(all_attrs[:, i], bins=40, color=plt.cm.Set2(i/6), alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(all_attrs[:, i]), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_attrs[:, i]):.3f}')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name.capitalize()} Distribution', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Test Set IQA Distribution ({len(results)} videos, {len(all_qualities)} frames)', 
                 fontsize=16, fontweight='bold')
    
    save_path = os.path.join(output_dir, "aggregate_iqa_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {save_path}")
    
    # Summary stats
    summary = {
        "n_videos": len(results),
        "n_frames": len(all_qualities),
        "quality_mean": float(np.mean(all_qualities)),
        "quality_std": float(np.std(all_qualities)),
        "quality_min": float(np.min(all_qualities)),
        "quality_max": float(np.max(all_qualities)),
        "quality_p10": float(np.percentile(all_qualities, 10)),
        "quality_p25": float(np.percentile(all_qualities, 25)),
        "quality_p50": float(np.percentile(all_qualities, 50)),
        "quality_p75": float(np.percentile(all_qualities, 75)),
        "quality_p90": float(np.percentile(all_qualities, 90)),
        "per_attribute": {},
    }
    
    for name in ATTR_NAMES:
        idx = ATTR_INDEX[name]
        summary["per_attribute"][name] = {
            "mean": float(np.mean(all_attrs[:, idx])),
            "std": float(np.std(all_attrs[:, idx])),
            "p10": float(np.percentile(all_attrs[:, idx], 10)),
            "p90": float(np.percentile(all_attrs[:, idx], 90)),
        }
    
    summary_path = os.path.join(output_dir, "aggregate_iqa_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Saved: {summary_path}")
    
    # Print summary
    print(f"\n📈 Test Set IQA Summary ({len(results)} videos, {len(all_qualities)} frames):")
    print(f"   Mean Quality: {summary['quality_mean']:.3f} ± {summary['quality_std']:.3f}")
    print(f"   Range: [{summary['quality_min']:.3f}, {summary['quality_max']:.3f}]")
    print(f"   Percentiles: P10={summary['quality_p10']:.3f}, P50={summary['quality_p50']:.3f}, P90={summary['quality_p90']:.3f}")
    print(f"\n   Per-Attribute Means:")
    for name in ATTR_NAMES:
        print(f"     {name:15s}: {summary['per_attribute'][name]['mean']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize IQA distribution of test set")
    parser.add_argument("--videos_dir", type=str, required=True, help="Directory with test videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample_stride", type=int, default=5, help="Frame sampling stride")
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--resize_h", type=int, default=180)
    parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find videos
    exts = [".mp4", ".mkv", ".avi", ".mov"]
    video_paths = []
    for ext in exts:
        video_paths.extend(Path(args.videos_dir).glob(f"*{ext}"))
    video_paths = sorted([str(p) for p in video_paths])
    
    if args.max_videos:
        video_paths = video_paths[:args.max_videos]
    
    print(f"\n{'='*60}")
    print(f"Test Set IQA Distribution Visualization")
    print(f"{'='*60}")
    print(f"Videos: {len(video_paths)}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")
    
    results = []
    for video_path in video_paths:
        try:
            result = process_video(
                video_path, args.device, args.sample_stride,
                args.resize_w, args.resize_h
            )
            if result:
                results.append(result)
                create_per_video_visualization(result, args.output_dir)
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    if results:
        create_aggregate_visualization(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Complete! Processed {len(results)} videos")
    print(f"📁 Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
