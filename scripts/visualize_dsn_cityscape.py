#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize DSN probabilities as a cityscape-style visualization.

The height of each bar represents the selection probability from the DSN model.
Selected keyframes are highlighted in a different color.

Usage:
    python scripts/visualize_dsn_cityscape.py \
        --keyframes outputs/dsn_infer/14652/keyframes.csv \
        --scenes outputs/dsn_infer/14652/scenes.json \
        --output outputs/dsn_infer/14652/cityscape.png
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_keyframes(keyframes_path: str) -> List[Dict[str, Any]]:
    """Load keyframes from CSV file."""
    keyframes = []
    with open(keyframes_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kf_dict = {
                'scene_id': int(row['scene_id']),
                'frame_global': int(row['frame_global']),
                'frame_in_scene': int(row['frame_in_scene']),
                'time': row['time'],
                'prob': float(row['prob']),
            }
            # Check if 'selected' column exists (from all_probs.csv)
            if 'selected' in row:
                kf_dict['selected'] = int(row['selected'])
            keyframes.append(kf_dict)
    return keyframes


def load_scenes(scenes_path: str) -> List[Dict[str, Any]]:
    """Load scenes from JSON file."""
    with open(scenes_path, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    return scenes


def create_cityscape_visualization(
    keyframes: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    output_path: str,
    figsize: tuple = (16, 4),
    selected_color: str = '#4ECDC4',  # Turquoise for selected keyframes
    unselected_color: str = '#95A5A6',  # Gray for non-selected frames
    bar_width_ratio: float = 0.8,
):
    """
    Create a cityscape-style visualization of DSN probabilities.
    
    Args:
        keyframes: List of frame dictionaries with 'frame_global', 'prob', and optionally 'selected'
        scenes: List of scene dictionaries
        output_path: Where to save the visualization
        figsize: Figure size (width, height)
        selected_color: Color for selected keyframes
        unselected_color: Color for non-selected frames
        bar_width_ratio: Width of bars relative to spacing (0-1)
    """
    if not keyframes:
        print("⚠️  No keyframes to visualize")
        return
    
    # Build frame data
    frame_probs = {}
    selected_frames = set()
    
    for kf in keyframes:
        frame_idx = kf['frame_global']
        prob = kf['prob']
        frame_probs[frame_idx] = prob
        
        # Check if this frame is selected (from all_probs.csv)
        if 'selected' in kf and kf['selected'] == 1:
            selected_frames.add(frame_idx)
        # Fallback: if no 'selected' field, assume all loaded frames are selected
        elif 'selected' not in kf:
            selected_frames.add(frame_idx)
    
    # Get frame range
    if scenes:
        min_frame = min(s['start_frame'] for s in scenes)
        max_frame = max(s['end_frame'] for s in scenes)
    else:
        min_frame = min(frame_probs.keys())
        max_frame = max(frame_probs.keys())
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort frames for plotting
    sorted_frames = sorted(frame_probs.keys())
    
    # Calculate bar width based on frame spacing
    if len(sorted_frames) > 1:
        avg_spacing = np.mean(np.diff(sorted_frames))
        bar_width = avg_spacing * bar_width_ratio
    else:
        bar_width = 1.0
    
    # Plot bars
    for frame_idx in sorted_frames:
        prob = frame_probs[frame_idx]
        color = selected_color if frame_idx in selected_frames else unselected_color
        
        # Draw bar
        ax.bar(
            frame_idx,
            prob,
            width=bar_width,
            color=color,
            edgecolor='none',
            alpha=0.85,
        )
    
    # Add scene boundaries as vertical lines
    for scene in scenes:
        ax.axvline(
            x=scene['start_frame'],
            color='#E74C3C',
            linestyle='--',
            linewidth=1.5,
            alpha=0.4,
            zorder=0,
        )
    
    # Styling
    ax.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Selection Probability', fontsize=12, fontweight='bold')
    ax.set_title('DSN Keyframe Selection Probabilities (Cityscape View)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    ax.set_xlim(min_frame - bar_width * 2, max_frame + bar_width * 2)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=selected_color, edgecolor='none', label='Selected Keyframes', alpha=0.85),
        Patch(facecolor=unselected_color, edgecolor='none', label='Non-selected Frames', alpha=0.85),
        plt.Line2D([0], [0], color='#E74C3C', linestyle='--', linewidth=1.5, 
                   label='Scene Boundaries', alpha=0.4),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)
    
    # Add statistics text
    stats_text = (
        f'Total Frames: {len(sorted_frames)}\n'
        f'Selected Keyframes: {len(selected_frames)}\n'
        f'Scenes: {len(scenes)}\n'
        f'Avg Prob: {np.mean(list(frame_probs.values())):.3f}'
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved cityscape visualization to: {output_path}")
    plt.close()


def create_scene_based_cityscape(
    keyframes: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    output_path: str,
    figsize: tuple = (18, 5),
):
    """
    Create a cityscape visualization with separate panels for each scene.
    
    This provides a clearer view of per-scene keyframe selection.
    """
    if not keyframes or not scenes:
        print("⚠️  No keyframes or scenes to visualize")
        return
    
    # Group keyframes by scene
    scene_keyframes = {}
    for kf in keyframes:
        scene_id = kf['scene_id']
        if scene_id not in scene_keyframes:
            scene_keyframes[scene_id] = []
        scene_keyframes[scene_id].append(kf)
    
    # Create subplots
    num_scenes = len(scenes)
    fig, axes = plt.subplots(1, num_scenes, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for scene_idx, scene in enumerate(scenes):
        ax = axes[scene_idx]
        scene_id = scene['scene_id']
        
        # Get keyframes for this scene
        kfs = scene_keyframes.get(scene_id, [])
        
        if not kfs:
            ax.text(0.5, 0.5, 'No keyframes', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Scene {scene_id}', fontsize=10, fontweight='bold')
            continue
        
        # Plot bars
        frames = [kf['frame_in_scene'] for kf in kfs]
        probs = [kf['prob'] for kf in kfs]
        
        bars = ax.bar(
            frames,
            probs,
            color='#4ECDC4',
            edgecolor='none',
            alpha=0.85,
        )
        
        # Styling
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Frame in Scene', fontsize=9)
        if scene_idx == 0:
            ax.set_ylabel('Probability', fontsize=9)
        ax.set_title(f'Scene {scene_id}\n({len(kfs)} keyframes)', 
                    fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_axisbelow(True)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('DSN Keyframe Selection by Scene', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved scene-based cityscape to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DSN probabilities as cityscape"
    )
    parser.add_argument(
        "--keyframes",
        required=True,
        type=str,
        help="Path to keyframes.csv or all_probs.csv from run_dsn_pipeline.py (use all_probs.csv for full visualization)",
    )
    parser.add_argument(
        "--scenes",
        required=True,
        type=str,
        help="Path to scenes.json from run_dsn_pipeline.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization (default: same dir as keyframes)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="combined",
        choices=["combined", "per_scene", "both"],
        help="Visualization style: combined (all scenes), per_scene (separate panels), or both",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=16,
        help="Figure width in inches (default: 16)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=4,
        help="Figure height in inches (default: 4)",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading keyframes from {args.keyframes}...")
    keyframes = load_keyframes(args.keyframes)
    
    print(f"Loading scenes from {args.scenes}...")
    scenes = load_scenes(args.scenes)
    
    print(f"✅ Loaded {len(keyframes)} keyframes across {len(scenes)} scenes")
    
    # Determine output path
    if args.output is None:
        keyframes_path = Path(args.keyframes)
        output_dir = keyframes_path.parent
        output_base = output_dir / "cityscape"
    else:
        output_base = Path(args.output).with_suffix('')
    
    # Create visualizations
    if args.style in ["combined", "both"]:
        output_path = str(output_base) + "_combined.png"
        print(f"\n📊 Creating combined cityscape visualization...")
        create_cityscape_visualization(
            keyframes,
            scenes,
            output_path,
            figsize=(args.width, args.height),
        )
    
    if args.style in ["per_scene", "both"]:
        output_path = str(output_base) + "_per_scene.png"
        print(f"\n📊 Creating per-scene cityscape visualization...")
        create_scene_based_cityscape(
            keyframes,
            scenes,
            output_path,
            figsize=(args.width, args.height),
        )
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()

"""
# Combined view (tất cả scenes trong 1 hình)
python scripts/visualize_dsn_cityscape.py \
    --keyframes outputs/dsn_infer/14652/keyframes.csv \
    --scenes outputs/dsn_infer/14652/scenes.json \
    --output outputs/dsn_infer/14652/cityscape.png \
    --style combined

# Per-scene view (mỗi scene 1 panel riêng)
python scripts/visualize_dsn_cityscape.py \
    --keyframes outputs/dsn_infer/14652/keyframes.csv \
    --scenes outputs/dsn_infer/14652/scenes.json \
    --style per_scene

# Tạo cả 2 loại
python scripts/visualize_dsn_cityscape.py \
    --keyframes /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_advanced_v1_no_motion_100_samples/ep1/pipeline_results/v12/keyframes.csv \
    --scenes /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_advanced_v1_no_motion_100_samples/ep1/pipeline_results/v12/scenes.json \
    --style both
"""