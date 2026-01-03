#!/usr/bin/env python3
"""
Precompute RAFT motion features for all scenes in the dataset.

This script:
1. Loads RAFT-small model from repos/RAFT
2. Processes all scenes in the dataset
3. Computes optical flow between consecutive frames
4. Extracts motion features via spatial pooling
5. Saves as motion_raft.npy (T, D_m) per scene

Usage:
    python scripts/precompute_raft_motion.py \
        --dataset_root data/sakuga_dataset_100_samples \
        --raft_model repos/RAFT/models/raft-small.pth \
        --device cuda \
        --motion_dim 128
"""

import sys
import os
from pathlib import Path

# Add RAFT core to path
RAFT_PATH = Path(__file__).parent.parent / "repos" / "RAFT" / "core"
sys.path.insert(0, str(RAFT_PATH))

import argparse
import glob
import json
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2

from raft import RAFT
from utils.utils import InputPadder


class MotionFeatureExtractor(nn.Module):
    """Extract compact motion features from RAFT flow fields."""
    
    def __init__(self, motion_dim: int = 128):
        super().__init__()
        self.motion_dim = motion_dim
        
        # Spatial pooling layers
        # Flow is (H, W, 2), we'll pool to (motion_dim,)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, motion_dim)
        
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow: (B, 2, H, W) optical flow
        Returns:
            features: (B, motion_dim)
        """
        x = F.relu(self.conv1(flow))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def load_raft_model(model_path: str, device: torch.device, small: bool = True):
    """Load RAFT model from checkpoint."""
    
    # Create args namespace for RAFT (must support 'in' operator)
    class Args:
        def __init__(self):
            self.small = small
            self.mixed_precision = False
            self.alternate_corr = False
            self.dropout = 0
        
        def __contains__(self, key):
            """Support 'in' operator for RAFT compatibility."""
            return hasattr(self, key)
    
    args = Args()
    model = RAFT(args)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model


def load_frames_from_scene(scene_dir: Path) -> List[np.ndarray]:
    """Load all frames from a scene directory."""
    frames_dir = scene_dir / "frames"
    if not frames_dir.exists():
        return []
    
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    frames = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(img)
    
    return frames


def frames_to_tensor(frames: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert BGR frames to RGB tensor (B, 3, H, W)."""
    tensors = []
    for frame in frames:
        # BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To tensor (H, W, 3) -> (3, H, W)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensors.append(tensor)
    
    # Stack to (B, 3, H, W)
    batch = torch.stack(tensors, dim=0).to(device)
    return batch


def compute_motion_features(
    raft_model: RAFT,
    motion_extractor: MotionFeatureExtractor,
    frames: List[np.ndarray],
    device: torch.device
) -> np.ndarray:
    """
    Compute motion features for a sequence of frames.
    
    Args:
        raft_model: RAFT model
        motion_extractor: Motion feature extractor
        frames: List of BGR frames
        device: torch device
    
    Returns:
        motion_feats: (T, motion_dim) motion features
    """
    T = len(frames)
    if T < 2:
        # Return zeros for single frame
        return np.zeros((T, motion_extractor.motion_dim), dtype=np.float32)
    
    motion_feats = []
    
    with torch.no_grad():
        for t in range(T - 1):
            # Load consecutive frames
            img1 = frames_to_tensor([frames[t]], device)
            img2 = frames_to_tensor([frames[t + 1]], device)
            
            # Pad to multiple of 8
            padder = InputPadder(img1.shape)
            img1_pad, img2_pad = padder.pad(img1, img2)
            
            # Compute flow
            _, flow_up = raft_model(img1_pad, img2_pad, iters=20, test_mode=True)
            
            # Unpad if needed
            flow = flow_up[:, :, :img1.shape[2], :img1.shape[3]]
            
            # Extract motion features
            feat = motion_extractor(flow)  # (1, motion_dim)
            motion_feats.append(feat.cpu().numpy())
        
        # Duplicate last feature for final frame
        motion_feats.append(motion_feats[-1])
    
    # Stack to (T, motion_dim)
    motion_feats = np.concatenate(motion_feats, axis=0)
    
    return motion_feats


def process_scene(
    scene_dir: Path,
    raft_model: RAFT,
    motion_extractor: MotionFeatureExtractor,
    device: torch.device,
    force: bool = False
) -> bool:
    """
    Process a single scene directory.
    
    Returns:
        True if processed, False if skipped
    """
    output_path = scene_dir / "motion_raft.npy"
    
    # Skip if already exists
    if output_path.exists() and not force:
        return False
    
    # Load frames
    frames = load_frames_from_scene(scene_dir)
    if len(frames) == 0:
        return False
    
    # Compute motion features
    motion_feats = compute_motion_features(raft_model, motion_extractor, frames, device)
    
    # Save
    np.save(output_path, motion_feats)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Precompute RAFT motion features")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of dataset (e.g., data/sakuga_dataset_100_samples)")
    parser.add_argument("--raft_model", type=str, default="repos/RAFT/models/raft-small.pth",
                        help="Path to RAFT model checkpoint")
    parser.add_argument("--motion_dim", type=int, default=128,
                        help="Motion feature dimension")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute even if motion_raft.npy exists")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum number of videos to process (for testing)")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load RAFT model
    print(f"Loading RAFT model from {args.raft_model}...")
    raft_model = load_raft_model(args.raft_model, device, small=True)
    print("RAFT model loaded successfully")
    
    # Create motion feature extractor
    motion_extractor = MotionFeatureExtractor(motion_dim=args.motion_dim).to(device)
    motion_extractor.eval()
    
    # Find all scene directories
    dataset_root = Path(args.dataset_root)
    video_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    
    if args.max_videos:
        video_dirs = video_dirs[:args.max_videos]
    
    print(f"Found {len(video_dirs)} video directories")
    
    # Process all scenes
    total_scenes = 0
    processed_scenes = 0
    
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        scene_dirs = sorted(video_dir.glob("scene_*"))
        
        for scene_dir in tqdm(scene_dirs, desc=f"  {video_dir.name}", leave=False):
            total_scenes += 1
            if process_scene(scene_dir, raft_model, motion_extractor, device, args.force):
                processed_scenes += 1
    
    print(f"\nDone! Processed {processed_scenes}/{total_scenes} scenes")
    print(f"Skipped {total_scenes - processed_scenes} scenes (already exist)")


if __name__ == "__main__":
    main()
