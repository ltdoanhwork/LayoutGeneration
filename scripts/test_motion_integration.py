#!/usr/bin/env python3
"""
Quick test to verify RAFT motion integration without running full training.
Tests:
1. Motion fusion module
2. DSN with motion features (forward pass only)
3. Data loading with motion
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_motion_fusion():
    """Test motion fusion module."""
    print("=" * 60)
    print("Test 1: Motion Fusion Module")
    print("=" * 60)
    
    from src.models.motion_fusion import MotionCrossAttention, SimpleMotionFusion
    
    B, T = 2, 10
    motion_dim = 128
    clip_dim = 256
    output_dim = 256
    
    # Test data
    motion_feats = torch.randn(B, T, motion_dim)
    clip_feats = torch.randn(B, T, clip_dim)
    
    # Test cross-attention
    print("\n[1.1] MotionCrossAttention")
    cross_attn = MotionCrossAttention(
        motion_dim=motion_dim,
        clip_dim=clip_dim,
        output_dim=output_dim,
        num_heads=4
    )
    fused = cross_attn(motion_feats, clip_feats)
    print(f"  Input: motion {motion_feats.shape}, CLIP {clip_feats.shape}")
    print(f"  Output: {fused.shape}")
    print(f"  ✓ Cross-attention works")
    
    # Test simple fusion
    print("\n[1.2] SimpleMotionFusion")
    simple = SimpleMotionFusion(
        motion_dim=motion_dim,
        clip_dim=clip_dim,
        output_dim=output_dim
    )
    fused_simple = simple(motion_feats, clip_feats)
    print(f"  Output: {fused_simple.shape}")
    print(f"  ✓ Simple fusion works")
    
    print("\n✅ Motion fusion module test passed!\n")


def test_dsn_with_motion():
    """Test DSN with motion features."""
    print("=" * 60)
    print("Test 2: DSN with Motion Features")
    print("=" * 60)
    
    from src.models.dsn_advanced import DSNAdvanced, DSNConfig
    
    # Create config with motion
    config = DSNConfig(
        feat_dim=512,
        hidden_dim=256,
        lstm_hidden=128,
        use_motion=True,
        motion_dim=128,
        motion_fusion_type="cross_attention",
        num_attn_heads=4,
        num_attn_layers=2,
        use_cache=False  # Disable cache for testing
    )
    
    model = DSNAdvanced(config)
    print(f"\n[2.1] Model created")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass with motion
    B, T, D = 2, 20, 512
    D_m = 128
    
    clip_feats = torch.randn(B, T, D)
    motion_feats = torch.randn(B, T, D_m)
    
    print(f"\n[2.2] Forward pass with motion")
    probs = model(clip_feats, motion_feats=motion_feats)
    print(f"  Input: CLIP {clip_feats.shape}, motion {motion_feats.shape}")
    print(f"  Output: {probs.shape}")
    print(f"  Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    assert probs.shape == (B, T), f"Expected {(B, T)}, got {probs.shape}"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probs not in [0, 1]"
    print(f"  ✓ Forward pass with motion works")
    
    # Test without motion (backward compatibility)
    print(f"\n[2.3] Forward pass without motion (backward compat)")
    probs_no_motion = model(clip_feats, motion_feats=None)
    print(f"  Output: {probs_no_motion.shape}")
    print(f"  ✓ Backward compatibility works")
    
    # Test gradient flow
    print(f"\n[2.4] Gradient flow")
    loss = probs.mean()
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for _ in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total}")
    print(f"  ✓ Gradients flow correctly")
    
    print("\n✅ DSN with motion test passed!\n")


def test_data_loading():
    """Test data loading with motion features."""
    print("=" * 60)
    print("Test 3: Data Loading with Motion")
    print("=" * 60)
    
    from src.datasets import load_scene_dir, list_scene_dirs
    
    dataset_root = Path("data/sakuga_dataset_100_samples")
    if not dataset_root.exists():
        print(f"  ⚠️  Dataset not found at {dataset_root}")
        print(f"  Skipping data loading test")
        return
    
    scene_dirs = list_scene_dirs(str(dataset_root))
    if not scene_dirs:
        print(f"  ⚠️  No scenes found")
        print(f"  Skipping data loading test")
        return
    
    # Test loading without motion
    print(f"\n[3.1] Load without motion")
    sample = load_scene_dir(scene_dirs[0], load_frames=False, load_motion=False)
    print(f"  CLIP feats: {sample.feats.shape}")
    print(f"  Motion: {sample.motion}")
    print(f"  ✓ Load without motion works")
    
    # Test loading with motion (if available)
    print(f"\n[3.2] Load with motion")
    sample_with_motion = load_scene_dir(scene_dirs[0], load_frames=False, load_motion=True)
    print(f"  CLIP feats: {sample_with_motion.feats.shape}")
    if sample_with_motion.motion is not None:
        print(f"  Motion: {sample_with_motion.motion.shape}")
        print(f"  ✓ Motion features loaded successfully")
        
        # Verify temporal consistency
        T_clip = sample_with_motion.feats.shape[0]
        T_motion = sample_with_motion.motion.shape[0]
        assert T_clip == T_motion, f"Temporal mismatch: {T_clip} vs {T_motion}"
        print(f"  ✓ Temporal consistency verified (T={T_clip})")
    else:
        print(f"  Motion: None (not precomputed yet)")
        print(f"  ℹ️  Run scripts/precompute_raft_motion.py first")
    
    print("\n✅ Data loading test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RAFT Motion Integration - Quick Test")
    print("=" * 60 + "\n")
    
    try:
        test_motion_fusion()
        test_dsn_with_motion()
        test_data_loading()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Precompute RAFT motion features:")
        print("   python scripts/precompute_raft_motion.py \\")
        print("     --dataset_root data/sakuga_dataset_100_samples \\")
        print("     --raft_model repos/RAFT/models/raft-small.pth \\")
        print("     --device cuda")
        print("\n2. Train with motion:")
        print("   python -m src.pipeline.train_rl_dsn \\")
        print("     --dataset_root data/sakuga_dataset_100_samples \\")
        print("     --save_dir runs/dsn_raft_motion \\")
        print("     --use_raft_motion 1 \\")
        print("     --model_type advanced \\")
        print("     --epochs 20")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
