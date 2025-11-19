#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify train_rl_dsn integration with batch_eval and tensorboard

This will:
1. Run a mini training (1 epoch, small dataset)
2. Validate with batch_eval
3. Check tensorboard logs
"""
import os
import sys
import json
import subprocess
from pathlib import Path


def test_training_integration():
    """Test if training can run and log to tensorboard"""
    print("="*80)
    print("TESTING TRAIN_RL_DSN INTEGRATION")
    print("="*80)
    
    # Check if dataset exists
    dataset_root = "outputs/sakuga_dataset"
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset not found at {dataset_root}")
        print("   Please run prepare_dataset_v2.py first")
        return False
    
    print(f"✅ Dataset found at {dataset_root}")
    
    # Check if validation videos exist
    val_videos_dir = "data/samples/Sakuga"
    if not os.path.exists(val_videos_dir):
        print(f"❌ Validation videos not found at {val_videos_dir}")
        return False
    
    print(f"✅ Validation videos found at {val_videos_dir}")
    
    # Create test output directories
    test_save_dir = "outputs/test_train_integration"
    test_val_dir = "outputs/test_val_integration"
    test_log_dir = "runs/test_train_integration"
    
    Path(test_save_dir).mkdir(parents=True, exist_ok=True)
    Path(test_val_dir).mkdir(parents=True, exist_ok=True)
    
    # Run minimal training
    print("\n" + "-"*80)
    print("Running minimal training (1 epoch, 2 validation videos)...")
    print("-"*80)
    
    cmd = [
        "python", "-m", "src.pipeline.train_rl_dsn",
        "--dataset_root", dataset_root,
        "--save_dir", test_save_dir,
        "--epochs", "1",
        "--device", "cuda",
        "--feat_dim", "512",
        "--enc_hidden", "256",
        "--lstm_hidden", "128",
        "--budget_ratio", "0.06",
        "--Bmin", "3",
        "--Bmax", "15",
        "--val_videos_dir", val_videos_dir,
        "--val_output_dir", test_val_dir,
        "--validate_every", "1",
        "--eval_embedder", "clip_vitb32",
        "--eval_backend", "pyscenedetect",
        "--eval_sample_stride", "5",
        "--eval_resize_w", "320",
        "--eval_resize_h", "180",
        "--eval_with_baselines",
        "--eval_max_videos", "2",  # Only 2 videos for testing
        "--log_dir", test_log_dir,
    ]
    
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print("\n❌ Training failed")
            print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
            print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return False
        
        print("✅ Training completed")
        
    except subprocess.TimeoutExpired:
        print("❌ Training timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False
    
    # Check outputs
    print("\n" + "-"*80)
    print("Checking outputs...")
    print("-"*80)
    
    # Check checkpoint
    ckpt_path = Path(test_save_dir) / "dsn_checkpoint_ep1.pt"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found at {ckpt_path}")
        return False
    print(f"✅ Checkpoint saved at {ckpt_path}")
    
    # Check validation summary
    val_summary = Path(test_val_dir) / "ep1" / "summary_results.json"
    if not val_summary.exists():
        print(f"❌ Validation summary not found at {val_summary}")
        return False
    print(f"✅ Validation summary at {val_summary}")
    
    # Check validation metrics
    with open(val_summary, "r") as f:
        summary = json.load(f)
    
    agg = summary.get("aggregate_metrics", {})
    print(f"\n📊 Validation Metrics (Epoch 1):")
    for key, val in sorted(agg.items()):
        if val is not None:
            print(f"  {key:40s}: {val:.6f}")
        else:
            print(f"  {key:40s}: N/A")
    
    # Check tensorboard logs
    tb_event_files = list(Path(test_log_dir).glob("events.out.tfevents.*"))
    if not tb_event_files:
        print(f"\n⚠️  No tensorboard event files found in {test_log_dir}")
        print("   Training metrics may not have been logged")
    else:
        print(f"\n✅ Tensorboard logs found: {len(tb_event_files)} file(s)")
        print(f"   View with: tensorboard --logdir {test_log_dir}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print("\nYou can now:")
    print(f"  1. View tensorboard: tensorboard --logdir {test_log_dir}")
    print(f"  2. Check validation results: cat {val_summary}")
    print(f"  3. Use checkpoint for inference: {ckpt_path}")
    
    return True


if __name__ == "__main__":
    success = test_training_integration()
    sys.exit(0 if success else 1)
