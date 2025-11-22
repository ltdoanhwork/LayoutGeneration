#!/bin/bash
# Test RAFT motion integration end-to-end

set -e  # Exit on error

echo "=========================================="
echo "RAFT Motion Integration Test"
echo "=========================================="

# Step 1: Test motion fusion module
echo ""
echo "[1/4] Testing motion fusion module..."
python -m src.models.motion_fusion

# Step 2: Test RAFT precomputation on single video
echo ""
echo "[2/4] Testing RAFT precomputation (1 video)..."
python scripts/precompute_raft_motion.py \
  --dataset_root data/sakuga_dataset_100_samples \
  --raft_model repos/RAFT/models/raft-small.pth \
  --motion_dim 128 \
  --device cuda \
  --max_videos 1

# Step 3: Verify motion features were created
echo ""
echo "[3/4] Verifying motion features..."
python -c "
import numpy as np
from pathlib import Path

dataset_root = Path('data/sakuga_dataset_100_samples')
video_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
first_video = video_dirs[0]
scene_dirs = sorted(first_video.glob('scene_*'))

if not scene_dirs:
    print('ERROR: No scenes found')
    exit(1)

first_scene = scene_dirs[0]
motion_path = first_scene / 'motion_raft.npy'

if not motion_path.exists():
    print(f'ERROR: Motion file not found at {motion_path}')
    exit(1)

motion = np.load(motion_path)
feats = np.load(first_scene / 'feats.npy')

print(f'✓ Motion features found: {motion_path}')
print(f'  Motion shape: {motion.shape}')
print(f'  CLIP shape: {feats.shape}')
print(f'  Motion dim: {motion.shape[1]}')
print(f'  Temporal consistency: T_motion={motion.shape[0]}, T_clip={feats.shape[0]}')

if motion.shape[0] != feats.shape[0]:
    print('ERROR: Temporal dimension mismatch!')
    exit(1)

print('✓ All checks passed!')
"

# Step 4: Test training with motion features (2 epochs)
echo ""
echo "[4/4] Testing DSN training with RAFT motion..."
python -m src.pipeline.train_rl_dsn \
  --dataset_root data/sakuga_dataset_100_samples \
  --save_dir runs/test_raft_motion \
  --log_dir runs/test_raft_motion/logs \
  --epochs 2 \
  --device cuda \
  --model_type advanced \
  --use_raft_motion 1 \
  --motion_dim 128 \
  --motion_fusion_type cross_attention \
  --feat_dim 512 \
  --enc_hidden 256 \
  --lstm_hidden 128 \
  --budget_ratio 0.06 \
  --Bmin 3 \
  --Bmax 15

echo ""
echo "=========================================="
echo "✅ All tests passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run full RAFT precomputation:"
echo "   python scripts/precompute_raft_motion.py --dataset_root data/sakuga_dataset_100_samples --raft_model repos/RAFT/models/raft-small.pth --device cuda"
echo ""
echo "2. Train with motion features:"
echo "   python -m src.pipeline.train_rl_dsn --dataset_root data/sakuga_dataset_100_samples --save_dir runs/dsn_raft_motion --use_raft_motion 1 --model_type advanced --epochs 20"
