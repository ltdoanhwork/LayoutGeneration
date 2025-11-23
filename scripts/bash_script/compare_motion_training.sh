#!/bin/bash
# Comparison script: Train both versions (with and without motion) for ablation study

set -e

echo "=========================================="
echo "DSN Training Comparison"
echo "With RAFT Motion vs Without Motion"
echo "=========================================="

DATASET_ROOT="data/sakuga_dataset_100_samples"
EPOCHS=20
DEVICE="cuda"

# Check if motion features exist
echo ""
echo "Checking prerequisites..."
FIRST_VIDEO=$(ls -d $DATASET_ROOT/*/ 2>/dev/null | head -n 1)
if [ -z "$FIRST_VIDEO" ]; then
    echo "❌ Dataset not found at $DATASET_ROOT"
    exit 1
fi

FIRST_SCENE=$(ls -d ${FIRST_VIDEO}scene_*/ 2>/dev/null | head -n 1)
MOTION_FILE="${FIRST_SCENE}motion_raft.npy"

if [ ! -f "$MOTION_FILE" ]; then
    echo "⚠️  WARNING: Motion features not found!"
    echo "Please run precomputation first:"
    echo "  python scripts/precompute_raft_motion.py \\"
    echo "    --dataset_root $DATASET_ROOT \\"
    echo "    --raft_model repos/RAFT/models/raft-small.pth \\"
    echo "    --device cuda"
    echo ""
    exit 1
fi

echo "✓ Dataset found: $DATASET_ROOT"
echo "✓ Motion features found"
echo ""

# Ask user which to run
echo "Select training mode:"
echo "  1) Train WITHOUT motion (baseline)"
echo "  2) Train WITH RAFT motion"
echo "  3) Train BOTH (sequential)"
echo ""
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "Training Baseline (No Motion)"
        echo "=========================================="
        bash scripts/bash_script/train_advanced_dsn.sh
        ;;
    2)
        echo ""
        echo "=========================================="
        echo "Training with RAFT Motion"
        echo "=========================================="
        bash scripts/bash_script/train_dsn_with_raft_motion.sh
        ;;
    3)
        echo ""
        echo "=========================================="
        echo "Training BOTH versions sequentially"
        echo "=========================================="
        echo ""
        echo "[1/2] Training baseline (no motion)..."
        bash scripts/bash_script/train_advanced_dsn.sh
        
        echo ""
        echo "[2/2] Training with RAFT motion..."
        bash scripts/bash_script/train_dsn_with_raft_motion.sh
        
        echo ""
        echo "=========================================="
        echo "✅ Both trainings completed!"
        echo "=========================================="
        echo ""
        echo "Compare results in TensorBoard:"
        echo "  tensorboard --logdir runs/ --port 6006"
        echo ""
        echo "Baseline logs: runs/dsn_advanced_v1_no_motion_100_samples/"
        echo "Motion logs:   runs/dsn_raft_motion/logs/"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Done!"
