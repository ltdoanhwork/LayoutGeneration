#!/bin/bash

# Mode 1: Evaluate a specific checkpoint (Zero-shot / Pre-trained)
# Usage: bash scripts/ablation_script/run_mode1_eval_original.sh [GPU_ID] [CHECKPOINT_PATH]

GPU_ID=${1:-0}
CHECKPOINT=${2:-""}
# Default to v10 dataset or whatever the user wants to test on. User said Sakuga eval.
DATASET_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_test_precompute"
SAVE_DIR="runs/ablation_vsumm/mode1_eval_original"

# Ensure we run from project root and include necessary paths
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/pytorch-vsumm-reinforce

SCRIPT_PATH="ablation/pytorch-vsumm-reinforce/train_custom.py"

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 [GPU_ID] [CHECKPOINT_PATH]"
    echo "Please provide the path to the original/pre-trained checkpoint."
    exit 1
fi

mkdir -p $SAVE_DIR

echo "========================================================"
echo "MODE 1: Evaluate Original Checkpoint (Zero-shot)"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET_ROOT"
echo "Output: $SAVE_DIR"
echo "========================================================"

python $SCRIPT_PATH \
    --evaluate \
    --dataset-root $DATASET_ROOT \
    --save-dir $SAVE_DIR \
    --resume $CHECKPOINT \
    --gpu $GPU_ID \
    --input-dim 512 \
    --verbose \
    --save-results \
    --full-metrics
