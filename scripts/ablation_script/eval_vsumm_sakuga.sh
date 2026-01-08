#!/bin/bash

# Evaluation script for VSUMM-Reinforce on Sakuga Test Data
# Usage: bash scripts/ablation_script/eval_vsumm_sakuga.sh [GPU_ID] [CHECKPOINT_PATH]

GPU_ID=${1:-0}
CHECKPOINT=${2:-""}
DATASET_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_test_precompute"
SAVE_DIR="runs/ablation_vsumm/sakuga_eval"
# Ensure we run from project root and include necessary paths
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/pytorch-vsumm-reinforce

SCRIPT_PATH="ablation/pytorch-vsumm-reinforce/train_custom.py"

if [ -z "$CHECKPOINT" ]; then
    echo "Please provide checkpoint path as second argument"
    exit 1
fi

mkdir -p $SAVE_DIR

echo "Starting evaluation on GPU $GPU_ID..."
echo "Checkpoint: $CHECKPOINT"
echo "Data: $DATASET_ROOT"

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
