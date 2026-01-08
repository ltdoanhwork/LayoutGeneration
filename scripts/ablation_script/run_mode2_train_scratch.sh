#!/bin/bash

# Mode 2: Train from Scratch on Sakuga
# Usage: bash scripts/ablation_script/run_mode2_train_scratch.sh [GPU_ID]

GPU_ID=${1:-0}
DATASET_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11"
SAVE_DIR="runs/ablation_vsumm/mode2_train_scratch"

# Ensure we run from project root and include necessary paths
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/pytorch-vsumm-reinforce

SCRIPT_PATH="ablation/pytorch-vsumm-reinforce/train_custom.py"

mkdir -p $SAVE_DIR

echo "========================================================"
echo "MODE 2: Train from Scratch"
echo "Dataset: $DATASET_ROOT"
echo "Output: $SAVE_DIR"
echo "========================================================"

# Parameters tuned for scratch training
python $SCRIPT_PATH \
    --dataset-root $DATASET_ROOT \
    --save-dir $SAVE_DIR \
    --gpu $GPU_ID \
    --max-epoch 60 \
    --input-dim 512 \
    --hidden-dim 256 \
    --lr 1e-4 \
    --num-episode 5 \
    --beta 0.01 \
    --verbose
