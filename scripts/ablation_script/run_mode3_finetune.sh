#!/bin/bash

# Mode 3: Fine-tune from a Checkpoint
# Usage: bash scripts/ablation_script/run_mode3_finetune.sh [GPU_ID] [CHECKPOINT_PATH]

GPU_ID=${1:-0}
CHECKPOINT=${2:-""}
DATASET_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11"
SAVE_DIR="runs/ablation_vsumm/mode3_finetune"

# Ensure we run from project root and include necessary paths
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/pytorch-vsumm-reinforce

SCRIPT_PATH="ablation/pytorch-vsumm-reinforce/train_custom.py"

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 [GPU_ID] [CHECKPOINT_PATH]"
    echo "Please provide the path to the checkpoint to fine-tune."
    exit 1
fi

mkdir -p $SAVE_DIR

echo "========================================================"
echo "MODE 3: Fine-tune"
echo "Resume from: $CHECKPOINT"
echo "Dataset: $DATASET_ROOT"
echo "Output: $SAVE_DIR"
echo "========================================================"

# Lower LR for fine-tuning usually, but let's keep 1e-4 or try 1e-5
# Going with 1e-5 for stable fine-tuning
python $SCRIPT_PATH \
    --dataset-root $DATASET_ROOT \
    --save-dir $SAVE_DIR \
    --resume $CHECKPOINT \
    --gpu $GPU_ID \
    --max-epoch 40 \
    --input-dim 512 \
    --hidden-dim 256 \
    --lr 1e-5 \
    --num-episode 5 \
    --beta 0.01 \
    --verbose
