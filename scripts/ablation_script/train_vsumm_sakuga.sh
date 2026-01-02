#!/bin/bash

# Training script for VSUMM-Reinforce on Sakuga Data
# Usage: bash scripts/ablation_script/train_vsumm_sakuga.sh [GPU_ID]

GPU_ID=${1:-0}
DATASET_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11"
SAVE_DIR="runs/ablation_vsumm/sakuga_train"
# Ensure we run from project root and include necessary paths
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/pytorch-vsumm-reinforce

SCRIPT_PATH="ablation/pytorch-vsumm-reinforce/train_custom.py"

mkdir -p $SAVE_DIR

echo "Starting training on GPU $GPU_ID..."
echo "Data: $DATASET_ROOT"
echo "Save: $SAVE_DIR"

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
    --evaluate \
    --verbose

# Note: The original repo used '--evaluate' flag in arguments sometimes confusingly,
# but here I removed --evaluate to TRAIN. 
# Wait, my previous write had --eval which was wrong if I want to train.
# Let's check train_custom.py argparser.
# parser.add_argument('--evaluate', action='store_true'...)
# So to TRAIN, I should NOT pass --evaluate.

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
