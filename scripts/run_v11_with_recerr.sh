#!/bin/bash
# ============================================================
# V11 Training with RecErr Optimization
# ============================================================
# 
# This script adds RecErr to the reward function to optimize
# for representativeness alongside aesthetic quality.
#
# R_total = R_quality + 0.3*R_diversity + rec_weight*(-RecErr)
#
# Usage: ./scripts/run_v11_with_recerr.sh [gpu_id] [rec_weight]
# Example: ./scripts/run_v11_with_recerr.sh 0 0.5
# ============================================================

GPU_ID=${1:-0}
REC_WEIGHT=${2:-0.2}
DATASET="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new"
VAL_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test"
SAVE_DIR="runs/training_v11_recerr_w${REC_WEIGHT}"
EPOCHS=60

echo "============================================================"
echo "V11 Training with RecErr Optimization"
echo "============================================================"
echo "GPU:         $GPU_ID"
echo "Dataset:     $DATASET"
echo "Val Data:    $VAL_ROOT"
echo "Output:      $SAVE_DIR"
echo "Epochs:      $EPOCHS"
echo "RecErr Wgt:  $REC_WEIGHT"
echo "============================================================"
echo ""
echo "Reward Formula:"
echo "  R_total = R_quality + 0.3*R_div + ${REC_WEIGHT}*(-RecErr)"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Limit CPU threads to prevent high CPU usage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

mkdir -p "$SAVE_DIR"

python -m src.pipeline.train_rl_dsn_v11_final \
    --dataset_root "$DATASET" \
    --val_root "$VAL_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs $EPOCHS \
    --lr 1e-4 \
    --diversity_weight 0.3 \
    --rec_weight $REC_WEIGHT \
    --budget_ratio 0.15 \
    --Bmin 3 \
    --Bmax 15 \
    --entropy_coef 0.02 \
    --clip_range 0.2 \
    --num_attn_layers 2 \
    --attr_suffix "anime_attrs.npy" \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/training.log"

echo ""
echo "============================================================"
echo "Training Complete!"
echo "Results:     $SAVE_DIR"
echo "Best Model:  $SAVE_DIR/best.pt"
echo "TensorBoard: tensorboard --logdir $SAVE_DIR/logs"
echo "============================================================"
