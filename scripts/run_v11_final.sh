#!/bin/bash
# ============================================================
# V11 Final Training - Comprehensive Metrics
# ============================================================
# 
# Features:
# - 6 Validation Metrics: RecErr, Frechet, MPR, Top10, LPIPS_Div, TempCov
# - Per-Attribute MPR with Radar Chart Visualization
# - Composite Score for Best Model Selection
# - Full TensorBoard Logging
#
# Usage: ./scripts/run_v11_final.sh [gpu_id]
# ============================================================

GPU_ID=${1:-0}
DATASET="data/sakuga_dataset_v11"
VAL_ROOT="data/sakuga_test_precompute"
SAVE_DIR="runs/training_v11_final"
EPOCHS=60

echo "============================================================"
echo "V11 Final Training with Comprehensive Metrics"
echo "============================================================"
echo "GPU:       $GPU_ID"
echo "Dataset:   $DATASET"
echo "Val Data:  $VAL_ROOT"
echo "Output:    $SAVE_DIR"
echo "Epochs:    $EPOCHS"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Limit CPU threads to prevent high CPU usage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python -m src.pipeline.train_rl_dsn_v11_final \
    --dataset_root "$DATASET" \
    --val_root "$VAL_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs $EPOCHS \
    --lr 1e-4 \
    --diversity_weight 0.3 \
    --budget_ratio 0.15 \
    --Bmin 3 \
    --Bmax 15 \
    --entropy_coef 0.02 \
    --clip_range 0.2 \
    --num_attn_layers 2 \
    --device cuda \
    2>&1 | tee "$SAVE_DIR/training.log"

echo ""
echo "============================================================"
echo "Training Complete!"
echo "Results:   $SAVE_DIR"
echo "Best Model: $SAVE_DIR/best.pt"
echo "TensorBoard: tensorboard --logdir $SAVE_DIR/logs"
echo "============================================================"
