#!/bin/bash
# ============================================================
# V11 Constrained Training (RecErr/Coverage Constraints)
# ============================================================
# 
# Uses Lagrangian Constraints (PremiumRewardV8) to balance
# Quality (High MPR) and Representativeness (Low RecErr).
#
# Unlike simple weighted sums, this allows the model to
# satisfy RecErr < 0.35 without sacrificing Quality.
#
# Usage: ./scripts/run_v11_constrained.sh [gpu_id]
# ============================================================

GPU_ID=${1:-0}
DATASET="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new"
VAL_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11_new_test"
SAVE_DIR="runs/training_v11_constrained_lagrangian"
EPOCHS=60

echo "============================================================"
echo "V11 Constrained Training (Lagrangian Optimization)"
echo "============================================================"
echo "GPU:         $GPU_ID"
echo "Dataset:     $DATASET"
echo "Val Data:    $VAL_ROOT"
echo "Output:      $SAVE_DIR"
echo "Epochs:      $EPOCHS"
echo "============================================================"
echo "Constraints:"
echo "  RecErr < 0.35"
echo "  Coverage Gap < 0.3"
echo "  Diversity > 0.25"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Limit CPU threads to prevent high CPU usage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

mkdir -p "$SAVE_DIR"

python -m src.pipeline.train_rl_dsn_v11_constrained \
    --dataset_root "$DATASET" \
    --val_root "$VAL_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs $EPOCHS \
    --lr 1e-4 \
    --rec_err_thresh 0.10 \
    --coverage_thresh 0.6 \
    --diversity_thresh 0.15 \
    --anime_scale 5.0 \
    --quantile_scale 4.0 \
    --budget_ratio 0.15 \
    --Bmin 3 \
    --Bmax 15 \
    --entropy_coef 0.05 \
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
