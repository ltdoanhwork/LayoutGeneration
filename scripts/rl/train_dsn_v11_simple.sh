#!/bin/bash
# ---------------------------------------------------------
# V11 SIMPLIFIED Training Script
# Based on V10_stable success (MPR 0.71)
# Quality percentile + Diversity reward
# ---------------------------------------------------------

DATA_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_v11_simple"

python3 -m src.pipeline.train_rl_dsn_v11_simple \
    --dataset_root "$DATA_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs 60 \
    --lr 1e-4 \
    --budget_ratio 0.15 \
    --Bmin 3 \
    --Bmax 15 \
    --entropy_coef 0.02 \
    --clip_range 0.2 \
    --diversity_weight 0.3 \
    --device cuda

echo "Training complete. Check $SAVE_DIR for results."
echo "Evaluate: python -m eval.eval_v11_comprehensive --checkpoint $SAVE_DIR/best.pt --train_dir $DATA_ROOT"
