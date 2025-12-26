#!/bin/bash
# ---------------------------------------------------------
# V10 STABLE Training Script
# Conservative hyperparameters for stable training
# Target: Mean Percentile Rank > 0.7
# ---------------------------------------------------------

DATA_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_v10_stable"

python3 -m src.pipeline.train_rl_dsn_v10_stable \
    --dataset_root "$DATA_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs 60 \
    --lr 5e-5 \
    --budget_ratio 0.10 \
    --Bmin 3 \
    --Bmax 15 \
    --entropy_coef 0.01 \
    --clip_range 0.1 \
    --device cuda

echo "Training complete. Check $SAVE_DIR for results."
