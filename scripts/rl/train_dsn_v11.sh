#!/bin/bash
# ---------------------------------------------------------
# V11 Training Script
# Quality + Temporal Diversity
# Target: MPR > 0.7 on both train and test
# ---------------------------------------------------------

DATA_ROOT="data/sakuga_dataset_100_samples"  # Use existing dataset first
SAVE_DIR="runs/dsn_v11"

python3 -m src.pipeline.train_rl_dsn_v11 \
    --dataset_root "$DATA_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs 60 \
    --lr 5e-5 \
    --budget_ratio 0.10 \
    --Bmin 3 \
    --Bmax 15 \
    --entropy_coef 0.01 \
    --clip_range 0.1 \
    --use_pcgrad 1 \
    --use_dpp 1 \
    --device cuda \
    --seed 42

echo "Training complete. Check $SAVE_DIR for results."
echo "Evaluate with: python -m eval.eval_on_scenes --checkpoint $SAVE_DIR/best.pt --dataset_dir $DATA_ROOT"
