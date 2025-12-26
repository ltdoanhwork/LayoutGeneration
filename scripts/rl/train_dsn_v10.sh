#!/bin/bash
# ---------------------------------------------------------
# V10 VLM-Guided RL Training Script
# Optimized for A100 GPU
# ---------------------------------------------------------

# Path configuration
DATA_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_v10_vlm_guided"
VAL_VIDEO_DIR="data/validation_videos"
VAL_OUT_DIR="runs/dsn_v10_val"

# Model hyperparameters
EPOCHS=60
LR=1e-4   # Lowered for stability
BUDGET_RATIO=0.1
B_MIN=3
B_MAX=15

# V10 Specific
VLM_SCALE=4.0
VLM_DECAY_EPOCHS=30
DISTILL_LR=1e-4

# Run Training
python3 -m src.pipeline.train_rl_dsn_v10 \
    --dataset_root "$DATA_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --budget_ratio "$BUDGET_RATIO" \
    --Bmin "$B_MIN" \
    --Bmax "$B_MAX" \
    --use_pcgrad 1 \
    --use_dpp 1 \
    --anime_scale 5.0 \
    --vlm_scale "$VLM_SCALE" \
    --vlm_decay_epochs "$VLM_DECAY_EPOCHS" \
    --distill_lr "$DISTILL_LR" \
    --val_videos_dir "$VAL_VIDEO_DIR" \
    --val_output_dir "$VAL_OUT_DIR" \
    --validate_every 5 \
    --device cuda \
    --eval_device cuda

echo "Training session finished. Check results in $SAVE_DIR"
