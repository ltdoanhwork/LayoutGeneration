#!/bin/bash
# V9 Quality-Focused RL Training Script
#
# Key improvements:
# - Increased anime_scale: 3.0 → 5.0
# - NEW percentile-based reward
# - NO motion features (simpler, faster)
#
# Goal: Maximize Mean Percentile Rank and Top-K Coverage

set -e

# Activate environment
eval "$(conda shell.bash hook)"
conda activate sam

# Dataset and output paths
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_v9_quality_focused"
VAL_VIDEOS_DIR="data/samples/Sakuga_test"
VAL_OUTPUT_DIR="$SAVE_DIR/validation"

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$VAL_OUTPUT_DIR"

echo "=============================================="
echo "V9 Quality-Focused DSN Training"
echo "=============================================="
echo "Dataset: $DATASET_ROOT"
echo "Output: $SAVE_DIR"
echo "=============================================="

python -m src.pipeline.train_rl_dsn_v9 \
    --dataset_root "$DATASET_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs 60 \
    --seed 42 \
    \
    --feat_dim 512 \
    --enc_hidden 256 \
    --lstm_hidden 128 \
    --use_anime_attrs 1 \
    --anime_attrs_dim 6 \
    --use_raft_motion 0 \
    \
    --use_pcgrad 1 \
    --use_dpp 1 \
    --use_tts 0 \
    \
    --rec_err_threshold 0.35 \
    --coverage_threshold 0.3 \
    --diversity_threshold 0.25 \
    --lambda_lr 0.01 \
    \
    --anime_scale 5.0 \
    --quantile_scale 3.0 \
    --percentile_scale 2.0 \
    --use_curriculum 1 \
    \
    --lr 2e-4 \
    --clip_range 0.2 \
    --n_ppo_epochs 4 \
    --entropy_coef 0.01 \
    \
    --budget_ratio 0.06 \
    --Bmin 3 \
    --Bmax 15 \
    \
    --device cuda \
    \
    --val_videos_dir "$VAL_VIDEOS_DIR" \
    --val_output_dir "$VAL_OUTPUT_DIR" \
    --validate_every 999 \\
    --eval_backend transnetv2 \
    --eval_embedder clip_vitb32

echo ""
echo "=============================================="
echo "V9 Training Complete!"
echo "Checkpoints: $SAVE_DIR"
echo "=============================================="
