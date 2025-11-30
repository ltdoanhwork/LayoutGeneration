#!/bin/bash
# Track A: Feature only (Concatenate Anime-CLIP-IQA attributes)

set -e

echo "=========================================="
echo "Training Track A: Anime-CLIP-IQA Features"
echo "=========================================="

# Configuration
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_track_a_features"
LOG_DIR="runs/dsn_track_a_features/logs"
EPOCHS=20
DEVICE="cuda"

# Validation
VAL_VIDEOS_DIR="/home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test"
VAL_OUTPUT_DIR="runs/dsn_track_a_features/val_runs"

# Anime-CLIP-IQA
USE_ANIME_ATTRS=1
ANIME_ATTRS_DIM=6 # 6 prompts defined in prepare_anime_attrs.py
FEAT_DIM=$((512 + ANIME_ATTRS_DIM))

echo "Configuration:"
echo "  Dataset: $DATASET_ROOT"
echo "  Save dir: $SAVE_DIR"
echo "  Epochs: $EPOCHS"
echo "  Device: $DEVICE"
echo "  Anime Attrs: ENABLED (Dim: $ANIME_ATTRS_DIM)"
echo "  Total Feat Dim: $FEAT_DIM"
echo ""

# Check for anime_attrs.npy
echo "Checking for Anime-CLIP-IQA attributes..."
FIRST_VIDEO=$(ls -d $DATASET_ROOT/*/ | head -n 1)
FIRST_SCENE=$(ls -d ${FIRST_VIDEO}scene_*/ | head -n 1)
ATTRS_FILE="${FIRST_SCENE}anime_attrs.npy"

if [ ! -f "$ATTRS_FILE" ]; then
    echo "⚠️  WARNING: Anime attributes not found at $ATTRS_FILE"
    echo "Please run precomputation first:"
    echo "  python scripts/prepare_anime_attrs.py --dataset_root $DATASET_ROOT --device cuda"
    exit 1
else
    echo "✓ Anime attributes found"
fi

python -m src.pipeline.train_rl_dsn \
  --dataset_root $DATASET_ROOT \
  --save_dir $SAVE_DIR \
  --log_dir $LOG_DIR \
  --epochs $EPOCHS \
  --device $DEVICE \
  \
  --model_type advanced \
  --feat_dim $FEAT_DIM \
  --enc_hidden 256 \
  --lstm_hidden 128 \
  --dropout 0.3 \
  \
  --num_attn_heads 4 \
  --num_attn_layers 2 \
  --num_scales 3 \
  --use_cache 1 \
  --cache_size 1000 \
  --pos_encoding_type sinusoidal \
  --use_lstm_in_advanced 1 \
  \
  --use_raft_motion 1 \
  --motion_dim 128 \
  --motion_fusion_type cross_attention \
  \
  --use_anime_attrs $USE_ANIME_ATTRS \
  --anime_attrs_dim $ANIME_ATTRS_DIM \
  \
  --lr 1e-4 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --entropy_coef 0.01 \
  --baseline_momentum 0.9 \
  \
  --budget_ratio 0.06 \
  --budget_penalty 0.05 \
  --Bmin 3 \
  --Bmax 15 \
  \
  --w_div 1.0 \
  --w_rep 1.0 \
  --w_rec 0.5 \
  --w_fd 0.2 \
  --w_ms 0.2 \
  --w_motion 0.2 \
  --ms_swd_scales 3 \
  --ms_swd_dirs 16 \
  --use_motion 1 \
  --use_lpips_div 0 \
  \
  --val_videos_dir $VAL_VIDEOS_DIR \
  --val_output_dir $VAL_OUTPUT_DIR \
  --validate_every 1 \
  --eval_embedder clip_vitb32 \
  --eval_backend transnetv2 \
  --eval_sample_stride 5 \
  --eval_resize_w 320 \
  --eval_resize_h 180 \
  --eval_with_baselines

echo "Done."
