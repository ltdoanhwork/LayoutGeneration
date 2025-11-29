#!/bin/bash
# Track D: Anime-CLIP-IQA Emphasis + Prob Separation
# Based on Track C but adds specific emphasis on cinematic/sakuga/sharpness and prob separation.

set -e

echo "=========================================="
echo "Training Track D: Anime Emphasis + Prob Sep"
echo "=========================================="

# Configuration
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_track_d_anime"
LOG_DIR="runs/dsn_track_d_anime/logs"
EPOCHS=20
DEVICE="cuda"

# Validation
VAL_VIDEOS_DIR="/home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test"
VAL_OUTPUT_DIR="runs/dsn_track_d_anime/val_runs"

# Anime-CLIP-IQA Features (Track A)
USE_ANIME_ATTRS=1
ANIME_ATTRS_DIM=6
FEAT_DIM=$((512 + ANIME_ATTRS_DIM))

# Anime-CLIP-IQA Rewards (Track B - Legacy)
# We keep these 0 or low if we want to rely on the new "emphasis" reward
USE_ANIME_REWARD=0 
W_LOOK=0.0
W_SAKUGA=0.0
W_STORY=0.0

# New Rewards (Track D)
W_ANIME=0.2       # Start small: 0.2-0.3
W_PROBSEP=0.1     # Start small: 0.1
ENTROPY_COEF=0.005 # Reduced from 0.01 to allow sharper probs

echo "Configuration:"
echo "  Dataset: $DATASET_ROOT"
echo "  Save dir: $SAVE_DIR"
echo "  Epochs: $EPOCHS"
echo "  Device: $DEVICE"
echo "  Anime Attrs: ENABLED (Dim: $ANIME_ATTRS_DIM)"
echo "  New Rewards: W_ANIME=$W_ANIME, W_PROBSEP=$W_PROBSEP"
echo "  Entropy Coef: $ENTROPY_COEF"
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
  --use_anime_reward $USE_ANIME_REWARD \
  --w_look $W_LOOK \
  --w_sakuga $W_SAKUGA \
  --w_story $W_STORY \
  \
  --w_anime $W_ANIME \
  --w_probsep $W_PROBSEP \
  \
  --lr 1e-4 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --entropy_coef $ENTROPY_COEF \
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
  --eval_backend pyscenedetect \
  --eval_sample_stride 5 \
  --eval_resize_w 320 \
  --eval_resize_h 180 \
  --eval_with_baselines

echo "Done."
