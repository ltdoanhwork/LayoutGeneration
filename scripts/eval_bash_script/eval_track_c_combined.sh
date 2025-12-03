#!/bin/bash
# Evaluation script for Track C: Combined Anime-CLIP-IQA
# Both features AND rewards from Anime-CLIP-IQA

set -e

echo "=========================================="
echo "Eval Track C: Combined Anime-CLIP-IQA"
echo "=========================================="

# Configuration matching train_track_c.sh
CHECKPOINT_DIR="runs/dsn_track_c_combined"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/dsn_checkpoint_ep5.pt"
VAL_VIDEOS_DIR="/home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test"
OUTPUT_DIR="runs/dsn_track_c_combined/eval_track_c_combined"
DEVICE="cuda"

# Anime-CLIP-IQA
USE_ANIME_ATTRS=1
ANIME_ATTRS_DIM=6
FEAT_DIM=$((512 + ANIME_ATTRS_DIM))

# Model architecture
ENC_HIDDEN=256
LSTM_HIDDEN=128

# Budget constraints
BUDGET_RATIO=0.06
BMIN=3
BMAX=15

# Evaluation settings - UNIFIED TO TRANSNETV2
EVAL_EMBEDDER="clip_vitb32"
EVAL_BACKEND="transnetv2"
EVAL_SAMPLE_STRIDE=5
EVAL_RESIZE_W=320
EVAL_RESIZE_H=180
EVAL_BACKBONE="resnet50"
EVAL_DEVICE="cuda"
EVAL_MAX_FRAMES=200
EVAL_TAU=0.5
MAX_VIDEOS=30

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Videos dir: $VAL_VIDEOS_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Backend: $EVAL_BACKEND (unified)"
echo "  Anime Attrs: ENABLED (Dim: $ANIME_ATTRS_DIM)"
echo "  Anime Reward: ENABLED"
echo "  Total Feat Dim: $FEAT_DIM"
echo ""

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "⚠️  WARNING: Checkpoint not found at $CHECKPOINT_PATH"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi
else
    echo "✓ Checkpoint found"
fi

echo ""
echo "Starting batch evaluation..."
echo ""

python -m eval.batch_eval \
  --videos_dir "$VAL_VIDEOS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT_PATH" \
  --device "$DEVICE" \
  \
  --feat_dim $FEAT_DIM \
  --enc_hidden $ENC_HIDDEN \
  --lstm_hidden $LSTM_HIDDEN \
  \
  --budget_ratio $BUDGET_RATIO \
  --Bmin $BMIN \
  --Bmax $BMAX \
  \
  --sample_stride $EVAL_SAMPLE_STRIDE \
  --resize_w $EVAL_RESIZE_W \
  --resize_h $EVAL_RESIZE_H \
  \
  --embedder "$EVAL_EMBEDDER" \
  --backend "$EVAL_BACKEND" \
  \
  --use_anime_attrs $USE_ANIME_ATTRS \
  --anime_attrs_dim $ANIME_ATTRS_DIM \
  \
  --eval_backbone "$EVAL_BACKBONE" \
  --eval_device "$EVAL_DEVICE" \
  --eval_sample_stride 1 \
  --eval_max_frames $EVAL_MAX_FRAMES \
  --eval_tau $EVAL_TAU \
  \
  --with_baselines \
  --max_videos $MAX_VIDEOS

echo ""
echo "=========================================="
echo "✅ Evaluation completed!"
echo "=========================================="
echo "Results: $OUTPUT_DIR/summary_results.json"
echo ""
