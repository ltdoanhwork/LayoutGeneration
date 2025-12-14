#!/bin/bash
# Evaluate all V4 checkpoints

CHECKPOINT_DIR="runs/dsn_anime_v4_scaled/checkpoints"
VIDEOS_DIR="data/samples/Sakuga_test"  # Test set for evaluation
OUTPUT_DIR="runs/dsn_anime_v4_scaled/checkpoint_eval"

# Model config (must match training)
FEAT_DIM=512
ENC_HIDDEN=1024
LSTM_HIDDEN=512

# Eval settings
BUDGET_RATIO=0.06
BACKEND="transnetv2"
EMBEDDER="clip_vitb32"
DEVICE="cuda"
USE_ANIME_ATTRS=1
MIN_SCENE_LEN=48

# Limit videos for faster testing (remove for full eval)
MAX_VIDEOS=  # Set to empty for full evaluation: MAX_VIDEOS=""

echo "=============================================="
echo "Evaluating All V4 Checkpoints"
echo "=============================================="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Videos: $VIDEOS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Max videos per eval: ${MAX_VIDEOS:-all}"
echo "=============================================="

python scripts/eval_all_checkpoints.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --videos_dir "$VIDEOS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --feat_dim $FEAT_DIM \
    --enc_hidden $ENC_HIDDEN \
    --lstm_hidden $LSTM_HIDDEN \
    --budget_ratio $BUDGET_RATIO \
    --backend "$BACKEND" \
    --embedder "$EMBEDDER" \
    --use_anime_attrs $USE_ANIME_ATTRS \
    --min_scene_len $MIN_SCENE_LEN \
    ${MAX_VIDEOS:+--max_videos $MAX_VIDEOS}

echo ""
echo "✅ Done! Check results in: $OUTPUT_DIR/checkpoint_comparison.json"
