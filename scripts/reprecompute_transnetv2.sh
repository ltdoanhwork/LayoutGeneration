#!/bin/bash
# Re-precompute dataset with TransNetV2
# This ensures training and evaluation use the same scene detection

set -e


# Activate environment
echo "=============================================="
echo "Re-Precompute Dataset with TransNetV2"
echo "=============================================="
echo ""

# Input/Output paths
VIDEO_DIR="data/samples/Sakuga"
OUT_DIR="data/sakuga_dataset_100_samples_transnetv2"

# TransNetV2 parameters (same as eval)
BACKEND="transnetv2"
MODEL_DIR="./src/models/TransNetV2"
PROB_THRESHOLD=0.5
SCENE_DEVICE="cuda"
MIN_SCENE_LEN=48

# Frame sampling (keep same as before)
FPS=6
STRIDE=1

echo "📁 Video dir: $VIDEO_DIR"
echo "📁 Output dir: $OUT_DIR"
echo "🔧 Backend: $BACKEND"
echo "⚙️  Prob threshold: $PROB_THRESHOLD"
echo "🎬 FPS: $FPS, Stride: $STRIDE"
echo "📐 Min scene len: $MIN_SCENE_LEN"
echo ""

# Backup old dataset if exists
if [ -d "$OUT_DIR" ]; then
    echo "⚠️  Output dir exists, backing up..."
    mv "$OUT_DIR" "${OUT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
fi

# Run precompute
python -m scripts.prepare_rl_dataset \
    --video_dir "$VIDEO_DIR" \
    --out_dir "$OUT_DIR" \
    --backend "$BACKEND" \
    --model_dir "$MODEL_DIR" \
    --prob_threshold $PROB_THRESHOLD \
    --scene_device "$SCENE_DEVICE" \
    --min_scene_len $MIN_SCENE_LEN \
    --fps $FPS \
    --stride $STRIDE \
    --extractor auto \
    --device cuda \
    --export_preview \
    --preview_which middle

echo ""
echo "✅ Precompute complete!"
echo ""

# Precompute RAFT motion features
echo "🎥 Precomputing RAFT motion features..."
python -m scripts.precompute_raft_motion \
    --dataset_root "$OUT_DIR" \
    --device cuda

echo ""
echo "✅ RAFT motion precompute complete!"
echo ""

# Precompute Anime-CLIP-IQA attributes
echo "🎨 Precomputing Anime-CLIP-IQA attributes..."
python -m scripts.prepare_anime_attrs \
    --dataset_root "$OUT_DIR" \
    --device cuda

echo ""
echo "=============================================="
echo "✅ All precomputation complete!"
echo ""
echo "📁 New dataset: $OUT_DIR"
echo ""
echo "NEXT STEPS:"
echo "1. Update train_dsn_v5_plus.slurm: DATASET_ROOT=\"$OUT_DIR\""
echo "2. Re-run training"
echo "=============================================="
