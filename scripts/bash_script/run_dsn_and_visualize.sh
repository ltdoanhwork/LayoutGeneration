#!/bin/bash
# Run DSN pipeline and create cityscape visualizations

VIDEO_PATH="data/samples/Sakuga/14652.mp4"
OUT_DIR="outputs/dsn_infer/14652"
CHECKPOINT="/home/serverai/ltdoanh/LayoutGeneration/runs/dsn_advanced_v1/dsn_checkpoint_ep17.pt"

echo "========================================="
echo "Step 1: Running DSN Pipeline"
echo "========================================="
python -m eval.run_dsn_pipeline \
    --video ${VIDEO_PATH} \
    --out_dir ${OUT_DIR} \
    --checkpoint ${CHECKPOINT} \
    --device cuda \
    --feat_dim 512 \
    --enc_hidden 256 \
    --lstm_hidden 128 \
    --budget_ratio 0.06 \
    --Bmin 3 \
    --Bmax 15 \
    --sample_stride 5 \
    --resize_w 320 \
    --resize_h 180 \
    --backend pyscenedetect \
    --threshold 27 \
    --embedder clip_vitb32

echo ""
echo "========================================="
echo "Step 2: Creating Cityscape Visualizations"
echo "========================================="

# Visualize with ALL probabilities (full cityscape)
echo "Creating full cityscape with all frame probabilities..."
python scripts/visualize_dsn_cityscape.py \
    --keyframes ${OUT_DIR}/all_probs.csv \
    --scenes ${OUT_DIR}/scenes.json \
    --style both \
    --width 20 \
    --height 5

# Also create visualization with only selected keyframes
echo ""
echo "Creating visualization with only selected keyframes..."
python scripts/visualize_dsn_cityscape.py \
    --keyframes ${OUT_DIR}/keyframes.csv \
    --scenes ${OUT_DIR}/scenes.json \
    --output ${OUT_DIR}/cityscape_selected_only.png \
    --style combined \
    --width 20 \
    --height 4

echo ""
echo "========================================="
echo "✅ Done! Check outputs in: ${OUT_DIR}"
echo "========================================="
echo "Generated files:"
echo "  - scenes.json: Scene detection results"
echo "  - keyframes.csv: Selected keyframes only"
echo "  - all_probs.csv: ALL frames with probabilities"
echo "  - cityscape_combined.png: Full cityscape (all frames)"
echo "  - cityscape_per_scene.png: Per-scene view (all frames)"
echo "  - cityscape_selected_only.png: Only selected keyframes"
