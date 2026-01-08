#!/bin/bash

VIDEO_DIR="/home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga"
CHECKPOINT="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final/best.pt"
OUTPUT_DIR="outputs/inference_v11"
DEVICE="cuda"

mkdir -p "$OUTPUT_DIR"

# Lấy 10 video đầu tiên
ls "$VIDEO_DIR"/*.mp4 | head -n 100 | while read -r VIDEO_PATH; do
    echo "Running inference on: $VIDEO_PATH"

    python -m scripts.run_inference_v11 \
        --video_path "$VIDEO_PATH" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE"
done

echo "Running keyframe extraction..."
python repos/keyframe_extracter.py
