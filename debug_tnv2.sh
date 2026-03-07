#!/bin/bash
# Debug script for TNv2 Fixed Short dataset generation
export CUDA_VISIBLE_DEVICES=0
PYTHON="/srv/conda/envs/serverai/sam/bin/python"
VIDEO_DIR="data/samples/Sakuga"
OUT_DIR="data/debug_tnv2_short"

echo "Running Debug for TNv2 Fixed Short..."
$PYTHON -m scripts.precompute_script.prepare_rl_dataset_v11 \
    --video_dir "$VIDEO_DIR" \
    --out_dir "$OUT_DIR" \
    --backend "transnetv2" \
    --min_scene_len 30 \
    --max_scene_len 100 \
    --force_split \
    --device cuda \
    --model_dir "src/models/TransNetV2"

echo "Done. Check logs."
