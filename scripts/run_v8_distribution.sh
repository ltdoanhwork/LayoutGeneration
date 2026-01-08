#!/bin/bash
# Run distribution analysis with V8 checkpoint on Sakuga test set

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate sam

# Set paths
CHECKPOINT="/home/serverai/ltdoanh/LayoutGeneration/runs/dsn_v8_constrained/best_anime.pt"
VIDEOS_DIR="/home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test"
OUTPUT_DIR="/home/serverai/ltdoanh/LayoutGeneration/outputs/v8_distribution_sakuga_test"

# Run inference distribution visualization
python -m eval.inference_distribution \
    --checkpoint "$CHECKPOINT" \
    --videos_dir "$VIDEOS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --sample_stride 5 \
    --resize_w 320 \
    --resize_h 180 \
    --budget_ratio 0.06 \
    --Bmin 3 \
    --Bmax 15

echo ""
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
