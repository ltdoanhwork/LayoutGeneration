#!/bin/bash

# Batch run: 5 videos x 3 layouts = 15 combinations
# Using layout_decomposer_final.py with grid layout

# Activate conda yoloe environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate yoloe

cd /home/serverai/ltdoanh/LayoutGeneration

# Define videos (5 selected from Sakuga)
VIDEOS=(
    "data/samples/Sakuga/10049.mp4"
    "data/samples/Sakuga/11021.mp4"
    "data/samples/Sakuga/122288.mp4"
    "data/samples/Sakuga/124133.mp4"
    "data/samples/Sakuga/13925.mp4"
)

# Define layouts (3 different car shapes)
LAYOUTS=(
    "repos/Colla/input_data/image_collections/cars/01.jpg"
    "repos/Colla/input_data/image_collections/cars/15.jpg"
    "repos/Colla/input_data/image_collections/cars/27.jpg"
)

# Output base directory
OUTPUT_BASE="outputs/batch_experiments"
mkdir -p "$OUTPUT_BASE"

# Checkpoint path (using latest trained checkpoint)
CHECKPOINT="runs/dsn_advanced_v1/dsn_checkpoint_ep16.pt"

# Counter
count=0
total=$((${#VIDEOS[@]} * ${#LAYOUTS[@]}))

echo "========================================"
echo "BATCH EXPERIMENT: 5 videos x 3 layouts"
echo "Total combinations: $total"
echo "========================================"
echo ""

# Run each combination
for video in "${VIDEOS[@]}"; do
    video_name=$(basename "$video" .mp4)
    
    for layout in "${LAYOUTS[@]}"; do
        layout_name=$(basename "$layout" .jpg)
        count=$((count + 1))
        
        echo ""
        echo "========================================"
        echo "[$count/$total] Video: $video_name, Layout: $layout_name"
        echo "========================================"
        
        # Create output directory name
        out_dir="${OUTPUT_BASE}/v${video_name}_l${layout_name}"
        
        # Run pipeline
        python layout_decomposer_final.py \
            --video "$video" \
            --out_dir "$out_dir" \
            --checkpoint "$CHECKPOINT" \
            --device cuda \
            --embedder clip_vitb32 \
            --backend transnetv2 \
            --model_dir src/models/TransNetV2 \
            --prob_threshold 0.5 \
            --scene_device cuda \
            --min_scene_len 24 \
            --budget_ratio 0.08 --Bmin 5 --Bmax 20 \
            --sample_stride 3 \
            --resize_w 320 --resize_h 180 \
            --input_shape_layout "$layout" \
            --scaling_factor 2 \
            --use_grid_layout \
            --use_object_detection \
            --debug
        
        # Check if successful
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] $video_name + $layout_name completed!"
        else
            echo "[FAILED] $video_name + $layout_name"
        fi
        
        echo ""
    done
done

echo ""
echo "========================================"
echo "BATCH EXPERIMENT COMPLETED!"
echo "Results saved to: $OUTPUT_BASE"
echo "========================================"

# Generate summary
echo ""
echo "Summary of outputs:"
ls -la "$OUTPUT_BASE"/*/colla_layout/collage.png 2>/dev/null | wc -l
echo "collages generated"
