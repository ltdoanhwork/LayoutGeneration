#!/bin/bash

# Benchmark: 10 videos from Sakuga_test with timing measurement
# Using layout_decomposer_final.py with grid layout

# Activate conda yoloe environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate yoloe

cd /home/serverai/ltdoanh/LayoutGeneration

# Define 10 test videos from Sakuga_test
VIDEOS=(
    "data/samples/Sakuga_test/70025.mp4"
    "data/samples/Sakuga_test/70039.mp4"
    "data/samples/Sakuga_test/70064.mp4"
    "data/samples/Sakuga_test/70076.mp4"
    "data/samples/Sakuga_test/70115.mp4"
    "data/samples/Sakuga_test/70144.mp4"
    "data/samples/Sakuga_test/70165.mp4"
    "data/samples/Sakuga_test/70207.mp4"
    "data/samples/Sakuga_test/70263.mp4"
    "data/samples/Sakuga_test/70396.mp4"
)

# Use single layout for benchmark (cars/15.jpg - medium complexity)
LAYOUT="repos/Colla/input_data/image_collections/cars/15.jpg"

# Output base directory
OUTPUT_BASE="outputs/test_benchmark"
mkdir -p "$OUTPUT_BASE"

# Checkpoint path
CHECKPOINT="runs/dsn_advanced_v1/dsn_checkpoint_ep16.pt"

# Timing results file
TIMING_FILE="${OUTPUT_BASE}/timing_results.csv"
echo "video_name,duration_seconds,status,keyframes,scenes" > "$TIMING_FILE"

# Counter
count=0
total=${#VIDEOS[@]}
total_time=0

echo "========================================"
echo "BENCHMARK: 10 videos from Sakuga_test"
echo "Layout: cars/15.jpg"
echo "Total videos: $total"
echo "========================================"
echo ""

# Run each video
for video in "${VIDEOS[@]}"; do
    video_name=$(basename "$video" .mp4)
    count=$((count + 1))
    
    echo ""
    echo "========================================"
    echo "[$count/$total] Video: $video_name"
    echo "========================================"
    
    # Create output directory name
    out_dir="${OUTPUT_BASE}/v${video_name}"
    
    # Get start time
    start_time=$(date +%s.%N)
    
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
        --input_shape_layout "$LAYOUT" \
        --scaling_factor 2 \
        --use_grid_layout \
        --use_object_detection \
        --debug
    
    exit_code=$?
    
    # Get end time and calculate duration
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    total_time=$(echo "$total_time + $duration" | bc)
    
    # Check results
    if [ $exit_code -eq 0 ]; then
        status="SUCCESS"
        # Count keyframes and scenes
        keyframes=$(ls "$out_dir/keyframes"/*.jpg 2>/dev/null | wc -l)
        scenes=$(cat "$out_dir/scenes.json" 2>/dev/null | grep -c '"scene_id"' || echo "0")
        echo "[SUCCESS] $video_name completed in ${duration}s (keyframes: $keyframes, scenes: $scenes)"
    else
        status="FAILED"
        keyframes=0
        scenes=0
        echo "[FAILED] $video_name (exit code: $exit_code)"
    fi
    
    # Record timing
    echo "$video_name,$duration,$status,$keyframes,$scenes" >> "$TIMING_FILE"
    
    echo ""
done

# Calculate statistics
avg_time=$(echo "scale=2; $total_time / $total" | bc)

echo ""
echo "========================================"
echo "BENCHMARK COMPLETED!"
echo "========================================"
echo "Total videos: $total"
echo "Total time: ${total_time}s"
echo "Average time per video: ${avg_time}s"
echo ""
echo "Timing results saved to: $TIMING_FILE"
echo ""

# Print summary table
echo "Summary:"
echo "--------"
cat "$TIMING_FILE"
echo ""

# Count successes
success_count=$(grep -c "SUCCESS" "$TIMING_FILE" || echo "0")
echo "Success rate: $success_count / $total"

# Check collages
echo ""
echo "Generated collages:"
ls -la "$OUTPUT_BASE"/v*/colla_layout/collage.png 2>/dev/null | wc -l
