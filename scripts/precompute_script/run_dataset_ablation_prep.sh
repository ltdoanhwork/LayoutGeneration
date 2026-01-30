#!/bin/bash
# =========================================================================================
# Dataset Preparation for Ablation Study
# Generates variants of Sakuga dataset with different scene detectors and length strategies.
# =========================================================================================
#
# Variants:
# 1. TransNetV2 Fixed Short: min=30, max=100, force_split
# 2. TransNetV2 Fixed Long: min=150, max=300, force_split
# 3. PySceneDetect Diverse: min=30, max=500
# 4. PySceneDetect Fixed Short: min=30, max=100, force_split
#
# Input Data:
#   Train: data/samples/Sakuga
#   Test:  data/samples/Sakuga_test
#
# Output Data:
#   data/sakuga_dataset_v11_*
#   data/sakuga_dataset_v11_*_test
# =========================================================================================

# =========================================================================================

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
PYTHON="/srv/conda/envs/serverai/sam/bin/python"

# Paths
SRC_TRAIN="data/samples/Sakuga"
SRC_TEST="data/samples/Sakuga_test"

# Function to run prep
run_prep() {
    SUFFIX=$1
    BACKEND=$2
    MIN_LEN=$3
    MAX_LEN=$4
    EXTRA_ARGS=$5
    
    echo ""
    echo "============================================================"
    echo "Generating Dataset: $SUFFIX"
    echo "Backend: $BACKEND | Min: $MIN_LEN | Max: $MAX_LEN | Args: $EXTRA_ARGS"
    echo "============================================================"
    
    # Train Set
    OUT_TRAIN="data/sakuga_dataset_v11_${SUFFIX}"
    if [ -d "$OUT_TRAIN" ]; then
        echo "  [SKIP] Train dataset exists: $OUT_TRAIN"
    else
        echo "  >> Processing Train Set..."
        $PYTHON -m scripts.precompute_script.prepare_rl_dataset_v11 \
            --video_dir "$SRC_TRAIN" \
            --out_dir "$OUT_TRAIN" \
            --backend "$BACKEND" \
            --min_scene_len "$MIN_LEN" \
            --max_scene_len "$MAX_LEN" \
            --device cpu \
            $EXTRA_ARGS
    fi
    
    # Test Set
    OUT_TEST="data/sakuga_dataset_v11_${SUFFIX}_test"
    if [ -d "$OUT_TEST" ]; then
        echo "  [SKIP] Test dataset exists: $OUT_TEST"
    else
        echo "  >> Processing Test Set..."
        $PYTHON -m scripts.precompute_script.prepare_rl_dataset_v11 \
            --video_dir "$SRC_TEST" \
            --out_dir "$OUT_TEST" \
            --backend "$BACKEND" \
            --min_scene_len "$MIN_LEN" \
            --max_scene_len "$MAX_LEN" \
            --device cpu \
            $EXTRA_ARGS
    fi
}

# ============================================================
# 1. TransNetV2 Fixed Short
# ============================================================
run_prep "tnv2_short" "transnetv2" 30 100 "--force_split"

# ============================================================
# 2. TransNetV2 Fixed Long
# ============================================================
run_prep "tnv2_long" "transnetv2" 150 300 "--force_split"

# ============================================================
# 3. PySceneDetect Diverse
# ============================================================
run_prep "pyscene_div" "pyscenedetect" 30 500 ""

# ============================================================
# 4. PySceneDetect Fixed Short
# ============================================================
run_prep "pyscene_short" "pyscenedetect" 30 100 "--force_split"

echo ""
echo "All Datasets Prepared!"
