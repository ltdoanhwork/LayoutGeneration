#!/bin/bash
# ============================================================================
# Anime-CLIP-IQA Ablation Study (Enhanced)
# ============================================================================
#
# 5 Groups covering key ablation dimensions:
# 1. Tracks: A (Feature-only), B (Reward-only), C (Combined)
# 2. Reward Design: diversity weight + representativeness
# 3. Architecture: transformer layers, gating, LSTM
# 4. Budget & Exploration: budget size, entropy
# 5. Learning Rate: LR variations
#
# Usage: ./scripts/ablation_script/run_ablation_anime_clip_iqa.sh [gpu_id]
# ============================================================================

GPU_ID=${1:-0}
DATA_ROOT="data/sakuga_dataset_v11_new"
VAL_ROOT="data/sakuga_dataset_v11_new_test"
SAVE_ROOT="runs/ablation_anime_iqa"
EPOCHS=50

mkdir -p $SAVE_ROOT

echo "============================================================================"
echo "Anime-CLIP-IQA Ablation Study (5 Groups)"
echo "============================================================================"
echo "GPU:       $GPU_ID"
echo "Data:      $DATA_ROOT"
echo "Val Data:  $VAL_ROOT"
echo "Output:    $SAVE_ROOT"
echo "Epochs:    $EPOCHS"
echo "============================================================================"

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Limit CPU threads to prevent high CPU usage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Add project root to PYTHONPATH to find 'src'
export PYTHONPATH=$PYTHONPATH:$(pwd)

run_exp() {
    EXP_NAME=$1
    shift
    echo ""
    echo ">> Running: $EXP_NAME"
    echo "   Args: $@"
    
    python3 -m src.pipeline.train_rl_dsn_v11_final \
        --save_dir "$SAVE_ROOT/$EXP_NAME" \
        --dataset_root "$DATA_ROOT" \
        --val_root "$VAL_ROOT" \
        --epochs $EPOCHS \
        --device cuda \
        "$@" 2>&1 | tee "$SAVE_ROOT/${EXP_NAME}.log"
        
    [ ${PIPESTATUS[0]} -eq 0 ] && echo "   [OK] $EXP_NAME" || echo "   [FAIL] $EXP_NAME"
}

# ============================================================================
# GROUP 1: TRACKS (Original Anime-CLIP-IQA tracks)
# ============================================================================
echo -e "\n====== GROUP 1: TRACKS ======"

# Track A: Feature-only (CLIP + Anime attrs as input)
# Note: Optimizes generic reward (Rec + Div) but has access to Anime features
run_exp "1_track_A_features" --diversity_weight 0.3 --no_anime_reward --rec_weight 5.0

# Track B: Reward-only (CLIP only, but optimize for anime quality)
run_exp "1_track_B_reward" --diversity_weight 0.3 --no_anime_attrs

# Track C: Combined (CLIP + Anime attrs + anime-aware reward)
run_exp "1_track_C_combined" --diversity_weight 0.3


# ============================================================================
# GENERATE COMPARISON PLOTS
# ============================================================================
echo -e "\n====== GENERATING COMPARISONS ======"

python3 -m eval.compare_ablation \
    --ablation_root "$SAVE_ROOT" \
    --output_dir "$SAVE_ROOT/comparisons"

echo ""
echo "============================================================================"
echo "Anime-CLIP-IQA Ablation Complete!"
echo "Results:     $SAVE_ROOT"
echo "Comparisons: $SAVE_ROOT/comparisons"
echo "TensorBoard: tensorboard --logdir $SAVE_ROOT"
echo "============================================================================"
