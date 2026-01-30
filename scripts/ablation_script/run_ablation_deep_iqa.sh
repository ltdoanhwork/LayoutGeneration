#!/bin/bash
# ============================================================================
# Deep CLIP-IQA Ablation Study
# ============================================================================
# 
# This script covers:
# 1. Data Preparation: Generates anime_attrs for 1, 2, 3, 4, 5 prompt pairs.
# 2. Group 1: Prompt Sensitivity (Impact of prompt engineering).
# 3. Group 2: Component Contribution (Isolating Feat vs Reward).
#
# Usage: ./scripts/ablation_script/run_ablation_deep_iqa.sh [gpu_id] [stage]
# Stage: "all", "prep", "exp" (default: all)
# ============================================================================

GPU_ID=${1:-0}
STAGE=${2:-"all"}
# Sanitize input (remove whitespace/newline/return)
STAGE=$(echo "$STAGE" | tr -d '[:space:]')

echo "DEBUG: GPU_ID='$GPU_ID' STAGE='$STAGE'"

DATA_ROOT="data/sakuga_dataset_v11_new"
VAL_ROOT="data/sakuga_dataset_v11_new_test"
SAVE_ROOT="runs/ablation_deep_iqa"
EPOCHS=50

export CUDA_VISIBLE_DEVICES=$GPU_ID
export OMP_NUM_THREADS=4

mkdir -p $SAVE_ROOT

log() {
    echo -e "\n[$(date '+%H:%M:%S')] $1"
}

run_train() {
    EXP_NAME=$1
    shift
    log ">> Training: $EXP_NAME"
    python3 -m src.pipeline.train_rl_dsn_v11_final \
        --save_dir "$SAVE_ROOT/$EXP_NAME" \
        --dataset_root "$DATA_ROOT" \
        --val_root "$VAL_ROOT" \
        --epochs $EPOCHS \
        --device cuda \
        "$@" 2>&1 | tee "$SAVE_ROOT/${EXP_NAME}.log"
}

# ============================================================================
# STAGE 1: DATA PREPARATION
# ============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "prep" ]]; then
    log "====== STAGE 1: DATA PREPARATION (1-5 PAIRS) ======"
    
    # Run generation for 1, 2, 3, 4, 5 pairs
    # Note: 3-pair is roughly equivalent to original v11 but we regenerate to be sure
    for N in 1 2 3 4 5; do
        log "Generating attributes for ${N}-pair prompts..."
        python3 scripts/precompute_script/prepare_anime_attrs_ablation.py \
            --dataset_dir "$DATA_ROOT" \
            --num_pairs $N \
            --device cuda
            
        # Also generate for validation set
        log "Generating attributes for validation set (N=$N)..."
        python3 scripts/precompute_script/prepare_anime_attrs_ablation.py \
            --dataset_dir "$VAL_ROOT" \
            --num_pairs $N \
            --device cuda
    done
fi

# ============================================================================
# STAGE 2: EXPERIMENTS
# ============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "exp" ]]; then
    log "====== STAGE 2: RUNNING EXPERIMENTS ======"

    # Common Settings for Fair Comparison
    # RecErr=1.0 (Standard preservation)
    # Div=0.3 (Standard diversity)
    SETTINGS="--rec_weight 1.0 --diversity_weight 0.3"

    # ------------------------------------------------------------------------
    # GROUP 1: Prompt Sensitivity (Fix Method=Combined, Vary Prompts)
    # ------------------------------------------------------------------------
    log "--- Group 1: Prompt Sensitivity ---"
    
    # 1-Pair
    run_train "G1_prompt_1pair" $SETTINGS --attr_suffix "anime_attrs_1pair.npy"
    
    # 2-Pair
    run_train "G1_prompt_2pair" $SETTINGS --attr_suffix "anime_attrs_2pair.npy"
    
    # 3-Pair (Original Baseline level)
    run_train "G1_prompt_3pair" $SETTINGS --attr_suffix "anime_attrs_3pair.npy"
    
    # 4-Pair
    run_train "G1_prompt_4pair" $SETTINGS --attr_suffix "anime_attrs_4pair.npy"
    
    # 5-Pair (Augmented)
    run_train "G1_prompt_5pair" $SETTINGS --attr_suffix "anime_attrs_5pair.npy"

    # ------------------------------------------------------------------------
    # GROUP 2: Component Isolation (Fix Prompts=3pair, Vary Components)
    # Using 3-pair as the standard reference
    # ------------------------------------------------------------------------
    log "--- Group 2: Component Isolation ---"
    
    # Baseline: No CLIP-IQA at all (No feat, No reward)
    run_train "G2_baseline_plain" $SETTINGS --no_anime_attrs --no_anime_reward
    
    # Track A: Features Only (Input=CLIP+Anime, Reward=Rec+Div)
    run_train "G2_track_A_feat" $SETTINGS --attr_suffix "anime_attrs_3pair.npy" --no_anime_reward
    
    # Track B: Reward Only (Input=CLIP, Reward=Rec+Div+Anime)
    run_train "G2_track_B_reward" $SETTINGS --attr_suffix "anime_attrs_3pair.npy" --no_anime_attrs
    
    # Track C is technically "G1_prompt_3pair" (Combined), so we can symlink or skip
    # run_train "G2_track_C_combined" ... (Already ran as G1_prompt_3pair)

fi

log "Ablation Study Complete. Results in $SAVE_ROOT"
