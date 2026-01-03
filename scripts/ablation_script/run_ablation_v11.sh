#!/bin/bash
# Run full V11 Ablation Study
# Usage: ./scripts/run_ablation_v11.sh [gpu_id]

GPU_ID=${1:-0}
DATASET="data/sakuga_dataset_v11"
VAL_ROOT="data/sakuga_test_precompute"
OUT_ROOT="runs/ablation_v11"
EPOCHS=60

mkdir -p $OUT_ROOT

echo "========================================================"
echo "Starting V11 Ablation on GPU $GPU_ID"
echo "Dataset: $DATASET"
echo "Output: $OUT_ROOT"
echo "========================================================"

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Function to run training
run_exp() {
    EXP_NAME=$1
    shift
    echo ">> Running Experiment: $EXP_NAME"
    echo "   Args: $@"
    
    python -m src.pipeline.train_rl_dsn_v11_enhanced \
        --save_dir "$OUT_ROOT/$EXP_NAME" \
        --dataset_root "$DATASET" \
        --val_root "$VAL_ROOT" \
        --epochs $EPOCHS \
        --device cuda \
        "$@" 2>&1 | tee "$OUT_ROOT/$EXP_NAME.log"
        
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "   [SUCCESS] $EXP_NAME"
    else
        echo "   [FAILED] $EXP_NAME (See log for details)"
    fi
}

# --- GROUP 1: REWARD COMPONENTS ---

# 1.0 Baseline (MPR + Div 0.3)
run_exp "1_baseline" --diversity_weight 0.3 --reward_mode mpr_div

# 1.1 No Diversity (MPR Only)
run_exp "1_no_div" --diversity_weight 0.0 --reward_mode mpr_only

# 1.2 No Quality (Diversity Only)
run_exp "1_no_quality" --reward_mode div_only

# 1.3 RecErr Optimization
run_exp "1_optimize_rec" --diversity_weight 0.3 --optimize_rec


# --- GROUP 2: INPUT FEATURES ---

# 2.1 No Anime Attributes (Visual Features Only)
run_exp "2_no_attrs" --diversity_weight 0.3 --no_anime_attrs --feat_dim 512


# --- GROUP 3: DIVERSITY WEIGHT SENSITIVITY ---

# 3.1 Weak Diversity (0.1)
run_exp "3_div_0.1" --diversity_weight 0.1

# 3.2 Strong Diversity (0.5)
run_exp "3_div_0.5" --diversity_weight 0.5

# 3.3 Very Strong Diversity (1.0)
run_exp "3_div_1.0" --diversity_weight 1.0


# --- GROUP 4: ARCHITECTURE ---

# 4.1 No Transformer (BiLSTM Only)
run_exp "4_no_transformer" --diversity_weight 0.3 --num_attn_layers 0

# 4.2 Fixed Gating (No Dynamic Mixing)
run_exp "4_fixed_gating" --diversity_weight 0.3 --fixed_gating


echo "========================================================"
echo "Ablation Study Completed."
echo "Results available in $OUT_ROOT"
echo "========================================================"
