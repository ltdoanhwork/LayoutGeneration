#!/bin/bash
# ============================================================
# V11 Final Reorganized Ablation Study
# ============================================================
# 
# Baseline Model: training_v11_recerr_w0.2 (RecErr Opt)
# - Div: 0.3, Rec: 0.2, Frechet: 0.0
# - Attn: 2 Layers
# - Budget: 15%
# - Dataset: TransNetV2 Diverse (sakuga_dataset_v11_new)
#
# Groups:
# 1. Reward Design
# 2. Input Signals
# 3. Architecture
# 4. RL Budget
# 5. Entropy Reg
# 6. Dataset & Scene Length (New)
#
# Usage: ./scripts/ablation_script/run_ablation_final_reorg.sh [gpu_id] [--dry-run]
# ============================================================

GPU_ID=${1:-0}
DRY_RUN=${2:-""}

DATASET_DEFAULT="data/sakuga_dataset_v11_new"
VAL_ROOT_DEFAULT="data/sakuga_dataset_v11_new_test"
OUT_ROOT="runs/ablation_final_reorg"
EPOCHS=50

mkdir -p $OUT_ROOT

echo "============================================================"
echo "V11 Final Ablation Study (Reorganized)"
echo "============================================================"
echo "GPU:       $GPU_ID"
echo "Output:    $OUT_ROOT"
echo "Epochs:    $EPOCHS"
echo "Dry Run:   $DRY_RUN"
echo "============================================================"

# Auto-prepare datasets if not in dry-run
if [ "$DRY_RUN" != "--dry-run" ]; then
    echo ">> Checking/Preparing Datasets..."
    bash scripts/precompute_script/run_dataset_ablation_prep.sh $GPU_ID
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Limit CPU threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

PYTHON="/srv/conda/envs/serverai/sam/bin/python"

run_exp() {
    EXP_NAME=$1
    shift
    
    # Check if we are changing dataset for this run, otherwise use default
    local DATASET_ARG="$DATASET_DEFAULT"
    local VAL_ARG="$VAL_ROOT_DEFAULT"
    
    # Simple check if arguments contain --dataset_root to override
    # Actually, python script takes last arg priority usually, but better to be safe.
    # We will pass defaults *before* "$@" so "$@" can override them.
    
    echo ""
    echo ">> Running: $EXP_NAME"
    echo "   Args: $@"
    
    if [ "$DRY_RUN" == "--dry-run" ]; then
        echo "   [DRY-RUN] Would execute:"
        echo "   $PYTHON -m src.pipeline.train_rl_dsn_v11_final --save_dir $OUT_ROOT/$EXP_NAME --dataset_root $DATASET_DEFAULT --val_root $VAL_ARG --epochs $EPOCHS --device cuda $@"
        return
    fi
    
    $PYTHON -m src.pipeline.train_rl_dsn_v11_final \
        --save_dir "$OUT_ROOT/$EXP_NAME" \
        --dataset_root "$DATASET_DEFAULT" \
        --val_root "$VAL_ARG" \
        --epochs $EPOCHS \
        --device cuda \
        "$@" 2>&1 | tee "$OUT_ROOT/${EXP_NAME}.log"
        
    [ ${PIPESTATUS[0]} -eq 0 ] && echo "   [OK] $EXP_NAME" || echo "   [FAIL] $EXP_NAME"
}

# Wrapper for dataset variation experiments where we MUST override defaults explicitly
run_exp_dataset() {
    EXP_NAME=$1
    DS_TRAIN=$2
    DS_TEST=$3
    shift 3
    
    echo ""
    echo ">> Running Dataset Exp: $EXP_NAME"
    echo "   Train: $DS_TRAIN"
    echo "   Test:  $DS_TEST"
    
    if [ ! -d "$DS_TRAIN" ] || [ ! -d "$DS_TEST" ]; then
        echo "   [WARNING] Dataset missing! Please run scripts/precompute_script/run_dataset_ablation_prep.sh"
        if [ "$DRY_RUN" != "--dry-run" ]; then
             echo "   Skipping..."
             return
        fi
    fi
    
    if [ "$DRY_RUN" == "--dry-run" ]; then
        echo "   [DRY-RUN] Would execute with dataset override..."
        return
    fi
    
    $PYTHON -m src.pipeline.train_rl_dsn_v11_final \
        --save_dir "$OUT_ROOT/$EXP_NAME" \
        --dataset_root "$DS_TRAIN" \
        --val_root "$DS_TEST" \
        --epochs $EPOCHS \
        --device cuda \
        "$@" 2>&1 | tee "$OUT_ROOT/${EXP_NAME}.log"
        
    [ ${PIPESTATUS[0]} -eq 0 ] && echo "   [OK] $EXP_NAME" || echo "   [FAIL] $EXP_NAME"
}

# ============================================================
# BASELINE CONFIGURATION ARGS (For reference, applied via defaults or overrides)
# --diversity_weight 0.3 --rec_weight 0.2 --frechet_weight 0.0
# --num_attn_layers 2 --budget_ratio 0.15 --entropy_coef 0.02
# ============================================================

BASE_ARGS="--diversity_weight 0.3 --rec_weight 0.2 --frechet_weight 0.0"

# ============================================================
# GROUP 1: REWARD DESIGN
# ============================================================
echo -e "\n====== GROUP 1: REWARD DESIGN ======"

# Baseline (Div Only) - Removes RecErr
run_exp "1_baseline_div_only" --diversity_weight 0.3 --rec_weight 0.0 --frechet_weight 0.0

# No Diversity - Removes Div, Rec? (Table says "No Diversity", typically implies Div=0)
# Assuming it retains RecErr if RecErr is part of "Ours", but let's follow the plan:
# No Diversity: Div=0, Rec=0, Frechet=0 (Pure Visual/Aes?)
run_exp "1_no_diversity" --diversity_weight 0.0 --rec_weight 0.0 --frechet_weight 0.0

# Strong Diversity
run_exp "1_strong_diversity" --diversity_weight 1.0 --rec_weight 0.0 --frechet_weight 0.0 --entropy_coef 0.05

# RecErr Opt (Ours - Main Baseline)
run_exp "1_rec_opt_ours" $BASE_ARGS

# Frechet Opt
run_exp "1_frechet_opt" --diversity_weight 0.3 --rec_weight 0.0 --frechet_weight 1.0

# Combined Rep
run_exp "1_combined_rep" --diversity_weight 0.3 --rec_weight 0.2 --frechet_weight 0.5


# ============================================================
# GROUP 2: INPUT SIGNALS (Ref: Ours)
# ============================================================
echo -e "\n====== GROUP 2: INPUT SIGNALS ======"

# Visual Features Only
run_exp "2_visual_only" $BASE_ARGS --no_anime_attrs

# Full (Visual + Aes) -> Is Ours Default
run_exp "2_full_features" $BASE_ARGS


# ============================================================
# GROUP 3: ARCHITECTURE (Ref: Ours)
# ============================================================
echo -e "\n====== GROUP 3: ARCHITECTURE ======"

run_exp "3_no_attn"   $BASE_ARGS --num_attn_layers 0
run_exp "3_attn_1L"   $BASE_ARGS --num_attn_layers 1
run_exp "3_attn_2L"   $BASE_ARGS --num_attn_layers 2 # (Redundant to Ours, but explicit)
run_exp "3_attn_4L"   $BASE_ARGS --num_attn_layers 4

run_exp "3_gate_small" $BASE_ARGS --gating_hidden 32
run_exp "3_gate_large" $BASE_ARGS --gating_hidden 128


# ============================================================
# GROUP 4: RL BUDGET (Ref: Ours)
# ============================================================
echo -e "\n====== GROUP 4: RL BUDGET ======"

run_exp "4_budget_10" $BASE_ARGS --budget_ratio 0.10 --Bmin 2 --Bmax 8
run_exp "4_budget_15" $BASE_ARGS --budget_ratio 0.15 # (Redundant to Ours)
run_exp "4_budget_25" $BASE_ARGS --budget_ratio 0.25 --Bmin 5 --Bmax 20


# ============================================================
# GROUP 5: ENTROPY REG (Ref: Ours)
# ============================================================
echo -e "\n====== GROUP 5: ENTROPY REG ======"

run_exp "5_low_entropy"  $BASE_ARGS --entropy_coef 0.005
run_exp "5_high_entropy" $BASE_ARGS --entropy_coef 0.05


# ============================================================
# GROUP 6: DATASET & SCENE LENGTH (New)
# ============================================================
echo -e "\n====== GROUP 6: DATASET & SCENE LENGTH ======"

# A. TransNetV2
# TNv2 Diverse (Base) -> Ours Default (Already run as 1_rec_opt_ours, but good to compare)
run_exp "6_tnv2_diverse" $BASE_ARGS

# TNv2 Fixed Short
run_exp_dataset "6_tnv2_fixed_short" \
    "data/sakuga_dataset_v11_tnv2_short" \
    "data/sakuga_dataset_v11_tnv2_short_test" \
    $BASE_ARGS

# TNv2 Fixed Long
run_exp_dataset "6_tnv2_fixed_long" \
    "data/sakuga_dataset_v11_tnv2_long" \
    "data/sakuga_dataset_v11_tnv2_long_test" \
    $BASE_ARGS

# B. PySceneDetect
# PyScene Diverse
run_exp_dataset "6_pyscene_diverse" \
    "data/sakuga_dataset_v11_pyscene_div" \
    "data/sakuga_dataset_v11_pyscene_div_test" \
    $BASE_ARGS

# PyScene Fixed Short
run_exp_dataset "6_pyscene_fixed_short" \
    "data/sakuga_dataset_v11_pyscene_short" \
    "data/sakuga_dataset_v11_pyscene_short_test" \
    $BASE_ARGS


echo ""
echo "============================================================"
echo "Ablation Complete! Results: $OUT_ROOT"
echo "============================================================"
