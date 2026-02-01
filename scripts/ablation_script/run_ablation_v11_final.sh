#!/bin/bash
# ============================================================
# V11 Final Ablation Study (Consolidated)
# ============================================================
# 
# 5 Groups covering all key ablation dimensions:
# 1. Reward Design: diversity weight + representativeness
# 2. Input & Features: anime attrs, feature dims
# 3. Architecture: transformers, gating, LSTM
# 4. Budget & Exploration: budget size, entropy
# 5. Learning Rate: LR variations
#
# Usage: ./scripts/ablation_script/run_ablation_v11_final.sh [gpu_id]
# ============================================================

GPU_ID=${1:-0}
DATASET="data/sakuga_dataset_v11_new"
VAL_ROOT="data/sakuga_dataset_v11_new_test"
OUT_ROOT="runs/ablation_v11_final"
EPOCHS=50

mkdir -p $OUT_ROOT

echo "============================================================"
echo "V11 Final Ablation Study (5 Groups)"
echo "============================================================"
echo "GPU:       $GPU_ID"
echo "Dataset:   $DATASET"
echo "Val Data:  $VAL_ROOT"
echo "Output:    $OUT_ROOT"
echo "Epochs:    $EPOCHS"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Limit CPU threads
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
        --save_dir "$OUT_ROOT/$EXP_NAME" \
        --dataset_root "$DATASET" \
        --val_root "$VAL_ROOT" \
        --epochs $EPOCHS \
        --device cuda \
        "$@" 2>&1 | tee "$OUT_ROOT/${EXP_NAME}.log"
        
    [ ${PIPESTATUS[0]} -eq 0 ] && echo "   [OK] $EXP_NAME" || echo "   [FAIL] $EXP_NAME"
}

# ============================================================
# GROUP 1: REWARD DESIGN (Diversity + Representativeness)
# ============================================================
echo -e "\n====== GROUP 1: REWARD DESIGN ======"

run_exp "1_baseline" --diversity_weight 0.3
run_exp "1_no_div" --diversity_weight 0.0
run_exp "1_strong_div" --diversity_weight 1.0
run_exp "1_rec_opt" --diversity_weight 0.3 --rec_weight 1.0
run_exp "1_frechet_opt" --diversity_weight 0.3 --frechet_weight 1.0
run_exp "1_combined_rep" --diversity_weight 0.3 --rec_weight 1.0 --frechet_weight 0.5


# ============================================================
# GROUP 2: INPUT & FEATURES
# ============================================================
echo -e "\n====== GROUP 2: INPUT & FEATURES ======"

run_exp "2_full_features" --diversity_weight 0.3
run_exp "2_visual_only" --diversity_weight 0.3 --no_anime_attrs
run_exp "2_low_feat_dim" --diversity_weight 0.3 --feat_dim 256
run_exp "2_high_feat_dim" --diversity_weight 0.3 --feat_dim 768


# ============================================================
# GROUP 3: ARCHITECTURE (Transformer + Gating + LSTM)
# ============================================================
echo -e "\n====== GROUP 3: ARCHITECTURE ======"

# Transformer layers
run_exp "3_no_attn" --diversity_weight 0.3 --num_attn_layers 0
run_exp "3_attn_1L" --diversity_weight 0.3 --num_attn_layers 1
run_exp "3_attn_2L" --diversity_weight 0.3 --num_attn_layers 2
run_exp "3_attn_4L" --diversity_weight 0.3 --num_attn_layers 4

# Gating network
run_exp "3_gate_small" --diversity_weight 0.3 --gating_hidden 32
run_exp "3_gate_large" --diversity_weight 0.3 --gating_hidden 128

# LSTM
run_exp "3_lstm_small" --diversity_weight 0.3 --lstm_hidden 64
run_exp "3_lstm_large" --diversity_weight 0.3 --lstm_hidden 256


# ============================================================
# GROUP 4: BUDGET & EXPLORATION
# ============================================================
echo -e "\n====== GROUP 4: BUDGET & EXPLORATION ======"

# Budget
run_exp "4_budget_low" --diversity_weight 0.3 --budget_ratio 0.10 --Bmin 2 --Bmax 8
run_exp "4_budget_default" --diversity_weight 0.3 --budget_ratio 0.15 --Bmin 3 --Bmax 15
run_exp "4_budget_high" --diversity_weight 0.3 --budget_ratio 0.25 --Bmin 5 --Bmax 20

# Entropy (Exploration)
run_exp "4_low_entropy" --diversity_weight 0.3 --entropy_coef 0.005
run_exp "4_high_entropy" --diversity_weight 0.3 --entropy_coef 0.05


# ============================================================
# GROUP 5: LEARNING RATE
# ============================================================
echo -e "\n====== GROUP 5: LEARNING RATE ======"

run_exp "5_lr_1e-5" --diversity_weight 0.3 --lr 1e-5
run_exp "5_lr_1e-4" --diversity_weight 0.3 --lr 1e-4
run_exp "5_lr_5e-4" --diversity_weight 0.3 --lr 5e-4


# ============================================================
# GENERATE COMPARISON PLOTS
# ============================================================
echo -e "\n====== GENERATING COMPARISONS ======"

python3 -m eval.compare_ablation \
    --ablation_root "$OUT_ROOT" \
    --output_dir "$OUT_ROOT/comparisons"

echo ""
echo "============================================================"
echo "Ablation Complete! Results: $OUT_ROOT"
echo "TensorBoard: tensorboard --logdir $OUT_ROOT"
echo "============================================================"
