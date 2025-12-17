#!/bin/bash
# ============================================================================
# V7 Dual-Objective Training Script
# ============================================================================
# Improves BOTH:
# - RecErr/Frechet (reconstruction quality)
# - Anime quality (CLIP-IQA attributes)
#
# Key features:
# - Curriculum learning: RecErr first → then add Anime
# - Enhanced diversity and coverage rewards for RecErr
# - Per-attribute optimization for all 6 CLIP-IQA dimensions
# - Balanced reward scaling between objectives
# ============================================================================

# Training parameters
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_v7_dual_objective"
VAL_VIDEOS_DIR="data/samples/Sakuga"
VAL_OUTPUT_DIR="${SAVE_DIR}/val_runs"
LOG_DIR="${SAVE_DIR}/logs"

# Model parameters
FEAT_DIM=512
ENC_HIDDEN=256
LSTM_HIDDEN=128
EPOCHS=75

# V7 Dual-Objective Rewards
REC_ERR_SCALE=3.0        # RecErr reward scale
FRECHET_SCALE=2.0        # Frechet distance scale
DIVERSITY_WEIGHT=1.0     # Diversity bonus weight
COVERAGE_WEIGHT=1.0      # Temporal coverage weight
ANIME_SCALE=2.5          # Anime quality scale (balanced with RecErr)
TOP_K_RATIO=0.1          # Top 10% quality frames
USE_CURRICULUM=1         # Enable curriculum learning

# PPO parameters
LR=2e-4
CLIP_RANGE=0.2
N_PPO_EPOCHS=4
ENTROPY_COEF=0.01
VF_COEF=0.5

# Budget
BUDGET_RATIO=0.06
BMIN=3
BMAX=15

# Device
DEVICE="cuda"
EVAL_DEVICE="cuda"

# Validation
VALIDATE_EVERY=5
EVAL_BACKEND="transnetv2"
EVAL_EMBEDDER="clip_vitb32"

# Create directories
mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR
mkdir -p $VAL_OUTPUT_DIR

echo "============================================================"
echo "V7 Dual-Objective Training"
echo "============================================================"
echo "Improves BOTH RecErr AND Anime Quality"
echo "------------------------------------------------------------"
echo "Dataset: $DATASET_ROOT"
echo "Save dir: $SAVE_DIR"
echo ""
echo "RecErr Settings:"
echo "  - rec_err_scale: $REC_ERR_SCALE"
echo "  - frechet_scale: $FRECHET_SCALE"
echo "  - diversity_weight: $DIVERSITY_WEIGHT"
echo "  - coverage_weight: $COVERAGE_WEIGHT"
echo ""
echo "Anime Settings:"
echo "  - anime_scale: $ANIME_SCALE"
echo "  - top_k_ratio: $TOP_K_RATIO"
echo ""
echo "Curriculum: Stage1 (RecErr) → Stage2 (Balance) → Stage3 (Full)"
echo "============================================================"

# Run training
python -m src.pipeline.train_rl_dsn_v7 \
    --dataset_root $DATASET_ROOT \
    --save_dir $SAVE_DIR \
    --epochs $EPOCHS \
    --device $DEVICE \
    --feat_dim $FEAT_DIM \
    --enc_hidden $ENC_HIDDEN \
    --lstm_hidden $LSTM_HIDDEN \
    --use_anime_attrs 1 \
    --anime_attrs_dim 6 \
    --use_raft_motion 1 \
    --motion_dim 128 \
    --rec_err_scale $REC_ERR_SCALE \
    --frechet_scale $FRECHET_SCALE \
    --diversity_weight $DIVERSITY_WEIGHT \
    --coverage_weight $COVERAGE_WEIGHT \
    --anime_scale $ANIME_SCALE \
    --top_k_ratio $TOP_K_RATIO \
    --use_curriculum $USE_CURRICULUM \
    --lr $LR \
    --clip_range $CLIP_RANGE \
    --n_ppo_epochs $N_PPO_EPOCHS \
    --entropy_coef $ENTROPY_COEF \
    --vf_coef $VF_COEF \
    --budget_ratio $BUDGET_RATIO \
    --Bmin $BMIN \
    --Bmax $BMAX \
    --log_dir $LOG_DIR \
    --val_videos_dir $VAL_VIDEOS_DIR \
    --val_output_dir $VAL_OUTPUT_DIR \
    --validate_every $VALIDATE_EVERY \
    --eval_backend $EVAL_BACKEND \
    --eval_embedder $EVAL_EMBEDDER \
    --eval_device $EVAL_DEVICE \
    --save_visualizations 1 \
    --eval_with_baselines

echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo "Checkpoints: $SAVE_DIR"
echo "  - best_rec_err.pt: Best RecErr model"
echo "  - best_anime.pt: Best Anime quality model"
echo ""
echo "TensorBoard: tensorboard --logdir=$LOG_DIR"
echo "Plots: ${SAVE_DIR}/plots"
echo "============================================================"
