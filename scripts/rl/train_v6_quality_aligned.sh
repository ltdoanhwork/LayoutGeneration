#!/bin/bash
# ============================================================================
# V6 Quality-Aligned Training Script
# ============================================================================
# Key improvements over V5:
# - Continuous quality rewards (not binary percentile)
# - Per-attribute optimization for all 6 CLIP-IQA dimensions
# - Top-k coverage rewards to ensure best frames selected
# - Balanced reward scaling between RecErr and Anime tasks
# - Enhanced visualizations with quality improvement plots
# ============================================================================

# Training parameters
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_v6_quality_aligned"
VAL_VIDEOS_DIR="data/samples/Sakuga"
VAL_OUTPUT_DIR="${SAVE_DIR}/val_runs"
LOG_DIR="${SAVE_DIR}/logs"

# Model parameters
FEAT_DIM=512
ENC_HIDDEN=256
LSTM_HIDDEN=128
EPOCHS=75

# V6 Quality-Aligned rewards
ANIME_REWARD_SCALE=2.5    # Scale anime rewards to match RecErr magnitude
TOP_K_RATIO=0.1           # Consider top 10% as high-quality frames
USE_CONTINUOUS_REWARDS=1  # Use continuous quality rewards
USE_CURRICULUM=1          # Enable curriculum learning

# PPO parameters
LR=2e-4
CLIP_RANGE=0.2
N_PPO_EPOCHS=6
ENTROPY_COEF=0.02
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
echo "V6 Quality-Aligned Training"
echo "============================================================"
echo "Dataset: $DATASET_ROOT"
echo "Save dir: $SAVE_DIR"
echo "Anime reward scale: $ANIME_REWARD_SCALE"
echo "Top-k ratio: $TOP_K_RATIO"
echo "Continuous rewards: $USE_CONTINUOUS_REWARDS"
echo "============================================================"

# Run training
python -m src.pipeline.train_rl_dsn_v6 \
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
    --lr $LR \
    --clip_range $CLIP_RANGE \
    --n_ppo_epochs $N_PPO_EPOCHS \
    --entropy_coef $ENTROPY_COEF \
    --vf_coef $VF_COEF \
    --budget_ratio $BUDGET_RATIO \
    --Bmin $BMIN \
    --Bmax $BMAX \
    --anime_reward_scale $ANIME_REWARD_SCALE \
    --top_k_ratio $TOP_K_RATIO \
    --use_continuous_rewards $USE_CONTINUOUS_REWARDS \
    --use_curriculum $USE_CURRICULUM \
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
echo "Checkpoints saved to: $SAVE_DIR"
echo "TensorBoard logs: $LOG_DIR"
echo "Visualizations: ${SAVE_DIR}/outputs/visualizations"
echo "============================================================"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=$LOG_DIR"
echo ""
echo "To generate final visualizations:"
echo "  python -m eval.visualize_validation --val_output_dir $VAL_OUTPUT_DIR --save_images"
