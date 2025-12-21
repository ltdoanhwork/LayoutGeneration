#!/bin/bash
# Train DSN V4 (PPO + Actor-Critic + Reward Ensemble) for Anime Summary
# Scaled Up Version (4x Params)

# 1. Dataset & Paths
# -----------------------------------------------
DATASET_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_anime_v4_scaled/checkpoints"
LOG_DIR="runs/dsn_anime_v4_scaled/logs"
VAL_VIDEOS="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_100_samples"
VAL_OUTPUT="runs/dsn_anime_v4_scaled/val_runs"

# 2. Hyperparameters
# -----------------------------------------------
EPOCHS=30
DEVICE="cuda"

# --- Model Configuration (SCALED UP 4x) ---
# Previous: 256/128/2/4. New: 1024/512/6/8
ENC_HIDDEN=1024
LSTM_HIDDEN=512
NUM_ATTN_LAYERS=6
NUM_ATTN_HEADS=8
DROPOUT=0.3
VALUE_HIDDEN=512

# --- Optimization (Stabilized) ---
# Reduced LR, increased PPO epochs
LR=5e-5
N_PPO_EPOCHS=8
BATCH_SIZE=4
CLIP_RANGE=0.2
TARGET_KL=0.03
ENTROPY_COEF=0.02
MAX_GRAD_NORM=0.5

# --- Reward Weights (Focused on Improvement) ---
# Increased weights for RecErr and FD to force improvement
W_REC=2.0      # Increased from 1.0
W_FD=1.0       # Increased from 0.5
W_REP=0.5
W_DIV=0.5
W_PROBSEP=0.2

# Premium Anime Rewards
W_ANIME_LOOK=3.0
W_ANIME_SAKUGA=3.0
W_ANIME_STORY=1.5
W_TEMPORAL=0.8

# Curriculum
PERCENTILE_THRESHOLD=0.75
CONTRASTIVE_MARGIN=0.15
HARD_NEG_MARGIN=0.05
USE_CURRICULUM=1

# PPO Advanced
VF_COEF=0.5
# GAMMA/LAMBDA default in script are 0.99/0.95

# Ensemble
USE_REWARD_ENSEMBLE=1
N_ENSEMBLE=3

echo "=============================================="
echo "Starting V4 SCALED-UP Training"
echo "Model: ${ENC_HIDDEN} hidden, ${NUM_ATTN_LAYERS} layers"
echo "LR: ${LR}, PPO Epochs: ${N_PPO_EPOCHS}"
echo "Weights: Rec=${W_REC}, FD=${W_FD}, Look=${W_ANIME_LOOK}"
echo "=============================================="

python -m src.pipeline.train_rl_dsn_v4 \
    --dataset_root "$DATASET_ROOT" \
    --save_dir "$SAVE_DIR" \
    --log_dir "$LOG_DIR" \
    --epochs $EPOCHS \
    --device "$DEVICE" \
    --multi_video 1 \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --model_type "advanced" \
    --feat_dim 512 \
    --enc_hidden $ENC_HIDDEN \
    --lstm_hidden $LSTM_HIDDEN \
    --num_attn_layers $NUM_ATTN_LAYERS \
    --num_attn_heads $NUM_ATTN_HEADS \
    --dropout $DROPOUT \
    --value_hidden_dim $VALUE_HIDDEN \
    --use_lstm_in_advanced 1 \
    --use_cache 1 \
    --use_raft_motion 1 \
    --motion_dim 128 \
    --use_anime_attrs 1 \
    --anime_attrs_dim 6 \
    --clip_range $CLIP_RANGE \
    --target_kl $TARGET_KL \
    --n_ppo_epochs $N_PPO_EPOCHS \
    --entropy_coef $ENTROPY_COEF \
    --vf_coef $VF_COEF \
    --w_rec $W_REC \
    --w_fd $W_FD \
    --w_rep $W_REP \
    --w_div $W_DIV \
    --w_probsep $W_PROBSEP \
    --w_anime_look $W_ANIME_LOOK \
    --w_anime_sakuga $W_ANIME_SAKUGA \
    --w_anime_story $W_ANIME_STORY \
    --w_temporal $W_TEMPORAL \
    --use_reward_ensemble $USE_REWARD_ENSEMBLE \
    --n_ensemble $N_ENSEMBLE \
    --percentile_threshold $PERCENTILE_THRESHOLD \
    --contrastive_margin $CONTRASTIVE_MARGIN \
    --hard_negative_margin $HARD_NEG_MARGIN \
    --use_curriculum $USE_CURRICULUM \
    --val_videos_dir "$VAL_VIDEOS" \
    --val_output_dir "$VAL_OUTPUT" \
    --validate_every 2 \
    --max_grad_norm $MAX_GRAD_NORM \
    --eval_with_baselines 
