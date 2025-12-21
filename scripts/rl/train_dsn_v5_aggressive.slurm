#!/bin/bash
#SBATCH --job-name=dsn_v5
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu

# ============================================
# V5 - AGGRESSIVE IMPROVEMENT
# ============================================
# Goal: Significant improve on ALL metrics:
# - RecErr < 0.30
# - Frechet < 0.50
# - Anime scores > current
#
# Strategy:
# - VERY HIGH anime weights (direct optimization)
# - Heavy RecErr/Frechet focus
# - Longer training (80 epochs)
# - Higher exploration (entropy)
# - Smaller model for faster convergence
# ============================================

cd /home/serverai/ltdoanh/LayoutGeneration

# Activate environment
source /srv/conda/envs/serverai/sam/bin/activate

mkdir -p logs

# Paths
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_v5_aggressive/checkpoints"
LOG_DIR="runs/dsn_v5_aggressive/logs"
VAL_VIDEOS="data/samples/Sakuga_test"
VAL_OUTPUT="runs/dsn_v5_aggressive/val_runs"

# Training
EPOCHS=80
DEVICE="cuda"

# --- Model (SMALLER for better convergence) ---
ENC_HIDDEN=256
LSTM_HIDDEN=128
NUM_ATTN_LAYERS=2
NUM_ATTN_HEADS=4
DROPOUT=0.1
VALUE_HIDDEN=128

# --- Optimization ---
LR=3e-4
N_PPO_EPOCHS=10
BATCH_SIZE=16
CLIP_RANGE=0.1
TARGET_KL=0.015
ENTROPY_COEF=0.03   # Higher exploration
MAX_GRAD_NORM=0.5

# --- Reward Weights (AGGRESSIVE) ---
# Primary focus: RecErr + Frechet
W_REC=10.0     # VERY HIGH
W_FD=5.0       # HIGH
W_REP=2.0
W_DIV=1.0
W_PROBSEP=0.2

# --- Anime Rewards (VERY HIGH) ---
W_ANIME_LOOK=5.0      # Was 1.0
W_ANIME_SAKUGA=5.0    # Was 1.0
W_ANIME_STORY=2.0     # Was 0.5
W_TEMPORAL=0.5

# Curriculum
PERCENTILE_THRESHOLD=0.80   # Higher threshold = select better frames
CONTRASTIVE_MARGIN=0.20
HARD_NEG_MARGIN=0.10
USE_CURRICULUM=1

# PPO
VF_COEF=0.5
GAMMA=0.99
GAE_LAMBDA=0.95

# Ensemble
USE_REWARD_ENSEMBLE=1
N_ENSEMBLE=5   # More ensemble for stable rewards

echo "=============================================="
echo "V5 AGGRESSIVE Training"
echo "=============================================="
echo "Target: RecErr<0.30, Frechet<0.50, Anime↑"
echo "Model: ${ENC_HIDDEN} hidden, ${NUM_ATTN_LAYERS} layers"
echo "Weights: Rec=${W_REC}, FD=${W_FD}, Look=${W_ANIME_LOOK}"
echo "Epochs: ${EPOCHS}, LR: ${LR}"
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
    --gamma $GAMMA \
    --gae_lambda $GAE_LAMBDA \
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
    --validate_every 5 \
    --max_grad_norm $MAX_GRAD_NORM \
    --eval_with_baselines

echo "V5 Training complete!"
