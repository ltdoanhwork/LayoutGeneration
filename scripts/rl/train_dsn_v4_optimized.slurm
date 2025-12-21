#!/bin/bash
#SBATCH --job-name=dsn_v4_opt
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu

# ============================================
# V4 OPTIMIZED - Focus on RecErr/Frechet
# ============================================
# Changes from V4-Scaled:
# - Reduced model size: 1024→512 hidden, 6→4 layers
# - Increased LR: 5e-5 → 2e-4
# - Heavier RecErr/Frechet weights
# - Reduced anime weights to focus on core metrics
# ============================================

cd /home/serverai/ltdoanh/LayoutGeneration

# Activate environment
source /srv/conda/envs/serverai/sam/bin/activate

# Create log dir
mkdir -p logs

# Paths
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_v4_optimized/checkpoints"
LOG_DIR="runs/dsn_v4_optimized/logs"
VAL_VIDEOS="data/samples/Sakuga_test"
VAL_OUTPUT="runs/dsn_v4_optimized/val_runs"

# Training
EPOCHS=50
DEVICE="cuda"

# --- Model (REDUCED from scaled) ---
ENC_HIDDEN=512
LSTM_HIDDEN=256
NUM_ATTN_LAYERS=4
NUM_ATTN_HEADS=4
DROPOUT=0.2
VALUE_HIDDEN=256

# --- Optimization (AGGRESSIVE) ---
LR=2e-4
N_PPO_EPOCHS=6
BATCH_SIZE=8
CLIP_RANGE=0.2
TARGET_KL=0.02
ENTROPY_COEF=0.015
MAX_GRAD_NORM=1.0

# --- Reward Weights (FOCUS RecErr/Frechet) ---
W_REC=5.0      # Heavily increased
W_FD=3.0       # Heavily increased
W_REP=1.0      # Representativeness
W_DIV=0.5      # Diversity
W_PROBSEP=0.1

# --- Anime Rewards (REDUCED) ---
W_ANIME_LOOK=1.0      # Reduced from 3.0
W_ANIME_SAKUGA=1.0    # Reduced from 3.0
W_ANIME_STORY=0.5     # Reduced from 1.5
W_TEMPORAL=0.3

# Curriculum
PERCENTILE_THRESHOLD=0.75
CONTRASTIVE_MARGIN=0.15
HARD_NEG_MARGIN=0.05
USE_CURRICULUM=1

# PPO
VF_COEF=0.5
GAMMA=0.99
GAE_LAMBDA=0.95

# Ensemble
USE_REWARD_ENSEMBLE=1
N_ENSEMBLE=3

echo "=============================================="
echo "V4 OPTIMIZED Training"
echo "=============================================="
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

echo "Training complete!"
