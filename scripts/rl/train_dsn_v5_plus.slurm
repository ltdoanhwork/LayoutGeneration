#!/bin/bash
#SBATCH --job-name=dsn_v5_plus
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu

# ============================================
# V5+ MULTI-TASK RL WITH ENHANCED CLIP-IQA
# ============================================
# Innovations in this version:
#
# 1. MULTI-TASK RL
#    - Task 1: RecErr + Frechet optimization
#    - Task 2: Anime quality optimization
#    - Separate policy heads → no competing rewards!
#
# 2. ENHANCED CLIP-IQA (addresses paper limitations)
#    - Learnable Prompt Embeddings (vs fixed text)
#    - Anime-Specific Vocabulary (sakuga, genga, etc.)
#    - Task-Specific Head (trained for anime quality)
#
# 3. PREMIUM ANIME REWARDS
#    - Percentile-based thresholding
#    - Curriculum learning across epochs
#
# Target Metrics:
#    RecErr < 0.30  |  Frechet < 0.50  |  Anime ↑↑
# ============================================

cd /home/serverai/ltdoanh/LayoutGeneration

# Activate environment
source /srv/conda/envs/serverai/sam/bin/activate

mkdir -p logs

# Paths
DATASET_ROOT="data/sakuga_dataset_100_samples_transnetv2"
SAVE_DIR="runs/dsn_v5_plus_transnetv2/checkpoints"
LOG_DIR="runs/dsn_v5_plus_transnetv2/logs"
VAL_VIDEOS="data/samples/Sakuga_test"
VAL_OUTPUT="runs/dsn_v5_plus_transnetv2/val_runs"

# Training
EPOCHS=80
DEVICE="cuda"

# --- Model (Balanced for performance & speed) ---
ENC_HIDDEN=256
LSTM_HIDDEN=128
NUM_ATTN_LAYERS=3
NUM_ATTN_HEADS=4
DROPOUT=0.15
VALUE_HIDDEN=128

# --- Optimization ---
LR=2e-4
N_PPO_EPOCHS=8
BATCH_SIZE=12
CLIP_RANGE=0.15
TARGET_KL=0.015
ENTROPY_COEF=0.025
MAX_GRAD_NORM=0.5

# --- PPO ---
VF_COEF=0.5
GAMMA=0.99
GAE_LAMBDA=0.95

# --- Premium Rewards ---
PERCENTILE_THRESHOLD=0.85
USE_CURRICULUM=1

echo "=============================================="
echo "V5+ MULTI-TASK RL + ENHANCED CLIP-IQA"
echo "=============================================="
echo ""
echo "🔧 Innovations:"
echo "   1. Multi-Task RL (RecErr vs Anime heads)"
echo "   2. Enhanced CLIP-IQA (learnable prompts)"
echo "   3. Premium Anime Rewards (curriculum)"
echo ""
echo "📐 Model: ${ENC_HIDDEN} hidden, ${NUM_ATTN_LAYERS} layers"
echo "📊 LR: ${LR}, Epochs: ${EPOCHS}"
echo "🎯 Target: RecErr<0.30, Frechet<0.50, Anime↑"
echo "=============================================="
echo ""

python -m src.pipeline.train_rl_dsn_v5 \
    --dataset_root "$DATASET_ROOT" \
    --save_dir "$SAVE_DIR" \
    --log_dir "$LOG_DIR" \
    --epochs $EPOCHS \
    --device "$DEVICE" \
    --multi_video 1 \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --feat_dim 512 \
    --enc_hidden $ENC_HIDDEN \
    --lstm_hidden $LSTM_HIDDEN \
    --num_attn_layers $NUM_ATTN_LAYERS \
    --num_attn_heads $NUM_ATTN_HEADS \
    --dropout $DROPOUT \
    --value_hidden_dim $VALUE_HIDDEN \
    --use_cache 1 \
    --use_lstm_in_advanced 1 \
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
    --percentile_threshold $PERCENTILE_THRESHOLD \
    --use_curriculum $USE_CURRICULUM \
    --val_videos_dir "$VAL_VIDEOS" \
    --val_output_dir "$VAL_OUTPUT" \
    --validate_every 5 \
    --max_grad_norm $MAX_GRAD_NORM \
    --eval_embedder "clip_vitb32" \
    --eval_backend "transnetv2" \
    --eval_with_baselines

echo ""
echo "=============================================="
echo "✅ V5+ Training Complete!"
echo ""
echo "📁 Checkpoints: $SAVE_DIR"
echo "📊 TensorBoard: tensorboard --logdir $LOG_DIR"
echo "✨ Best checkpoint saved as: dsn_checkpoint_best.pt"
echo "=============================================="
