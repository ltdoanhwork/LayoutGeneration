#!/bin/bash
# V4 PPO Training for Anime DSN with Actor-Critic
# State-of-the-art RL with PPO, GAE, and Reward Ensemble

set -e

# Configuration
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_anime_v4"
LOG_DIR="runs/dsn_anime_v4/logs"
EPOCHS=20
DEVICE="cuda"
BATCH_SIZE=4

# Validation
VAL_VIDEOS_DIR="/home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test"
VAL_OUTPUT_DIR="runs/dsn_anime_v4/val_runs"

# V4 PPO Configuration (Key hyperparameters)
CLIP_RANGE=0.2          # PPO clipping range (0.1-0.3 typical)
TARGET_KL=0.01          # Target KL for early stopping
N_PPO_EPOCHS=4          # Number of PPO update epochs
VF_COEF=0.5             # Value function loss coefficient
VALUE_HIDDEN_DIM=256    # Value head hidden dimension

# V4 GAE Configuration
GAMMA=0.99              # Discount factor
GAE_LAMBDA=0.95         # GAE lambda (bias-variance tradeoff)

# V4 Reward Ensemble
USE_REWARD_ENSEMBLE=1
N_ENSEMBLE=3

# Premium Reward Weights
W_ANIME_LOOK=2.5
W_ANIME_SAKUGA=2.5
W_ANIME_STORY=1.2
W_TEMPORAL=0.5

# Standard Weights
W_DIV=0.5
W_REP=0.0       # Replaced by W_REC
W_REC=1.0       # Explicit Reconstruction reward (for RecErr)
W_FD=0.5        # Frechet distance reward (for Frechet)

# V3 Compatible Parameters
PERCENTILE_THRESHOLD=0.75
CONTRASTIVE_MARGIN=0.15
HARD_NEG_MARGIN=0.05
USE_CURRICULUM=1

echo "=============================================="
echo "V4 PPO Training for Anime DSN (Actor-Critic)"
echo "=============================================="
echo "Dataset: $DATASET_ROOT"
echo "Save Dir: $SAVE_DIR"
echo ""
echo "V4 Features:"
echo "  • PPO with clipped objective (ε=$CLIP_RANGE)"
echo "  • GAE advantage estimation (λ=$GAE_LAMBDA)"
echo "  • Actor-Critic with learned value baseline"
echo "  • Reward ensemble (n=$N_ENSEMBLE)"
echo "=============================================="

python -m src.pipeline.train_rl_dsn_v4 \
  --dataset_root $DATASET_ROOT \
  --save_dir $SAVE_DIR \
  --log_dir $LOG_DIR \
  --epochs $EPOCHS \
  --device $DEVICE \
  \
  --multi_video 1 \
  --batch_size $BATCH_SIZE \
  \
  --model_type advanced \
  --feat_dim 512 \
  --enc_hidden 256 \
  --lstm_hidden 128 \
  --dropout 0.3 \
  \
  --num_attn_heads 4 \
  --num_attn_layers 2 \
  --num_scales 3 \
  --use_cache 1 \
  --cache_size 1000 \
  --pos_encoding_type sinusoidal \
  --use_lstm_in_advanced 1 \
  \
  --use_raft_motion 1 \
  --motion_dim 128 \
  --motion_fusion_type cross_attention \
  \
  --use_anime_attrs 1 \
  --anime_attrs_dim 6 \
  \
  --clip_range $CLIP_RANGE \
  --target_kl $TARGET_KL \
  --n_ppo_epochs $N_PPO_EPOCHS \
  --vf_coef $VF_COEF \
  --value_hidden_dim $VALUE_HIDDEN_DIM \
  \
  --gamma $GAMMA \
  --gae_lambda $GAE_LAMBDA \
  \
  --use_reward_ensemble $USE_REWARD_ENSEMBLE \
  --n_ensemble $N_ENSEMBLE \
  \
  --w_anime_look $W_ANIME_LOOK \
  --w_anime_sakuga $W_ANIME_SAKUGA \
  --w_anime_story $W_ANIME_STORY \
  --w_temporal $W_TEMPORAL \
  \
  --percentile_threshold $PERCENTILE_THRESHOLD \
  --contrastive_margin $CONTRASTIVE_MARGIN \
  --hard_negative_margin $HARD_NEG_MARGIN \
  --use_curriculum $USE_CURRICULUM \
  \
  --w_div $W_DIV \
  --w_rep $W_REP \
  --w_rec $W_REC \
  --w_fd $W_FD \
  --w_probsep 0.1 \
  \
  --lr 1e-4 \
  --weight_decay 0.0 \
  --max_grad_norm 0.5 \
  --entropy_coef 0.01 \
  \
  --budget_ratio 0.06 \
  --budget_penalty 0.05 \
  --Bmin 3 \
  --Bmax 15 \
  \
  --val_videos_dir $VAL_VIDEOS_DIR \
  --val_output_dir $VAL_OUTPUT_DIR \
  --validate_every 2 \
  --eval_embedder clip_vitb32 \
  --eval_backend transnetv2 \
  --eval_sample_stride 5 \
  --eval_resize_w 320 \
  --eval_resize_h 180

echo ""
echo "=============================================="
echo "V4 PPO Training Complete!"
echo "Results saved to $SAVE_DIR"
echo "View tensorboard: tensorboard --logdir $LOG_DIR"
echo "=============================================="
