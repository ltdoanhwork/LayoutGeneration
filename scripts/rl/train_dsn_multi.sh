#!/bin/bash
# Script for Multi-Video RL Training of DSN
# Uses Gradient Accumulation to train on batches of videos

set -e

# Configuration
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_rl_multi_video"
LOG_DIR="runs/dsn_rl_multi_video/logs"
EPOCHS=20
DEVICE="cuda"
BATCH_SIZE=4  # Number of videos to accumulate gradients over

# Validation
VAL_VIDEOS_DIR="/home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test"
VAL_OUTPUT_DIR="runs/dsn_rl_multi_video/val_runs"

# Anime Reward Weights (V2 Unified)
W_ANIME_LOOK=0.3
W_ANIME_SAKUGA=0.4
W_ANIME_STORY=0.2

echo "=========================================="
echo "Starting Multi-Video RL Training"
echo "=========================================="
echo "Dataset: $DATASET_ROOT"
echo "Batch Size: $BATCH_SIZE (Gradient Accumulation)"
echo "Save Dir: $SAVE_DIR"
echo "=========================================="

python -m src.pipeline.train_rl_dsn_multi \
  --dataset_root $DATASET_ROOT \
  --save_dir $SAVE_DIR \
  --log_dir $LOG_DIR \
  --epochs $EPOCHS \
  --device $DEVICE \
  \
  --multi_video 1 \
  --batch_size $BATCH_SIZE \
  --sampling_strategy random_uniform \
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
  --use_anime_reward 1 \
  --w_anime_look $W_ANIME_LOOK \
  --w_anime_sakuga $W_ANIME_SAKUGA \
  --w_anime_story $W_ANIME_STORY \
  \
  --lr 1e-4 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --entropy_coef 0.01 \
  --baseline_momentum 0.9 \
  \
  --budget_ratio 0.06 \
  --budget_penalty 0.05 \
  --Bmin 3 \
  --Bmax 15 \
  \
  --w_div 1.0 \
  --w_rep 1.0 \
  --w_rec 0.5 \
  --w_fd 0.2 \
  --w_ms 0.2 \
  --w_motion 0.2 \
  --ms_swd_scales 3 \
  --ms_swd_dirs 16 \
  --use_motion 1 \
  --use_lpips_div 0 \
  \
  --val_videos_dir $VAL_VIDEOS_DIR \
  --val_output_dir $VAL_OUTPUT_DIR \
  --validate_every 1 \
  --eval_embedder clip_vitb32 \
  --eval_backend transnetv2 \
  --eval_sample_stride 5 \
  --eval_resize_w 320 \
  --eval_resize_h 180 \
  --eval_with_baselines

echo ""
echo "Training Complete!"
echo "Results saved to $SAVE_DIR"
