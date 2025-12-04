#!/bin/bash
# Premium Multi-Video RL Training for Anime DSN
# Optimized for maximizing Anime-CLIP-IQA scores (Look, Sakuga, Story)

set -e

# Configuration
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_anime_premium_v1"
LOG_DIR="runs/dsn_anime_premium_v1/logs"
EPOCHS=20
DEVICE="cuda"
BATCH_SIZE=4

# Validation
VAL_VIDEOS_DIR="/home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test"
VAL_OUTPUT_DIR="runs/dsn_anime_premium_v1/val_runs"

# Premium Reward Weights (Curriculum will ramp these up)
W_ANIME_LOOK=2.0
W_ANIME_SAKUGA=2.0
W_ANIME_STORY=1.0

# Standard Weights (Lowered to prioritize aesthetics)
W_DIV=0.5
W_REP=0.5

echo "=========================================="
echo "Starting PREMIUM Multi-Video RL Training"
echo "=========================================="
echo "Dataset: $DATASET_ROOT"
echo "Batch Size: $BATCH_SIZE"
echo "Save Dir: $SAVE_DIR"
echo "=========================================="

python -m src.pipeline.train_rl_dsn_multi_anime_premium \
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
  --w_anime_look $W_ANIME_LOOK \
  --w_anime_sakuga $W_ANIME_SAKUGA \
  --w_anime_story $W_ANIME_STORY \
  --percentile_threshold 0.75 \
  --use_curriculum 1 \
  \
  --w_div $W_DIV \
  --w_rep $W_REP \
  --w_probsep 0.1 \
  \
  --lr 1e-4 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
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
echo "Premium Training Complete!"
echo "Results saved to $SAVE_DIR"
