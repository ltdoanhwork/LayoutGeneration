#!/bin/bash
# V3 Premium Multi-Video RL Training for Anime DSN
# Enhanced with V3 features: 3-stage curriculum, reward normalization, temporal consistency

set -e

# Configuration
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_anime_v3"
LOG_DIR="runs/dsn_anime_v3/logs"
EPOCHS=20
DEVICE="cuda"
BATCH_SIZE=4

# Validation
VAL_VIDEOS_DIR="/home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test"
VAL_OUTPUT_DIR="runs/dsn_anime_v3/val_runs"

# V3 Premium Reward Weights (Curriculum will ramp these up)
W_ANIME_LOOK=2.5        # Increased from 2.0 in V1
W_ANIME_SAKUGA=2.5     # Increased from 2.0 in V1
W_ANIME_STORY=1.2      # Increased from 1.0 in V1
W_TEMPORAL=0.5         # NEW in V3
W_QUALITY_VAR=0.2      # NEW in V3

# Standard Weights (Balanced for multi-objective)
W_DIV=0.5
W_REP=0.5

# V3 Specific Parameters
PERCENTILE_THRESHOLD=0.75
CONTRASTIVE_MARGIN=0.15
HARD_NEG_MARGIN=0.05
USE_CURRICULUM=1
USE_QUALITY_CALIBRATION=1
USE_REWARD_NORM=1

echo "=========================================="
echo "Starting V3 PREMIUM Multi-Video RL Training"
echo "=========================================="
echo "Dataset: $DATASET_ROOT"
echo "Batch Size: $BATCH_SIZE"
echo "Save Dir: $SAVE_DIR"
echo "V3 Features: Curriculum | Reward Norm | Temporal | Hard Neg"
echo "=========================================="

python -m src.pipeline.train_rl_dsn_v3 \
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
  --w_temporal $W_TEMPORAL \
  --w_quality_var $W_QUALITY_VAR \
  \
  --percentile_threshold $PERCENTILE_THRESHOLD \
  --contrastive_margin $CONTRASTIVE_MARGIN \
  --hard_negative_margin $HARD_NEG_MARGIN \
  --use_curriculum $USE_CURRICULUM \
  --use_quality_calibration $USE_QUALITY_CALIBRATION \
  --use_reward_norm $USE_REWARD_NORM \
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
echo "V3 Premium Training Complete!"
echo "Results saved to $SAVE_DIR"
echo "View tensorboard: tensorboard --logdir $LOG_DIR"
