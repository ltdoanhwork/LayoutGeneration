#!/bin/bash
# Baseline training script for DSN keyframe selection
# This script uses stable hyperparameters for reliable training

# Exit on error
set -e

# Configuration
DATASET_ROOT="outputs/sakuga_dataset"
SAVE_DIR="outputs/dsn_runs/baseline_v1"
VAL_VIDEOS_DIR="data/samples/Sakuga"
VAL_OUTPUT_DIR="outputs/val_runs/baseline_v1"
LOG_DIR="runs/dsn_baseline_v1"

# Training hyperparameters (stable baseline)
EPOCHS=20
DEVICE="cuda"
LR=1e-4
ENTROPY_COEF=0.01
BASELINE_MOMENTUM=0.9
MAX_GRAD_NORM=1.0

# Model architecture
FEAT_DIM=512
ENC_HIDDEN=256
LSTM_HIDDEN=128
DROPOUT=0.3

# Budget constraints
BUDGET_RATIO=0.06
BUDGET_PENALTY=0.05
BMIN=3
BMAX=15

# Reward weights (balanced)
W_DIV=1.0
W_REP=1.0
W_REC=0.5
W_FD=0.2
W_MS=0.2
W_MOTION=0.2

# Feature settings
USE_MOTION=1
USE_LPIPS_DIV=0
MS_SWD_SCALES=3
MS_SWD_DIRS=16

# Evaluation settings
EVAL_EMBEDDER="clip_vitb32"
EVAL_BACKEND="pyscenedetect"
EVAL_THRESHOLD=27
EVAL_SAMPLE_STRIDE=5
EVAL_RESIZE_W=320
EVAL_RESIZE_H=180
VALIDATE_EVERY=2
EVAL_MAX_VIDEOS=10

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$VAL_OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Print configuration
echo "=================================="
echo "DSN BASELINE TRAINING"
echo "=================================="
echo "Dataset: $DATASET_ROOT"
echo "Save to: $SAVE_DIR"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "Learning rate: $LR"
echo "Validation every: $VALIDATE_EVERY epochs"
echo "=================================="
echo ""

# Activate conda environment

# Run training
python -m src.pipeline.train_rl_dsn \
  --dataset_root "$DATASET_ROOT" \
  --save_dir "$SAVE_DIR" \
  --epochs $EPOCHS \
  --device "$DEVICE" \
  --lr $LR \
  --entropy_coef $ENTROPY_COEF \
  --baseline_momentum $BASELINE_MOMENTUM \
  --max_grad_norm $MAX_GRAD_NORM \
  --feat_dim $FEAT_DIM \
  --enc_hidden $ENC_HIDDEN \
  --lstm_hidden $LSTM_HIDDEN \
  --dropout $DROPOUT \
  --budget_ratio $BUDGET_RATIO \
  --budget_penalty $BUDGET_PENALTY \
  --Bmin $BMIN \
  --Bmax $BMAX \
  --w_div $W_DIV \
  --w_rep $W_REP \
  --w_rec $W_REC \
  --w_fd $W_FD \
  --w_ms $W_MS \
  --w_motion $W_MOTION \
  --use_motion $USE_MOTION \
  --use_lpips_div $USE_LPIPS_DIV \
  --ms_swd_scales $MS_SWD_SCALES \
  --ms_swd_dirs $MS_SWD_DIRS \
  --val_videos_dir "$VAL_VIDEOS_DIR" \
  --val_output_dir "$VAL_OUTPUT_DIR" \
  --validate_every $VALIDATE_EVERY \
  --eval_embedder "$EVAL_EMBEDDER" \
  --eval_backend "$EVAL_BACKEND" \
  --eval_threshold $EVAL_THRESHOLD \
  --eval_sample_stride $EVAL_SAMPLE_STRIDE \
  --eval_resize_w $EVAL_RESIZE_W \
  --eval_resize_h $EVAL_RESIZE_H \
  --eval_with_baselines \
  --eval_max_videos $EVAL_MAX_VIDEOS \
  --log_dir "$LOG_DIR"

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
echo "Checkpoints saved to: $SAVE_DIR"
echo "Validation results: $VAL_OUTPUT_DIR"
echo "TensorBoard logs: $LOG_DIR"
echo ""
echo "View results:"
echo "  tensorboard --logdir $LOG_DIR"
echo ""
echo "Visualize validation:"
echo "  python -m eval.visualize_validation --val_output_dir $VAL_OUTPUT_DIR"
echo ""
