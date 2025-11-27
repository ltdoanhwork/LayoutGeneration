#!/bin/bash
# Fast training script for quick experiments (fewer epochs, less validation)
set -e

DATASET_ROOT="outputs/sakuga_dataset"
SAVE_DIR="outputs/dsn_runs/quick_test"
VAL_VIDEOS_DIR="data/samples/Sakuga"
VAL_OUTPUT_DIR="outputs/val_runs/quick_test"
LOG_DIR="runs/dsn_quick_test"

# Quick settings
EPOCHS=5
VALIDATE_EVERY=5
EVAL_MAX_VIDEOS=5

mkdir -p "$SAVE_DIR" "$VAL_OUTPUT_DIR" "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam

python -m src.pipeline.train_rl_dsn \
  --dataset_root "$DATASET_ROOT" \
  --save_dir "$SAVE_DIR" \
  --epochs $EPOCHS \
  --device cuda \
  --lr 1e-4 \
  --feat_dim 512 \
  --enc_hidden 256 \
  --lstm_hidden 128 \
  --budget_ratio 0.06 \
  --Bmin 3 \
  --Bmax 15 \
  --w_div 1.0 \
  --w_rep 1.0 \
  --w_rec 0.5 \
  --w_fd 0.2 \
  --w_ms 0.2 \
  --w_motion 0.2 \
  --use_motion 1 \
  --val_videos_dir "$VAL_VIDEOS_DIR" \
  --val_output_dir "$VAL_OUTPUT_DIR" \
  --validate_every $VALIDATE_EVERY \
  --eval_embedder clip_vitb32 \
  --eval_backend pyscenedetect \
  --eval_sample_stride 5 \
  --eval_resize_w 320 \
  --eval_resize_h 180 \
  --eval_max_videos $EVAL_MAX_VIDEOS \
  --log_dir "$LOG_DIR"

echo "Quick test completed! View with: tensorboard --logdir $LOG_DIR"
