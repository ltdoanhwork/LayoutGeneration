#!/bin/bash
# Training script for Advanced DSN model
# This script demonstrates how to train the advanced model with all features enabled

# Activate conda environment
conda activate sam

# Training command for Advanced DSN
python -m src.pipeline.train_rl_dsn \
  --model_type advanced \
  --dataset_root /home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_100_samples \
  --save_dir runs/dsn_advanced_v1_no_motion_100_samples \
  --log_dir runs/dsn_advanced_v1_no_motion_100_samples \
  --epochs 20 \
  --device cuda:0 \
  \
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
  --budget_ratio 0.06 \
  --Bmin 3 \
  --Bmax 15 \
  --budget_penalty 0.05 \
  \
  --w_div 1.0 \
  --w_rep 1.0 \
  --w_rec 0.5 \
  --w_fd 0.2 \
  --w_ms 0.2 \
  --w_motion 0.2 \
  --use_motion 0 \
  --ms_swd_scales 3 \
  --ms_swd_dirs 16 \
  \
  --lr 1e-4 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --entropy_coef 0.01 \
  --baseline_momentum 0.9 \
  \
  --val_videos_dir /home/serverai/ltdoanh/LayoutGeneration/data/samples/vssum \
  --val_output_dir runs/dsn_advanced_v1_no_motion_100_samples \
  --validate_every 1 \
  --eval_embedder clip_vitb32 \
  --eval_backend transnetv2 \
  --eval_sample_stride 5 \
  --eval_resize_w 320 \
  --eval_resize_h 180 \
  --eval_with_baselines

echo "Training completed! Check TensorBoard for results:"
echo "tensorboard --logdir runs/dsn_advanced_v1_no_motion_100_samples/"
