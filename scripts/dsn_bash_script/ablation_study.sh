#!/bin/bash
# Ablation study scripts for Advanced DSN model

# 1. Baseline (for comparison)
echo "=== Training Baseline DSN ==="
python -m src.pipeline.train_rl_dsn \
  --model_type baseline \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/ablation/baseline \
  --log_dir runs/ablation/baseline \
  --epochs 10 \
  --device cuda:0 \
  --feat_dim 512 --enc_hidden 256 --lstm_hidden 128 \
  --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
  --w_div 1.0 --w_rep 1.0 --w_rec 0.5 --w_fd 0.2 --w_ms 0.2 --w_motion 0.2 \
  --use_motion 1

# 2. Attention Only (no LSTM, no multi-scale)
echo "=== Training Attention Only ==="
python -m src.pipeline.train_rl_dsn \
  --model_type advanced \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/ablation/attention_only \
  --log_dir runs/ablation/attention_only \
  --epochs 10 \
  --device cuda:0 \
  --feat_dim 512 --enc_hidden 256 --lstm_hidden 128 \
  --num_attn_heads 4 --num_attn_layers 2 \
  --num_scales 1 \
  --use_lstm_in_advanced 0 \
  --use_cache 0 \
  --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
  --w_div 1.0 --w_rep 1.0 --w_rec 0.5 --w_fd 0.2 --w_ms 0.2 --w_motion 0.2 \
  --use_motion 1

# 3. Multi-Scale Only (no attention)
echo "=== Training Multi-Scale Only ==="
python -m src.pipeline.train_rl_dsn \
  --model_type advanced \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/ablation/multiscale_only \
  --log_dir runs/ablation/multiscale_only \
  --epochs 10 \
  --device cuda:0 \
  --feat_dim 512 --enc_hidden 256 --lstm_hidden 128 \
  --num_attn_heads 4 --num_attn_layers 0 \
  --num_scales 3 \
  --use_lstm_in_advanced 1 \
  --use_cache 0 \
  --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
  --w_div 1.0 --w_rep 1.0 --w_rec 0.5 --w_fd 0.2 --w_ms 0.2 --w_motion 0.2 \
  --use_motion 1

# 4. Attention + Multi-Scale (no cache)
echo "=== Training Attention + Multi-Scale ==="
python -m src.pipeline.train_rl_dsn \
  --model_type advanced \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/ablation/attn_multiscale \
  --log_dir runs/ablation/attn_multiscale \
  --epochs 10 \
  --device cuda:0 \
  --feat_dim 512 --enc_hidden 256 --lstm_hidden 128 \
  --num_attn_heads 4 --num_attn_layers 2 \
  --num_scales 3 \
  --use_lstm_in_advanced 1 \
  --use_cache 0 \
  --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
  --w_div 1.0 --w_rep 1.0 --w_rec 0.5 --w_fd 0.2 --w_ms 0.2 --w_motion 0.2 \
  --use_motion 1

# 5. Full Model (all features)
echo "=== Training Full Advanced Model ==="
python -m src.pipeline.train_rl_dsn \
  --model_type advanced \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/ablation/full_advanced \
  --log_dir runs/ablation/full_advanced \
  --epochs 10 \
  --device cuda:0 \
  --feat_dim 512 --enc_hidden 256 --lstm_hidden 128 \
  --num_attn_heads 4 --num_attn_layers 2 \
  --num_scales 3 \
  --use_lstm_in_advanced 1 \
  --use_cache 1 --cache_size 1000 \
  --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
  --w_div 1.0 --w_rep 1.0 --w_rec 0.5 --w_fd 0.2 --w_ms 0.2 --w_motion 0.2 \
  --use_motion 1

echo ""
echo "=== Ablation Study Complete ==="
echo "Compare results in TensorBoard:"
echo "tensorboard --logdir runs/ablation/"
