#!/bin/bash
# Quick test to verify advanced DSN training works
# Runs 1 epoch on a small dataset

echo "Testing Advanced DSN training setup..."

conda run -n sam python -m src.pipeline.train_rl_dsn \
  --model_type advanced \
  --dataset_root outputs/sakuga_dataset \
  --save_dir /tmp/test_advanced_dsn \
  --log_dir /tmp/test_advanced_dsn \
  --epochs 1 \
  --device cuda:0 \
  --feat_dim 512 \
  --enc_hidden 256 \
  --lstm_hidden 128 \
  --num_attn_heads 4 \
  --num_attn_layers 2 \
  --num_scales 3 \
  --use_cache 1 \
  --cache_size 100 \
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
  --lr 1e-4

if [ $? -eq 0 ]; then
    echo "✅ Test passed! Advanced DSN training works."
    echo "Checkpoint saved to: /tmp/test_advanced_dsn/dsn_checkpoint_ep1.pt"
else
    echo "❌ Test failed! Check error messages above."
    exit 1
fi
