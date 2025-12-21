#!/bin/bash
# V5 LARGE MODEL - More capacity for complex patterns
# Goal: See if bigger model helps

cd /home/serverai/ltdoanh/LayoutGeneration
source /srv/conda/envs/serverai/sam/bin/activate
mkdir -p logs

DATASET_ROOT="data/sakuga_dataset_100_samples_transnetv2"
SAVE_DIR="runs/dsn_v5_large_transnetv2/checkpoints"
LOG_DIR="runs/dsn_v5_large_transnetv2/logs"
VAL_VIDEOS="data/samples/Sakuga_test"
VAL_OUTPUT="runs/dsn_v5_large_transnetv2/val_runs"

python -m src.pipeline.train_rl_dsn_v5 \
    --dataset_root "$DATASET_ROOT" \
    --save_dir "$SAVE_DIR" \
    --log_dir "$LOG_DIR" \
    --epochs 80 \
    --device cuda \
    --batch_size 6 \
    --lr 1e-4 \
    --feat_dim 512 \
    --enc_hidden 512 \
    --lstm_hidden 256 \
    --num_attn_layers 4 \
    --num_attn_heads 8 \
    --dropout 0.2 \
    --value_hidden_dim 256 \
    --use_cache 1 \
    --use_lstm_in_advanced 1 \
    --use_raft_motion 1 \
    --motion_dim 128 \
    --use_anime_attrs 1 \
    --anime_attrs_dim 6 \
    --clip_range 0.1 \
    --target_kl 0.01 \
    --n_ppo_epochs 8 \
    --entropy_coef 0.015 \
    --vf_coef 0.5 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --percentile_threshold 0.80 \
    --use_curriculum 1 \
    --val_videos_dir "$VAL_VIDEOS" \
    --val_output_dir "$VAL_OUTPUT" \
    --validate_every 5 \
    --max_grad_norm 0.5 \
    --eval_embedder "clip_vitb32" \
    --eval_backend "transnetv2" \
    --eval_with_baselines

echo "✅ V5 Large Model training complete!"
