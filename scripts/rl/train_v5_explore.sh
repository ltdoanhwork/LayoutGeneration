#!/bin/bash
# V5 HIGH EXPLORATION - More entropy, smaller model, faster learning
# Goal: Break out of local minima with more exploration

cd /home/serverai/ltdoanh/LayoutGeneration
source /srv/conda/envs/serverai/sam/bin/activate
mkdir -p logs

DATASET_ROOT="data/sakuga_dataset_100_samples_transnetv2"
SAVE_DIR="runs/dsn_v5_explore_transnetv2/checkpoints"
LOG_DIR="runs/dsn_v5_explore_transnetv2/logs"
VAL_VIDEOS="data/samples/Sakuga_test"
VAL_OUTPUT="runs/dsn_v5_explore_transnetv2/val_runs"

python -m src.pipeline.train_rl_dsn_v5 \
    --dataset_root "$DATASET_ROOT" \
    --save_dir "$SAVE_DIR" \
    --log_dir "$LOG_DIR" \
    --epochs 100 \
    --device cuda \
    --batch_size 8 \
    --lr 5e-4 \
    --feat_dim 512 \
    --enc_hidden 128 \
    --lstm_hidden 64 \
    --num_attn_layers 1 \
    --num_attn_heads 2 \
    --dropout 0.1 \
    --value_hidden_dim 64 \
    --use_cache 1 \
    --use_lstm_in_advanced 1 \
    --use_raft_motion 1 \
    --motion_dim 128 \
    --use_anime_attrs 1 \
    --anime_attrs_dim 6 \
    --clip_range 0.3 \
    --target_kl 0.05 \
    --n_ppo_epochs 4 \
    --entropy_coef 0.08 \
    --vf_coef 0.25 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --percentile_threshold 0.75 \
    --use_curriculum 1 \
    --val_videos_dir "$VAL_VIDEOS" \
    --val_output_dir "$VAL_OUTPUT" \
    --validate_every 5 \
    --max_grad_norm 1.0 \
    --eval_embedder "clip_vitb32" \
    --eval_backend "transnetv2" \
    --eval_with_baselines

echo "✅ V5 High Exploration training complete!"
