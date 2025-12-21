#!/bin/bash
# V8 Constrained Multi-Objective RL Training
# 
# Key features:
# - Anime quality as PRIMARY objective
# - RecErr/Coverage/Diversity as CONSTRAINTS (Lagrangian)
# - State-dependent gating (alpha_t per frame)
# - PCGrad gradient surgery
# - DPP two-stage selection
# - Test-time scaling

# turbo-all

set -e

# Paths
DATASET_ROOT="data/sakuga_dataset_100_samples"
SAVE_DIR="runs/dsn_v8_constrained"
VAL_VIDEOS="data/samples/Sakuga"
VAL_OUTPUT="outputs/val_v8"

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$VAL_OUTPUT"

echo "========================================"
echo "V8 Constrained MORL Training"
echo "========================================"
echo "Dataset: $DATASET_ROOT"
echo "Save: $SAVE_DIR"
echo "Validation: $VAL_VIDEOS"
echo "========================================"

python -m src.pipeline.train_rl_dsn_v8 \
    --dataset_root "$DATASET_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs 60 \
    --seed 42 \
    \
    --feat_dim 512 \
    --enc_hidden 256 \
    --lstm_hidden 128 \
    --use_anime_attrs 1 \
    --anime_attrs_dim 6 \
    --use_raft_motion 1 \
    --motion_dim 128 \
    --gating_hidden 64 \
    \
    --use_pcgrad 1 \
    --use_dpp 1 \
    --use_tts 1 \
    \
    --rec_err_threshold 0.35 \
    --coverage_threshold 0.3 \
    --diversity_threshold 0.25 \
    --lambda_lr 0.01 \
    \
    --dpp_beta 1.0 \
    --dpp_candidate_ratio 0.3 \
    \
    --tts_n_samples 8 \
    --tts_temperature 1.2 \
    \
    --anime_scale 3.0 \
    --quantile_scale 2.0 \
    --use_curriculum 1 \
    \
    --lr 2e-4 \
    --clip_range 0.2 \
    --n_ppo_epochs 4 \
    --entropy_coef 0.01 \
    --vf_coef 0.5 \
    --max_grad_norm 0.5 \
    \
    --budget_ratio 0.06 \
    --Bmin 3 \
    --Bmax 15 \
    \
    --device cuda \
    \
    --val_videos_dir "$VAL_VIDEOS" \
    --val_output_dir "$VAL_OUTPUT" \
    --validate_every 5 \
    --eval_backend transnetv2 \
    --eval_embedder clip_vitb32 \
    --eval_device cuda \
    --min_scene_len 48 \
    \
    --save_visualizations 1

echo "========================================"
echo "Training complete!"
echo "Checkpoints: $SAVE_DIR"
echo "TensorBoard: tensorboard --logdir $SAVE_DIR/logs"
echo "========================================"
