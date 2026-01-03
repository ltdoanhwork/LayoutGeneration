#!/bin/bash
# ============================================================================
# Anime-CLIP-IQA Ablation Study
# Runs Tracks A, B, and C for DSN v11
# ============================================================================

DATA_ROOT="data/sakuga_dataset_100_samples"
SAVE_ROOT="runs/ablation_anime_iqa"
EPOCHS=5 # Small number for testing, increase for real training

# Common args
ARGS="--dataset_root $DATA_ROOT \
      --epochs $EPOCHS \
      --lr 1e-4 \
      --budget_ratio 0.15 \
      --Bmin 3 \
      --Bmax 15 \
      --entropy_coef 0.02 \
      --clip_range 0.2 \
      --diversity_weight 0.3 \
      --device cuda"

echo "============================================================================"
echo "Starting Ablation Study"
echo "Data: $DATA_ROOT"
echo "Save Root: $SAVE_ROOT"
echo "============================================================================"

# Track A: Feature-only (Inputs=Feat+Anime, Reward=Standard)
echo ""
echo ">>> Running Track A: Feature-only (The Eyes)"
python3 -m src.pipeline.train_rl_dsn_v11_simple \
    --save_dir "$SAVE_ROOT" \
    --track A \
    $ARGS

# Track B: Reward-only (Inputs=Feat, Reward=Anime)
echo ""
echo ">>> Running Track B: Reward-only (The Incentive)"
python3 -m src.pipeline.train_rl_dsn_v11_simple \
    --save_dir "$SAVE_ROOT" \
    --track B \
    $ARGS

# Track C: Combined (Inputs=Feat+Anime, Reward=Anime)
echo ""
echo ">>> Running Track C: Combined (Full Integration)"
python3 -m src.pipeline.train_rl_dsn_v11_simple \
    --save_dir "$SAVE_ROOT" \
    --track C \
    $ARGS

echo ""
echo "============================================================================"
echo "Ablation Study Complete!"
echo "Results saved to $SAVE_ROOT"
echo "============================================================================"
