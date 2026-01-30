#!/bin/bash

# LLMVS Ablation - Option B: Train from Scratch with Visual Features
# Trains LLMVSVisual model on Sakuga dataset using REINFORCE
#
# Usage: bash scripts/ablation_script/run_ablation_llmvs_optionB.sh [GPU_ID] [MODE]
# MODE: train | eval | train_eval (default: train_eval)

GPU_ID=${1:-0}
MODE=${2:-"train_eval"}

TRAIN_DATASET="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_dataset_v11"
TEST_DATASET="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_test_precompute"
SAVE_DIR="runs/ablation_llmvs/optionB_visual"

# Ensure we run from project root
cd /home/serverai/ltdoanh/LayoutGeneration

# Add necessary paths
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/LLMVS

TRAIN_SCRIPT="ablation/LLMVS/train_sakuga.py"
EVAL_SCRIPT="ablation/LLMVS/eval_sakuga.py"

mkdir -p $SAVE_DIR

echo "========================================================"
echo "LLMVS ABLATION - OPTION B: LLMVSVisual"
echo "========================================================"
echo "GPU: $GPU_ID"
echo "Mode: $MODE"
echo "Train Dataset: $TRAIN_DATASET"
echo "Test Dataset: $TEST_DATASET"
echo "Output: $SAVE_DIR"
echo "========================================================"

# Training phase
if [[ "$MODE" == "train" || "$MODE" == "train_eval" ]]; then
    echo ""
    echo "========== TRAINING PHASE =========="
    
    PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/LLMVS python $TRAIN_SCRIPT \
        --dataset-root $TRAIN_DATASET \
        --save-dir $SAVE_DIR \
        --gpu $GPU_ID \
        --max-epoch 60 \
        --input-dim 512 \
        --reduced-dim 2048 \
        --num-heads 2 \
        --num-layers 3 \
        --lr 1e-4 \
        --num-episode 5 \
        --beta 0.01 \
        --verbose \
        --save-results
fi

# Evaluation phase
if [[ "$MODE" == "eval" || "$MODE" == "train_eval" ]]; then
    echo ""
    echo "========== EVALUATION PHASE =========="
    
    # Use best model if exists, otherwise use final model
    if [ -f "$SAVE_DIR/best_model.pth" ]; then
        CHECKPOINT="$SAVE_DIR/best_model.pth"
    elif [ -f "$SAVE_DIR/model_epoch60.pth" ]; then
        CHECKPOINT="$SAVE_DIR/model_epoch60.pth"
    else
        CHECKPOINT=""
    fi
    
    echo "Checkpoint: ${CHECKPOINT:-'No checkpoint found'}"
    
    PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/LLMVS python $EVAL_SCRIPT \
        --option B \
        --dataset-root $TEST_DATASET \
        --save-dir $SAVE_DIR/eval \
        --checkpoint "$CHECKPOINT" \
        --gpu $GPU_ID \
        --input-dim 512 \
        --reduced-dim 2048 \
        --num-heads 2 \
        --num-layers 3 \
        --verbose \
        --save-results \
        --full-metrics
fi

echo "========================================================"
echo "Done! Results saved to: $SAVE_DIR"
echo "========================================================"
