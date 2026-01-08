#!/bin/bash

# LLMVS Ablation - Option A: Zero-shot with Visual Projection
# Uses pretrained LLMVS checkpoint with projection from visual features to LLaMA space
#
# Usage: bash scripts/ablation_script/run_ablation_llmvs_optionA.sh [GPU_ID] [CHECKPOINT_PATH]

GPU_ID=${1:-0}
CHECKPOINT=${2:-""}  # Path to pretrained LLMVS checkpoint (SumMe/TVSum)

DATASET_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_test_precompute"
SAVE_DIR="runs/ablation_llmvs/optionA_zeroshot"

# Ensure we run from project root
cd /home/serverai/ltdoanh/LayoutGeneration

# Add necessary paths
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/LLMVS

SCRIPT_PATH="ablation/LLMVS/eval_sakuga.py"

mkdir -p $SAVE_DIR

echo "========================================================"
echo "LLMVS ABLATION - OPTION A: Zero-shot with Projection"
echo "========================================================"
echo "GPU: $GPU_ID"
echo "Checkpoint: ${CHECKPOINT:-'Random init (no checkpoint provided)'}"
echo "Dataset: $DATASET_ROOT"
echo "Output: $SAVE_DIR"
echo "========================================================"

# Run evaluation with Option A
conda run -n sam python $SCRIPT_PATH \
    --option A \
    --dataset-root $DATASET_ROOT \
    --save-dir $SAVE_DIR \
    --checkpoint "$CHECKPOINT" \
    --gpu $GPU_ID \
    --input-dim 512 \
    --reduced-dim 2048 \
    --num-heads 2 \
    --num-layers 3 \
    --verbose \
    --save-results \
    --full-metrics

echo "========================================================"
echo "Done! Results saved to: $SAVE_DIR"
echo "========================================================"
