#!/bin/bash

# LLMVS Ablation - Evaluation Only
# Evaluates a specific LLMVS checkpoint on Sakuga test data
#
# Usage: bash scripts/ablation_script/run_ablation_llmvs_eval.sh [GPU_ID] [OPTION] [CHECKPOINT]
# OPTION: A or B

GPU_ID=${1:-0}
OPTION=${2:-"B"}
CHECKPOINT=${3:-""}

DATASET_ROOT="/home/serverai/ltdoanh/LayoutGeneration/data/sakuga_test_precompute"
SAVE_DIR="runs/ablation_llmvs/eval_option${OPTION}"

# Ensure we run from project root
cd /home/serverai/ltdoanh/LayoutGeneration

# Add necessary paths
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/LLMVS

SCRIPT_PATH="ablation/LLMVS/eval_sakuga.py"

mkdir -p $SAVE_DIR

echo "========================================================"
echo "LLMVS ABLATION - EVALUATION"
echo "========================================================"
echo "GPU: $GPU_ID"
echo "Option: $OPTION"
echo "Checkpoint: ${CHECKPOINT:-'Random init (no checkpoint provided)'}"
echo "Dataset: $DATASET_ROOT"
echo "Output: $SAVE_DIR"
echo "========================================================"

conda run -n sam python $SCRIPT_PATH \
    --option $OPTION \
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
