#!/bin/bash
# ============================================================================
# VSUMM-Reinforce Training on Sakuga Data (Enhanced)
# ============================================================================
#
# Features:
# - Training + Evaluation with comprehensive metrics
# - Saves JSON results for comparison
# - Fixed nan diversity issue
#
# Usage: bash scripts/ablation_script/train_vsumm_sakuga.sh [GPU_ID]
# ============================================================================

GPU_ID=${1:-0}
DATASET_ROOT="data/sakuga_dataset_v11"
SAVE_DIR="runs/ablation_vsumm/sakuga_train"
EPOCHS=60

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/ablation/pytorch-vsumm-reinforce
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_PATH="ablation/pytorch-vsumm-reinforce/train_custom.py"

mkdir -p $SAVE_DIR

echo "============================================================================"
echo "VSUMM-Reinforce Training on Sakuga"
echo "============================================================================"
echo "GPU:     $GPU_ID"
echo "Data:    $DATASET_ROOT"
echo "Save:    $SAVE_DIR"
echo "Epochs:  $EPOCHS"
echo "============================================================================"

# ============================================================================
# STEP 1: TRAINING
# ============================================================================
echo -e "\n====== STEP 1: TRAINING ======"

python3 $SCRIPT_PATH \
    --dataset-root $DATASET_ROOT \
    --save-dir $SAVE_DIR \
    --gpu $GPU_ID \
    --max-epoch $EPOCHS \
    --input-dim 512 \
    --hidden-dim 256 \
    --lr 1e-4 \
    --num-episode 5 \
    --beta 0.01 \
    --save-results \
    --verbose

# ============================================================================
# STEP 2: FINAL EVALUATION
# ============================================================================
echo -e "\n====== STEP 2: FINAL EVALUATION ======"

# Find latest checkpoint
CKPT=$(ls -t $SAVE_DIR/model_epoch*.pth.tar 2>/dev/null | head -1)

if [ -n "$CKPT" ]; then
    echo "Evaluating checkpoint: $CKPT"
    
    python3 $SCRIPT_PATH \
        --dataset-root $DATASET_ROOT \
        --save-dir $SAVE_DIR \
        --gpu $GPU_ID \
        --input-dim 512 \
        --hidden-dim 256 \
        --resume "$CKPT" \
        --evaluate \
        --save-results \
        --verbose
else
    echo "No checkpoint found, skipping evaluation"
fi

# ============================================================================
# STEP 3: SUMMARY
# ============================================================================
echo -e "\n====== RESULTS SUMMARY ======"

if [ -f "$SAVE_DIR/eval_results.txt" ]; then
    echo "Evaluation results:"
    cat "$SAVE_DIR/eval_results.txt"
fi

echo ""
echo "============================================================================"
echo "VSUMM Training Complete!"
echo "============================================================================"
echo "Model:   $SAVE_DIR/model_epoch$EPOCHS.pth.tar"
echo "Results: $SAVE_DIR/eval_results.txt"
echo "Logs:    $SAVE_DIR/log_train.txt"
echo "============================================================================"
