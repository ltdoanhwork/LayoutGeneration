#!/bin/bash
# ============================================================
# Run Comparison on Single Video (V11 vs VSUMM vs LLMVS)
# ============================================================
# Usage: ./scripts/run_comparison.sh [VIDEO_DIR] [GPU_ID]
# Example: ./scripts/run_comparison.sh data/sakuga_dataset_v11_new_test/70025 0

VIDEO_DIR=${1}
GPU_ID=${2:-0}

if [ -z "$VIDEO_DIR" ]; then
    echo "Usage: $0 [VIDEO_DIR] [GPU_ID]"
    echo "Example: $0 data/sakuga_dataset_v11_new_test/70025 0"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Checkpoints (Update these if paths change)
V11_CKPT="runs/training_v11_recerr_w0.2/best.pt"
VSUMM_CKPT="runs/ablation_vsumm/sakuga_train/model_epoch60.pth.tar"
LLMVS_CKPT="runs/ablation_llmvs/optionB_visual/best_model.pth"

python scripts/run_comparison_example.py \
    --video_dir "$VIDEO_DIR" \
    --v11_ckpt "$V11_CKPT" \
    --vsumm_ckpt "$VSUMM_CKPT" \
    --llmvs_ckpt "$LLMVS_CKPT" \
    --output_dir "demo_outputs/comparison_example" \
    --device cuda
