#!/bin/bash
# ============================================================
# End-to-End Video Inference Pipeline
# ============================================================
# Input: Raw video file (.mp4, .avi, etc.)
# Output: Keyframe comparisons for V11, VSUMM, LLMVS
#
# Usage: ./scripts/run_video.sh <video_path> [gpu_id]
# Example: ./scripts/run_video.sh data/samples/Sakuga/6261.mp4 0
# ============================================================

VIDEO_PATH=${1}
GPU_ID=${2:-0}

if [ -z "$VIDEO_PATH" ]; then
    echo "Usage: $0 <video_path> [gpu_id]"
    echo "Example: $0 data/samples/Sakuga/6261.mp4 0"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Checkpoints (update if paths change)
V11_CKPT="runs/training_v11_recerr_w0.2/best.pt"
VSUMM_CKPT="runs/ablation_vsumm/sakuga_train/model_epoch60.pth.tar"
LLMVS_CKPT="runs/ablation_llmvs/optionB_visual/best_model.pth"

echo "============================================================"
echo "End-to-End Video Inference"
echo "============================================================"
echo "Video:    $VIDEO_PATH"
echo "GPU:      $GPU_ID"
echo "============================================================"

/srv/conda/envs/serverai/sam/bin/python scripts/run_inference_video.py \
    --video "$VIDEO_PATH" \
    --v11_ckpt "$V11_CKPT" \
    --vsumm_ckpt "$VSUMM_CKPT" \
    --llmvs_ckpt "$LLMVS_CKPT" \
    --output_dir "demo_outputs/video_inference" \
    --device cuda

echo ""
echo "============================================================"
echo "Done! Check demo_outputs/video_inference for results."
echo "============================================================"
