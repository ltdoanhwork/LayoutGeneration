#!/bin/bash
# Run evaluation on 6x6 combinations of layouts and image sets
# 6 layouts x 6 image sets = 36 runs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_EVAL_PY="${SCRIPT_DIR}/run_eval.py"

# Optional env controls:
#   ABLATION_MODE=full|wo_cap|wo_cvt|wo_fea (default: full)
#   FILTER_MODE=on|off                      (default: off)
ABLATION_MODE="${ABLATION_MODE:-full}"
FILTER_MODE="${FILTER_MODE:-off}"

EXTRA_ARGS=()
if [[ "${ABLATION_MODE}" != "full" ]]; then
    EXTRA_ARGS+=("--ablation=${ABLATION_MODE}")
fi
if [[ "${FILTER_MODE}" == "off" ]]; then
    EXTRA_ARGS+=("--no-filter-frames-by-isnet")
else
    EXTRA_ARGS+=("--filter-frames-by-isnet")
fi

BASE_DIR="/home/serverai/ltdoanh/LayoutGeneration/FINAL_data"
LAYOUT_DIR="${BASE_DIR}/input_layout/filter_mpeg7"
FRAME_DIR="${BASE_DIR}/keyframe/frame_eval"
OUTPUT_DIR="${BASE_DIR}/layout/Eval_loss_no_time"

# 6 layouts
LAYOUTS=(
    "Heart-1.png:heart"
    "device3-1.png:device3"
    "flatfish-1.png:flatfish"
    "Misk-1.png:misk"
    "Glas-1.png:glas"
    "dog-1.png:dog"
)

# 6 image sets
IMAGE_SETS=(
    "animals_15_1"
    "animals_20_0"
    "baby_25_0"
    "food_30_1"
    "food_50_0"
    "transportation_25_1"
)

echo "=============================================="
echo "CAST Evaluation: 6 layouts x 6 image sets"
echo "=============================================="
echo ""

TOTAL=$((${#LAYOUTS[@]} * ${#IMAGE_SETS[@]}))
COUNT=0

for IMAGE_SET in "${IMAGE_SETS[@]}"; do
    for LAYOUT_ENTRY in "${LAYOUTS[@]}"; do
        # Parse layout filename and short name
        LAYOUT_FILE="${LAYOUT_ENTRY%%:*}"
        LAYOUT_NAME="${LAYOUT_ENTRY##*:}"
        
        COUNT=$((COUNT + 1))
        OUTPUT_NAME="${IMAGE_SET}_${LAYOUT_NAME}"
        
        echo "[$COUNT/$TOTAL] Running: ${OUTPUT_NAME}"
        echo "  Layout: ${LAYOUT_FILE}"
        echo "  Images: ${IMAGE_SET}"
        if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
            echo "  Flags: ${EXTRA_ARGS[*]}"
        fi
        
        "${PYTHON_BIN}" "${RUN_EVAL_PY}" \
            "${LAYOUT_DIR}/${LAYOUT_FILE}" \
            "${FRAME_DIR}/${IMAGE_SET}/frame" \
            "${FRAME_DIR}/${IMAGE_SET}/mask" \
            "${OUTPUT_DIR}/${OUTPUT_NAME}" \
            1 \
            "${EXTRA_ARGS[@]}"
        
        echo "  Done: ${OUTPUT_DIR}/${OUTPUT_NAME}"
        echo ""
    done
done

echo "=============================================="
echo "All $TOTAL evaluations completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
