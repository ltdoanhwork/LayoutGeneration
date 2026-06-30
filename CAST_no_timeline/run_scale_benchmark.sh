#!/bin/bash
# Benchmark runtime under different collage scaling factors on one eval sample.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/srv/conda/envs/serverai/sc/bin/python}"
RUN_EVAL_PY="${SCRIPT_DIR}/run_eval.py"
PLOT_PY="${SCRIPT_DIR}/plot_scale_benchmark.py"

BASE_DIR="/home/serverai/ltdoanh/LayoutGeneration/FINAL_data"
LAYOUT_PATH="${LAYOUT_PATH:-${BASE_DIR}/input_layout/filter_mpeg7/dog-1.png}"
FRAME_DIR="${FRAME_DIR:-${BASE_DIR}/keyframe/frame_eval/animals_15_1/frame}"
MASK_DIR="${MASK_DIR:-${BASE_DIR}/keyframe/frame_eval/animals_15_1/mask}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/layout/scale_benchmark_animals15_dog}"

SCALES=(2 4 8)

mkdir -p "${OUTPUT_ROOT}"

echo "=============================================================="
echo "Scale Benchmark: animals_15_1 + dog layout"
echo "=============================================================="
echo "Layout : ${LAYOUT_PATH}"
echo "Frames : ${FRAME_DIR}"
echo "Masks  : ${MASK_DIR}"
echo "Output : ${OUTPUT_ROOT}"
echo "Python : ${PYTHON_BIN}"
echo ""

for SCALE in "${SCALES[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/scale_${SCALE}"
    echo "--------------------------------------------------------------"
    echo "Running scale=${SCALE}"
    echo "Output: ${OUT_DIR}"
    echo "Command:"
    echo "  ${PYTHON_BIN} ${RUN_EVAL_PY} ${LAYOUT_PATH} ${FRAME_DIR} ${MASK_DIR} ${OUT_DIR} ${SCALE} --no-filter-frames-by-isnet"

    "${PYTHON_BIN}" "${RUN_EVAL_PY}" \
        "${LAYOUT_PATH}" \
        "${FRAME_DIR}" \
        "${MASK_DIR}" \
        "${OUT_DIR}" \
        "${SCALE}" \
        --no-filter-frames-by-isnet
done

echo ""
echo "Rendering benchmark plot..."
"${PYTHON_BIN}" "${PLOT_PY}" "${OUTPUT_ROOT}"

echo ""
echo "Done."
echo "PNG: ${OUTPUT_ROOT}/scale_runtime_chart.png"
