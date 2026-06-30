#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${SCRIPT_DIR}/run.py"

MASK_PATH="${MASK_PATH:-/home/serverai/ltdoanh/LayoutGeneration/FINAL_data/input_layout/input_custom_mask/cropped/Squirrel.png}"
FRAMES_PATH="${FRAMES_PATH:-/home/serverai/ltdoanh/LayoutGeneration/FINAL_data/keyframe/inference_recerr_batch_deduplicate/Luca/recerr}"
OUT_BASE="${OUT_BASE:-/home/serverai/ltdoanh/LayoutGeneration/FINAL_data/layout/TEST/Luca_Squirrel}"
ISNET_WEIGHTS="${ISNET_WEIGHTS:-/home/serverai/ltdoanh/LayoutGeneration/CAST_loss/isnet-detector/weights/isnetis.ckpt}"

# tag|w_capacity|w_cvt_norm|w_fea|lr_sites|lr_weights|tau|warmup_ratio|core_alpha
CONFIG_LIST=(
    "wc30_cv20_f15_ls0.005_lw0.01_tau30_wu0.10_ca0.20|30|20|15|0.005|0.01|30|0.10|0.20"
    "wc30_cv40_f30_ls0.015_lw0.025_tau60_wu0.25_ca0.325|30|40|30|0.015|0.025|60|0.25|0.325"
    "wc30_cv60_f45_ls0.025_lw0.04_tau90_wu0.40_ca0.45|30|60|45|0.025|0.04|90|0.40|0.45"
    "wc50_cv20_f30_ls0.005_lw0.025_tau90_wu0.25_ca0.20|50|20|30|0.005|0.025|90|0.25|0.20"
    "wc50_cv40_f15_ls0.015_lw0.01_tau60_wu0.40_ca0.325|50|40|15|0.015|0.01|60|0.40|0.325"
    "wc50_cv60_f45_ls0.025_lw0.04_tau30_wu0.10_ca0.45|50|60|45|0.025|0.04|30|0.10|0.45"
    "wc70_cv20_f45_ls0.015_lw0.04_tau60_wu0.10_ca0.20|70|20|45|0.015|0.04|60|0.10|0.20"
    "wc70_cv40_f15_ls0.025_lw0.025_tau30_wu0.25_ca0.45|70|40|15|0.025|0.025|30|0.25|0.45"
    "wc70_cv60_f30_ls0.005_lw0.01_tau90_wu0.40_ca0.325|70|60|30|0.005|0.01|90|0.40|0.325"
    "wc90_cv20_f15_ls0.025_lw0.01_tau60_wu0.25_ca0.45|90|20|15|0.025|0.01|60|0.25|0.45"
    "wc90_cv40_f45_ls0.005_lw0.025_tau30_wu0.40_ca0.20|90|40|45|0.005|0.025|30|0.40|0.20"
    "wc90_cv60_f30_ls0.015_lw0.04_tau90_wu0.10_ca0.325|90|60|30|0.015|0.04|90|0.10|0.325"
    "wc30_cv20_f30_ls0.025_lw0.025_tau60_wu0.40_ca0.20|30|20|30|0.025|0.025|60|0.40|0.20"
    "wc30_cv40_f45_ls0.005_lw0.04_tau90_wu0.10_ca0.325|30|40|45|0.005|0.04|90|0.10|0.325"
    "wc30_cv60_f15_ls0.015_lw0.01_tau30_wu0.25_ca0.45|30|60|15|0.015|0.01|30|0.25|0.45"
    "wc50_cv20_f45_ls0.015_lw0.01_tau30_wu0.40_ca0.20|50|20|45|0.015|0.01|30|0.40|0.20"
    "wc50_cv40_f30_ls0.025_lw0.025_tau90_wu0.10_ca0.45|50|40|30|0.025|0.025|90|0.10|0.45"
    "wc50_cv60_f15_ls0.005_lw0.04_tau60_wu0.25_ca0.325|50|60|15|0.005|0.04|60|0.25|0.325"
    "wc70_cv20_f15_ls0.005_lw0.025_tau30_wu0.25_ca0.20|70|20|15|0.005|0.025|30|0.25|0.20"
    "wc70_cv40_f45_ls0.015_lw0.01_tau90_wu0.40_ca0.325|70|40|45|0.015|0.01|90|0.40|0.325"
    "wc70_cv60_f30_ls0.025_lw0.04_tau60_wu0.10_ca0.45|70|60|30|0.025|0.04|60|0.10|0.45"
    "wc90_cv20_f30_ls0.015_lw0.025_tau90_wu0.40_ca0.20|90|20|30|0.015|0.025|90|0.40|0.20"
    "wc90_cv40_f15_ls0.025_lw0.04_tau60_wu0.10_ca0.325|90|40|15|0.025|0.04|60|0.10|0.325"
    "wc90_cv60_f45_ls0.005_lw0.01_tau30_wu0.25_ca0.45|90|60|45|0.005|0.01|30|0.25|0.45"
)

for cfg in "${CONFIG_LIST[@]}"; do
    IFS='|' read -r CFG_TAG W_CAP W_CVT W_FEA LR_SITES LR_WEIGHTS TAU WARMUP CORE_ALPHA <<<"${cfg}"

    export CAST_VORONOI_W_CAPACITY="${W_CAP}"
    export CAST_VORONOI_W_CVT_NORM="${W_CVT}"
    export CAST_VORONOI_W_FEA="${W_FEA}"
    export CAST_VORONOI_LR_SITES="${LR_SITES}"
    export CAST_VORONOI_LR_WEIGHTS="${LR_WEIGHTS}"
    export CAST_VORONOI_TAU="${TAU}"
    export CAST_VORONOI_WARMUP_RATIO="${WARMUP}"
    export CAST_VORONOI_CORE_ALPHA="${CORE_ALPHA}"

    RUN_OUTPUT_DIR="${OUT_BASE}_${CFG_TAG}"
    mkdir -p "${RUN_OUTPUT_DIR}"

    echo ""
    echo "=============================================="
    echo "CONFIG: ${CFG_TAG}"
    echo "  w_capacity=${W_CAP} w_cvt_norm=${W_CVT} w_fea=${W_FEA}"
    echo "  lr_sites=${LR_SITES} lr_weights=${LR_WEIGHTS} tau=${TAU}"
    echo "  warmup_ratio=${WARMUP} core_alpha=${CORE_ALPHA}"
    echo "  output=${RUN_OUTPUT_DIR}"
    echo "=============================================="

    "${PYTHON_BIN}" "${RUN_PY}" \
        "${MASK_PATH}" \
        "${FRAMES_PATH}" \
        "${RUN_OUTPUT_DIR}" \
        2 \
        --shape-is-mask \
        --filter-frames-by-isnet \
        --isnet-weights="${ISNET_WEIGHTS}"

done

echo ""
echo "Done."
