#!/bin/bash
# Run ablation evaluation on the controlled AIC+MPEG-7 grid with optional
# filtering and the extra Mt metric.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_EVAL_PY="${SCRIPT_DIR}/run_eval.py"

FILTER_MODE="${FILTER_MODE:-off}"
ONLY_EVAL="${ONLY_EVAL:-0}"
LAYOUT_FILTER="${LAYOUT_FILTER:-}"
IMAGE_SET_FILTER="${IMAGE_SET_FILTER:-}"
MODE_FILTER="${MODE_FILTER:-}"

EXTRA_ARGS=()
if [[ "${FILTER_MODE}" == "off" ]]; then
    EXTRA_ARGS+=("--no-filter-frames-by-isnet")
else
    EXTRA_ARGS+=("--filter-frames-by-isnet")
fi

BASE_DIR="/home/serverai/ltdoanh/LayoutGeneration/FINAL_data"
LAYOUT_DIR="${BASE_DIR}/input_layout/filter_mpeg7"
FRAME_DIR="${BASE_DIR}/keyframe/frame_eval"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/layout/Eval_loss_ablation}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/ablation_eval_summary.csv}"
MODE_AVG_CSV="${MODE_AVG_CSV:-${OUTPUT_DIR}/ablation_eval_mode_average.csv}"
RANK_CSV="${RANK_CSV:-${OUTPUT_DIR}/ablation_eval_case_ranking.csv}"

ABLATION_MODES=("full" "wo_cap" "wo_cvt" "wo_fea")
LAYOUTS=(
    "Heart-1.png:heart"
    "device3-1.png:device3"
    "flatfish-1.png:flatfish"
    "Misk-1.png:misk"
    "Glas-1.png:glas"
    "dog-1.png:dog"
)
IMAGE_SETS=(
    "animals_15_1"
    "animals_20_0"
    "baby_25_0"
    "food_30_1"
    "food_50_0"
    "transportation_25_1"
)

contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        [[ "${item}" == "${needle}" ]] && return 0
    done
    return 1
}

metric_from_csv() {
    local csv_path="$1"
    local key="$2"
    if [[ ! -f "${csv_path}" ]]; then
        echo "N/A"
        return
    fi
    local value
    value=$(awk -F',' -v key="${key}" '$1==key {gsub(/\r/, "", $2); print $2; exit}' "${csv_path}" || true)
    if [[ -z "${value}" ]]; then
        echo "N/A"
    else
        echo "${value}"
    fi
}

apply_layout_filter() {
    if [[ -z "${LAYOUT_FILTER}" ]]; then
        return
    fi
    local requested=()
    IFS=',' read -r -a requested <<< "${LAYOUT_FILTER}"
    local filtered=()
    local entry name
    for entry in "${LAYOUTS[@]}"; do
        name="${entry##*:}"
        if contains "${name}" "${requested[@]}"; then
            filtered+=("${entry}")
        fi
    done
    LAYOUTS=("${filtered[@]}")
}

apply_image_set_filter() {
    if [[ -z "${IMAGE_SET_FILTER}" ]]; then
        return
    fi
    local requested=()
    IFS=',' read -r -a requested <<< "${IMAGE_SET_FILTER}"
    local filtered=()
    local name
    for name in "${IMAGE_SETS[@]}"; do
        if contains "${name}" "${requested[@]}"; then
            filtered+=("${name}")
        fi
    done
    IMAGE_SETS=("${filtered[@]}")
}

apply_mode_filter() {
    if [[ -z "${MODE_FILTER}" ]]; then
        return
    fi
    local requested=()
    IFS=',' read -r -a requested <<< "${MODE_FILTER}"
    local filtered=()
    local mode
    for mode in "${ABLATION_MODES[@]}"; do
        if contains "${mode}" "${requested[@]}"; then
            filtered+=("${mode}")
        fi
    done
    ABLATION_MODES=("${filtered[@]}")
}

collect_metrics() {
    local outdir="$1"
    local layout_name="$2"
    local image_set="$3"
    local mode="$4"
    local logfile="${outdir}/run_eval.log"
    local eval_csv="${outdir}/evaluation_metrics.csv"

    local dead=0
    local total="?"
    local cap="N/A"
    local cvt="N/A"
    local fea="N/A"
    local Ma Mc Mo Mn MnTimeline Mt Ms

    if [[ -f "${logfile}" ]]; then
        local dead_line
        dead_line=$(grep -oP '\d+/\d+ cells are invalid' "${logfile}" | tail -1 || true)
        if [[ -n "${dead_line}" ]]; then
            dead=$(echo "${dead_line}" | grep -oP '^\d+')
            total=$(echo "${dead_line}" | grep -oP '/\K\d+')
        else
            total=$(grep -oP 'Optimizing layout for \K\d+' "${logfile}" | tail -1 || echo "?")
        fi

        local last_iter
        last_iter=$(grep '\[iter ' "${logfile}" | tail -1 || true)
        if [[ -n "${last_iter}" ]]; then
            cap=$(echo "${last_iter}" | grep -oP 'cap_res=\K[0-9.]+' | head -1 || echo "N/A")
            cvt=$(echo "${last_iter}" | grep -oP 'cvt_norm=\K[0-9.]+' | head -1 || echo "N/A")
            fea=$(echo "${last_iter}" | grep -oP 'fea=\K[0-9.-]+' | head -1 || echo "N/A")
        fi
    fi

    Ma=$(metric_from_csv "${eval_csv}" "Ma")
    Mc=$(metric_from_csv "${eval_csv}" "Mc")
    Mo=$(metric_from_csv "${eval_csv}" "Mo")
    Mn=$(metric_from_csv "${eval_csv}" "Mn")
    MnTimeline=$(metric_from_csv "${eval_csv}" "Mn_timeline")
    Mt=$(metric_from_csv "${eval_csv}" "Mt")
    Ms=$(metric_from_csv "${eval_csv}" "Ms")

    echo "${layout_name},${image_set},${mode},${dead},${total},${cap},${cvt},${fea},${Ma},${Mc},${Mo},${Mn},${MnTimeline},${Mt},${Ms},${outdir}" >> "${SUMMARY_CSV}"
    printf '  summary: dead=%s/%s cap=%s cvt=%s fea=%s | Ma=%s Mc=%s Mo=%s Mn=%s Mn_timeline=%s Mt=%s Ms=%s\n' \
        "${dead}" "${total}" "${cap}" "${cvt}" "${fea}" \
        "${Ma}" "${Mc}" "${Mo}" "${Mn}" "${MnTimeline}" "${Mt}" "${Ms}"
}

write_reports() {
    python3 - "${SUMMARY_CSV}" "${MODE_AVG_CSV}" "${RANK_CSV}" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

summary_csv = Path(sys.argv[1])
mode_avg_csv = Path(sys.argv[2])
rank_csv = Path(sys.argv[3])
rows = []
with summary_csv.open(newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({k: (v.strip().replace('\r', '') if isinstance(v, str) else v) for k, v in row.items()})

def to_float(value):
    try:
        if value in ('', 'N/A', '?', None):
            return math.nan
        return float(value)
    except Exception:
        return math.nan

metric_columns = ['dead_cells', 'total_cells', 'cap_res', 'cvt_norm', 'fea', 'Ma', 'Mc', 'Mo', 'Mn', 'Mn_timeline', 'Mt', 'Ms']
by_mode = defaultdict(list)
for row in rows:
    by_mode[row['mode']].append(row)
with mode_avg_csv.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['mode', 'num_runs'] + [f'avg_{c}' for c in metric_columns])
    for mode in ['full', 'wo_cap', 'wo_cvt', 'wo_fea']:
        group = by_mode.get(mode, [])
        if not group:
            continue
        out = [mode, len(group)]
        for col in metric_columns:
            vals = [to_float(r[col]) for r in group]
            vals = [v for v in vals if not math.isnan(v)]
            out.append(sum(vals) / len(vals) if vals else math.nan)
        writer.writerow(out)

by_case = defaultdict(list)
for row in rows:
    case_key = f"{row['image_set']}__{row['layout']}"
    by_case[case_key].append(row)
rank_rows = []
for case_key, group in by_case.items():
    spans = {}
    for col in ['Ma', 'Mc', 'Mo', 'Mn', 'Mn_timeline', 'Mt', 'Ms']:
        vals = [to_float(r[col]) for r in group]
        vals = [v for v in vals if not math.isnan(v)]
        spans[col] = (max(vals) - min(vals)) if vals else math.nan
    score_order = sum(spans[k] for k in ['Mn_timeline', 'Mt'] if not math.isnan(spans[k]))
    score_total = score_order + sum(spans[k] for k in ['Ma', 'Mc', 'Mo', 'Mn', 'Ms'] if not math.isnan(spans[k]))
    rank_rows.append([
        case_key,
        len(group),
        score_total,
        score_order,
        spans['Ma'], spans['Mc'], spans['Mo'], spans['Mn'], spans['Mn_timeline'], spans['Mt'], spans['Ms'],
    ])
rank_rows.sort(key=lambda row: (row[2], row[3]), reverse=True)
with rank_csv.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['case', 'num_modes', 'score_total', 'score_order', 'Ma_span', 'Mc_span', 'Mo_span', 'Mn_span', 'Mn_timeline_span', 'Mt_span', 'Ms_span'])
    writer.writerows(rank_rows)
print(f'[summary] mode averages -> {mode_avg_csv}')
print(f'[summary] case ranking  -> {rank_csv}')
if rank_rows:
    print('[summary] top controlled cases for clearer ablation differences:')
    for row in rank_rows[:5]:
        print(f'  - {row[0]} | score_total={row[2]:.6f} | score_order={row[3]:.6f}')
PY
}

apply_layout_filter
apply_image_set_filter
apply_mode_filter

if [[ ${#LAYOUTS[@]} -eq 0 || ${#IMAGE_SETS[@]} -eq 0 || ${#ABLATION_MODES[@]} -eq 0 ]]; then
    echo "[ERROR] Empty layout/image_set/mode selection after filters."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
echo 'layout,image_set,mode,dead_cells,total_cells,cap_res,cvt_norm,fea,Ma,Mc,Mo,Mn,Mn_timeline,Mt,Ms,output_dir' > "${SUMMARY_CSV}"

TOTAL=$(( ${#LAYOUTS[@]} * ${#IMAGE_SETS[@]} * ${#ABLATION_MODES[@]} ))
COUNT=0

echo "=============================================================="
echo " CAST Ablation Evaluation (controlled grid)"
echo "   layouts     = ${#LAYOUTS[@]}"
echo "   image_sets  = ${#IMAGE_SETS[@]}"
echo "   modes       = ${ABLATION_MODES[*]}"
echo "   only_eval   = ${ONLY_EVAL}"
echo "   output      = ${OUTPUT_DIR}"
echo "=============================================================="

for IMAGE_SET in "${IMAGE_SETS[@]}"; do
    for LAYOUT_ENTRY in "${LAYOUTS[@]}"; do
        LAYOUT_FILE="${LAYOUT_ENTRY%%:*}"
        LAYOUT_NAME="${LAYOUT_ENTRY##*:}"
        BASE_OUTPUT_NAME="${IMAGE_SET}_${LAYOUT_NAME}"

        for MODE in "${ABLATION_MODES[@]}"; do
            COUNT=$((COUNT + 1))
            RUN_OUTPUT_DIR="${OUTPUT_DIR}/${BASE_OUTPUT_NAME}/${MODE}"
            mkdir -p "${RUN_OUTPUT_DIR}"

            MODE_ARGS=("${EXTRA_ARGS[@]}")
            if [[ "${MODE}" != "full" ]]; then
                MODE_ARGS+=("--ablation=${MODE}")
            fi

            echo "[$COUNT/$TOTAL] ${BASE_OUTPUT_NAME} | mode=${MODE}"
            if [[ "${ONLY_EVAL}" != "1" ]]; then
                "${PYTHON_BIN}" "${RUN_EVAL_PY}" \
                    "${LAYOUT_DIR}/${LAYOUT_FILE}" \
                    "${FRAME_DIR}/${IMAGE_SET}/frame" \
                    "${FRAME_DIR}/${IMAGE_SET}/mask" \
                    "${RUN_OUTPUT_DIR}" \
                    1 \
                    "${MODE_ARGS[@]}" \
                    2>&1 | tee "${RUN_OUTPUT_DIR}/run_eval.log"
            else
                echo "  [ONLY_EVAL=1] Skipping run_eval.py" | tee "${RUN_OUTPUT_DIR}/run_eval.log"
            fi

            collect_metrics "${RUN_OUTPUT_DIR}" "${LAYOUT_NAME}" "${IMAGE_SET}" "${MODE}"
            echo ""
        done
    done
done

write_reports

echo "=============================================================="
echo " Summary CSV : ${SUMMARY_CSV}"
echo " Mode Avg    : ${MODE_AVG_CSV}"
echo " Case Rank   : ${RANK_CSV}"
echo "=============================================================="
