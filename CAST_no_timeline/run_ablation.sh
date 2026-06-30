#!/bin/bash
# =============================================================================
# ABLATION STUDY: CAST_loss 3-loss model (VIDEO dataset)
#
# This version runs the full pipeline, re-evaluates each result with the
# collage metrics, adds Mt (initial-site order preservation), and ranks the
# videos where the ablations differ the most.
#
# Useful env vars:
#   VIDEO_GROUP=active|commented|all   (default: active)
#   VIDEO_FILTER=a,b,c                 (optional CSV subset)
#   MODE_FILTER=full,wo_cap,...        (optional CSV subset)
#   ONLY_EVAL=0|1                      (skip run.py and only rerun evaluation)
#   OUT_VIDEO=/custom/output/root
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAST_DIR="${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${CAST_DIR}/run.py"
EVAL_PY="${CAST_DIR}/evaluation.py"

DATA_ROOT="/home/serverai/ltdoanh/LayoutGeneration/FINAL_data"
MASK_DIR="${DATA_ROOT}/input_layout/input_custom_mask/cropped"
KF_DIR="${DATA_ROOT}/keyframe/inference_recerr_batch_deduplicate"
OUT_VIDEO="${OUT_VIDEO:-${DATA_ROOT}/layout/ablation_loss_video_eval}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_VIDEO}/ablation_eval_summary.csv}"
MODE_AVG_CSV="${MODE_AVG_CSV:-${OUT_VIDEO}/ablation_eval_mode_average.csv}"
RANK_CSV="${RANK_CSV:-${OUT_VIDEO}/ablation_eval_case_ranking.csv}"

DEFAULT_ISNET_WEIGHT="${CAST_DIR}/isnet-detector/weights/isnetis.ckpt"
GENERAL_ISNET_WEIGHT="/home/serverai/ltdoanh/LayoutGeneration/CAST_loss/isnet-detector/weights/isnet-general-use.pth"

ONLY_EVAL="${ONLY_EVAL:-0}"
VIDEO_GROUP="${VIDEO_GROUP:-active}"
VIDEO_FILTER="${VIDEO_FILTER:-}"
MODE_FILTER="${MODE_FILTER:-}"

ABLATION_MODES=("full" "wo_cap" "wo_cvt" "wo_fea")
ACTIVE_VIDEOS=(
    "Your_name" "Stranger_thing" "Bocchi_the_rock" "Nobody" "Kpop_demon_hunter"
    "Zootopia" "Quintessential" "Golden" "Umaru" "Swapped"
)
COMMENTED_VIDEOS=(
    "Inside_out" "Luca" "Onepiece" "Spider_man" "Avatar3"
    "Project_hail_mary" "Squirrel" "Moana"
)
GENERAL_VIDEOS=("Onepiece" "Spider_man" "Avatar3" "Project_hail_mary" "Squirrel" "Moana" "Swapped")

declare -A VIDEO_MASK
VIDEO_MASK[Your_name]="Your_name.png"
VIDEO_MASK[Stranger_thing]="Stranger_thing.png"
VIDEO_MASK[Bocchi_the_rock]="Bocchi_the_rock.png"
VIDEO_MASK[Nobody]="Nobody.png"
VIDEO_MASK[Kpop_demon_hunter]="Kpop_demon_hunter.png"
VIDEO_MASK[Zootopia]="Zootopia.png"
VIDEO_MASK[Inside_out]="Inside_out.png"
VIDEO_MASK[Quintessential]="Quintessential.png"
VIDEO_MASK[Golden]="Golden.png"
VIDEO_MASK[Luca]="Luca.png"
VIDEO_MASK[Umaru]="Umaru.png"
VIDEO_MASK[Onepiece]="Onepiece.png"
VIDEO_MASK[Spider_man]="Spider_man.png"
VIDEO_MASK[Avatar3]="Avatar3.png"
VIDEO_MASK[Project_hail_mary]="Project_hail_mary.png"
VIDEO_MASK[Squirrel]="Squirrel.png"
VIDEO_MASK[Moana]="Moana.png"
VIDEO_MASK[Swapped]="Swapped.png"

contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [[ "${item}" == "${needle}" ]]; then
            return 0
        fi
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

build_list_from_group() {
    case "${VIDEO_GROUP}" in
        active)
            VIDEO_LIST=("${ACTIVE_VIDEOS[@]}")
            ;;
        commented)
            VIDEO_LIST=("${COMMENTED_VIDEOS[@]}")
            ;;
        all)
            VIDEO_LIST=("${ACTIVE_VIDEOS[@]}" "${COMMENTED_VIDEOS[@]}")
            ;;
        *)
            echo "[ERROR] Unknown VIDEO_GROUP=${VIDEO_GROUP} (use active, commented, or all)"
            exit 1
            ;;
    esac
}

apply_video_filter() {
    if [[ -z "${VIDEO_FILTER}" ]]; then
        return
    fi
    local requested=()
    IFS=',' read -r -a requested <<< "${VIDEO_FILTER}"
    local filtered=()
    local video
    for video in "${VIDEO_LIST[@]}"; do
        if contains "${video}" "${requested[@]}"; then
            filtered+=("${video}")
        fi
    done
    VIDEO_LIST=("${filtered[@]}")
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

is_general_video() {
    local video="$1"
    contains "${video}" "${GENERAL_VIDEOS[@]}"
}

collect_summary_row() {
    local outdir="$1"
    local video="$2"
    local mode="$3"

    local logfile="${outdir}/run_eval.log"
    if [[ ! -f "${logfile}" ]]; then
        logfile="${outdir}/run.log"
    fi
    local eval_csv="${outdir}/evaluation_metrics.csv"

    local dead=0
    local total="?"
    local cap="N/A"
    local cvt="N/A"
    local fea="N/A"

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

    local Ma Mc Mo MnTimeline Mt Ms
    Ma=$(metric_from_csv "${eval_csv}" "Ma")
    Mc=$(metric_from_csv "${eval_csv}" "Mc")
    Mo=$(metric_from_csv "${eval_csv}" "Mo")
    MnTimeline=$(metric_from_csv "${eval_csv}" "Mn_timeline")
    Mt=$(metric_from_csv "${eval_csv}" "Mt")
    Ms=$(metric_from_csv "${eval_csv}" "Ms")

    echo "${video},${mode},${dead},${total},${cap},${cvt},${fea},${Ma},${Mc},${Mo},${MnTimeline},${Mt},${Ms},${outdir}" >> "${SUMMARY_CSV}"
    printf '  summary: dead=%s/%s cap=%s cvt=%s fea=%s | Ma=%s Mc=%s Mo=%s Mn_timeline=%s Mt=%s Ms=%s\n' \
        "${dead}" "${total}" "${cap}" "${cvt}" "${fea}" \
        "${Ma}" "${Mc}" "${Mo}" "${MnTimeline}" "${Mt}" "${Ms}"
}

run_one() {
    local mask_path="$1"
    local frames_path="$2"
    local outdir="$3"
    local mode="$4"
    local video="$5"
    local isnet_weight="$6"

    mkdir -p "${outdir}"
    local logfile="${outdir}/run_eval.log"
    local shape_for_eval="${outdir}/shape_mask_refined.png"
    if [[ ! -f "${shape_for_eval}" ]]; then
        shape_for_eval="${mask_path}"
    fi

    echo ""
    echo "================================================================"
    echo " [${video}] mode=${mode}"
    echo "   mask   = ${mask_path}"
    echo "   frames = ${frames_path}"
    echo "   out    = ${outdir}"
    echo "   isnet  = ${isnet_weight}"
    echo "================================================================"

    if [[ "${ONLY_EVAL}" != "1" ]]; then
        local mode_args=()
        if [[ "${mode}" != "full" ]]; then
            mode_args+=("--ablation=${mode}")
        fi

        "${PYTHON_BIN}" "${RUN_PY}" \
            "${mask_path}" \
            "${frames_path}" \
            "${outdir}" \
            2 \
            --shape-is-mask \
            --filter-frames-by-isnet \
            --isnet-weights="${isnet_weight}" \
            "${mode_args[@]}" \
            2>&1 | tee "${logfile}"
    else
        echo "  [ONLY_EVAL=1] Skipping run.py and re-running evaluation only" | tee "${logfile}"
    fi

    if [[ -f "${outdir}/slicing_result.json" ]]; then
        if [[ -f "${outdir}/shape_mask_refined.png" ]]; then
            shape_for_eval="${outdir}/shape_mask_refined.png"
        fi
        "${PYTHON_BIN}" "${EVAL_PY}" \
            --output_dir "${outdir}" \
            --shape "${shape_for_eval}" \
            2>&1 | tee -a "${logfile}" || echo "  [WARN] evaluation.py failed for ${video}/${mode}"
    else
        echo "  [WARN] Missing slicing_result.json for ${video}/${mode}" | tee -a "${logfile}"
    fi

    collect_summary_row "${outdir}" "${video}" "${mode}"
}

write_summary_reports() {
    python3 - "${SUMMARY_CSV}" "${MODE_AVG_CSV}" "${RANK_CSV}" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

summary_csv = Path(sys.argv[1])
mode_avg_csv = Path(sys.argv[2])
rank_csv = Path(sys.argv[3])
if not summary_csv.exists():
    raise SystemExit(f'missing summary csv: {summary_csv}')

rows = []
with summary_csv.open(newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        clean = {k: (v.strip().replace('\r', '') if isinstance(v, str) else v) for k, v in row.items()}
        rows.append(clean)

def to_float(value):
    try:
        if value in ('', 'N/A', '?', None):
            return math.nan
        return float(value)
    except Exception:
        return math.nan

mode_metrics = defaultdict(list)
for row in rows:
    mode_metrics[row['mode']].append(row)

metric_columns = ['dead_cells', 'total_cells', 'cap_res', 'cvt_norm', 'fea', 'Ma', 'Mc', 'Mo', 'Mn_timeline', 'Mt', 'Ms']
with mode_avg_csv.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['mode', 'num_runs'] + [f'avg_{c}' for c in metric_columns])
    for mode in ['full', 'wo_cap', 'wo_cvt', 'wo_fea']:
        group = mode_metrics.get(mode, [])
        if not group:
            continue
        out = [mode, len(group)]
        for col in metric_columns:
            vals = [to_float(r[col]) for r in group]
            vals = [v for v in vals if not math.isnan(v)]
            out.append(sum(vals) / len(vals) if vals else math.nan)
        writer.writerow(out)

by_video = defaultdict(list)
for row in rows:
    by_video[row['video']].append(row)

rank_rows = []
for video, group in by_video.items():
    spans = {}
    for col in ['Ma', 'Mc', 'Mo', 'Mn_timeline', 'Mt', 'Ms']:
        vals = [to_float(r[col]) for r in group]
        vals = [v for v in vals if not math.isnan(v)]
        spans[col] = (max(vals) - min(vals)) if vals else math.nan

    valid_mt = [(to_float(r['Mt']), r['mode']) for r in group]
    valid_mt = [(v, m) for v, m in valid_mt if not math.isnan(v)]
    best_mt = min(valid_mt)[1] if valid_mt else ''

    valid_mn = [(to_float(r['Mn_timeline']), r['mode']) for r in group]
    valid_mn = [(v, m) for v, m in valid_mn if not math.isnan(v)]
    best_mn = min(valid_mn)[1] if valid_mn else ''

    score_order = 0.0
    for key in ['Mn_timeline', 'Mt']:
        if not math.isnan(spans[key]):
            score_order += spans[key]
    score_total = score_order
    for key in ['Ma', 'Mc', 'Mo', 'Ms']:
        if not math.isnan(spans[key]):
            score_total += spans[key]

    rank_rows.append([
        video,
        len(group),
        score_total,
        score_order,
        spans['Ma'],
        spans['Mc'],
        spans['Mo'],
        spans['Mn_timeline'],
        spans['Mt'],
        spans['Ms'],
        best_mt,
        best_mn,
    ])

rank_rows.sort(key=lambda row: (row[2], row[3]), reverse=True)
with rank_csv.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'video', 'num_modes', 'score_total', 'score_order', 'Ma_span', 'Mc_span', 'Mo_span',
        'Mn_timeline_span', 'Mt_span', 'Ms_span', 'best_mode_by_Mt', 'best_mode_by_Mn_timeline'
    ])
    writer.writerows(rank_rows)

print(f'[summary] mode averages -> {mode_avg_csv}')
print(f'[summary] case ranking  -> {rank_csv}')
if rank_rows:
    print('[summary] top cases for clearer ablation differences:')
    for row in rank_rows[:5]:
        print(f'  - {row[0]} | score_total={row[2]:.6f} | score_order={row[3]:.6f} | best_Mt={row[10]} | best_Mn={row[11]}')
PY
}

build_list_from_group
apply_video_filter
apply_mode_filter

if [[ ${#VIDEO_LIST[@]} -eq 0 ]]; then
    echo "[ERROR] No videos selected. Check VIDEO_GROUP/VIDEO_FILTER."
    exit 1
fi
if [[ ${#ABLATION_MODES[@]} -eq 0 ]]; then
    echo "[ERROR] No ablation modes selected. Check MODE_FILTER."
    exit 1
fi

mkdir -p "${OUT_VIDEO}"
echo 'video,mode,dead_cells,total_cells,cap_res,cvt_norm,fea,Ma,Mc,Mo,Mn_timeline,Mt,Ms,output_dir' > "${SUMMARY_CSV}"

TOTAL=$(( ${#VIDEO_LIST[@]} * ${#ABLATION_MODES[@]} ))
COUNT=0

echo "=============================================================="
echo " CAST Loss Ablation (video)"
echo "   video_group = ${VIDEO_GROUP}"
echo "   videos      = ${VIDEO_LIST[*]}"
echo "   modes       = ${ABLATION_MODES[*]}"
echo "   only_eval   = ${ONLY_EVAL}"
echo "   output      = ${OUT_VIDEO}"
echo "=============================================================="

for video in "${VIDEO_LIST[@]}"; do
    mask_file="${VIDEO_MASK[$video]:-}"
    mask_path="${MASK_DIR}/${mask_file}"
    frames_path="${KF_DIR}/${video}/recerr"

    if [[ ! -f "${mask_path}" ]]; then
        echo "[SKIP] Mask not found: ${mask_path}"
        continue
    fi
    if [[ ! -d "${frames_path}" ]] || [[ -z "$(ls -A "${frames_path}" 2>/dev/null)" ]]; then
        echo "[SKIP] No frames in: ${frames_path}"
        continue
    fi

    if is_general_video "${video}"; then
        isnet_weight="${GENERAL_ISNET_WEIGHT}"
    else
        isnet_weight="${DEFAULT_ISNET_WEIGHT}"
    fi

    n_frames=$(find "${frames_path}" -maxdepth 1 \( -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' \) | wc -l)
    echo ""
    echo "━━━ ${video} (frames=${n_frames}, mask=${mask_file}) ━━━"

    for mode in "${ABLATION_MODES[@]}"; do
        COUNT=$((COUNT + 1))
        outdir="${OUT_VIDEO}/${video}/${mode}"
        echo "[$COUNT/$TOTAL] ${video} | mode=${mode}"
        run_one "${mask_path}" "${frames_path}" "${outdir}" "${mode}" "${video}" "${isnet_weight}"
    done
done

write_summary_reports

echo "=============================================================="
echo " Summary CSV : ${SUMMARY_CSV}"
echo " Mode Avg    : ${MODE_AVG_CSV}"
echo " Case Rank   : ${RANK_CSV}"
echo "=============================================================="
