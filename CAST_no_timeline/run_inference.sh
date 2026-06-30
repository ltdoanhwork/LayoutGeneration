#!/bin/bash
# =============================================================================
# INFERENCE: CAST_loss full-loss model (VIDEO ONLY)
#   Goal: run the same setup as ablation, but only full-loss inference
# =============================================================================
set -e

CAST_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/home/serverai/ltdoanh/LayoutGeneration/FINAL_data"
MASK_DIR="${DATA_ROOT}/input_layout/input_custom_mask/cropped"
KF_DIR="${DATA_ROOT}/keyframe/inference_recerr_batch_deduplicate"

# Output root
OUT_VIDEO="${DATA_ROOT}/layout/inference_loss_video_v4_filter"

DEFAULT_ISNET_WEIGHT="${CAST_DIR}/isnet-detector/weights/isnetis.ckpt"
GENERAL_ISNET_WEIGHT="/home/serverai/ltdoanh/LayoutGeneration/CAST_loss/isnet-detector/weights/isnet-general-use.pth"

# ─────────────────────────────────────────────────────────────
# Video → Mask mapping
# ─────────────────────────────────────────────────────────────
declare -A VIDEO_MASK
VIDEO_MASK[Your_name]="Your_name.png"
VIDEO_MASK[Nobody]="Nobody.png"
VIDEO_MASK[Kpop_demon_hunter]="Kpop_demon_hunter.png"
VIDEO_MASK[Zootopia]="Zootopia.png"
VIDEO_MASK[Inside_out]="Inside_out.png"
VIDEO_MASK[Quintessential]="Quintessential.png"
VIDEO_MASK[Stranger_thing]="Stranger_thing.png"
VIDEO_MASK[Golden]="Golden.png"
VIDEO_MASK[Luca]="Luca.png"
VIDEO_MASK[Bocchi_the_rock]="Bocchi_the_rock.png"
VIDEO_MASK[Umaru]="Umaru.png"
# General videos
VIDEO_MASK[Onepiece]="Onepiece.png"
VIDEO_MASK[Spider_man]="Spider_man.png"
VIDEO_MASK[Avatar3]="Avatar3.png"
VIDEO_MASK[Project_hail_mary]="Project_hail_mary.png"
VIDEO_MASK[Squirrel]="Squirrel.png"
VIDEO_MASK[Moana]="Moana.png"

GENERAL_VIDEOS=(
	"Onepiece"
	"Spider_man"
	"Avatar3"
	"Project_hail_mary"
	"Squirrel"
	"Moana"
)

# ─────────────────────────────────────────────────────────────
# Helper: run one inference + extract dead-cell count
# ─────────────────────────────────────────────────────────────
run_one() {
	local mask="$1"
	local frames="$2"
	local outdir="$3"
	local label="$4"
	local isnet_weight="$5"
	local extra_flags="${6:-}"   # optional extra flags

	echo ""
	echo "================================================================"
	echo " [${label}]  mode=full"
	echo "   mask   = ${mask}"
	echo "   frames = ${frames}"
	echo "   out    = ${outdir}"
	echo "   isnet  = ${isnet_weight}"
	echo "================================================================"

	mkdir -p "${outdir}"

	# Run pipeline (full loss = no --ablation flag)
	python run.py \
		"${mask}" \
		"${frames}" \
		"${outdir}" \
		2 \
		--shape-is-mask \
		--filter-frames-by-isnet \
		--isnet-weights="${isnet_weight}" \
		${extra_flags}

	# Extract dead-cell info from run.log
	local logfile="${outdir}/run.log"
	if [ -f "${logfile}" ]; then
		local dead_line
		dead_line=$(grep -i "cells are invalid" "${logfile}" | tail -1 || true)
		if [ -n "${dead_line}" ]; then
			echo "  >>> DEAD CELLS: ${dead_line}"
		else
			echo "  >>> DEAD CELLS: 0 (none reported)"
		fi
		# Also extract final loss values
		local last_iter
		last_iter=$(grep "\[iter " "${logfile}" | tail -1 || true)
		if [ -n "${last_iter}" ]; then
			echo "  >>> FINAL LOSS: ${last_iter}"
		fi
	fi

	echo "  >>> DONE: ${label} / full"
}

# ─────────────────────────────────────────────────────────────
# Video datasets (full-loss inference)
# ─────────────────────────────────────────────────────────────
echo ""
echo "=============================================================="
echo " Video Dataset Inference (${#VIDEO_MASK[@]} videos)"
echo "=============================================================="

for video in "${!VIDEO_MASK[@]}"; do
	mask_file="${VIDEO_MASK[$video]}"
	mask_path="${MASK_DIR}/${mask_file}"
	frames_path="${KF_DIR}/${video}/recerr"

	# Validate paths
	if [ ! -f "${mask_path}" ]; then
		echo "[SKIP] Mask not found: ${mask_path}"
		continue
	fi
	if [ ! -d "${frames_path}" ] || [ -z "$(ls -A "${frames_path}" 2>/dev/null)" ]; then
		echo "[SKIP] No frames in: ${frames_path}"
		continue
	fi

	n_frames=$(find "${frames_path}" -maxdepth 1 \( -name '*.jpg' -o -name '*.png' \) | wc -l)
	is_general=false
	for g in "${GENERAL_VIDEOS[@]}"; do
		if [ "${video}" = "${g}" ]; then
			is_general=true
			break
		fi
	done
	if [ "${is_general}" = true ]; then
		isnet_weight="${GENERAL_ISNET_WEIGHT}"
	else
		isnet_weight="${DEFAULT_ISNET_WEIGHT}"
	fi

	echo ""
	echo "━━━ ${video}  (mask=${mask_file}, frames=${n_frames}) ━━━"
	echo "    ISNet weight: ${isnet_weight}"

	out="${OUT_VIDEO}/${video}/full"
	run_one "${mask_path}" "${frames_path}" "${out}" "${video}" "${isnet_weight}"
done

# ─────────────────────────────────────────────────────────────
# PART 2: Summary — collect dead cells + loss per experiment
# ─────────────────────────────────────────────────────────────
echo ""
echo "=============================================================="
echo " SUMMARY"
echo "=============================================================="

SUMMARY_CSV="${DATA_ROOT}/layout/inference_loss_summary.csv"
echo "dataset,mode,dead_cells,total_cells,cap_res,cvt_norm,fea" > "${SUMMARY_CSV}"

collect_summary() {
	local root_dir="$1"
	for exp_dir in "${root_dir}"/*/full; do
		[ -d "${exp_dir}" ] || continue
		local logfile="${exp_dir}/run.log"
		[ -f "${logfile}" ] || continue

		local dataset mode dead total cap cvt fea
		dataset=$(basename "$(dirname "${exp_dir}")")
		mode="full"

		# Dead cells: "[Voronoi] WARNING: 3/14 cells are invalid: [...]"
		dead_line=$(grep -oP '\d+/\d+ cells are invalid' "${logfile}" | tail -1 || true)
		if [ -n "${dead_line}" ]; then
			dead=$(echo "${dead_line}" | grep -oP '^\d+')
			total=$(echo "${dead_line}" | grep -oP '/\K\d+')
		else
			dead=0
			total=$(grep -oP 'Optimizing layout for \K\d+' "${logfile}" | tail -1 || echo "?")
		fi

		# Final losses from last [iter ...] line
		last_iter=$(grep "\[iter " "${logfile}" | tail -1 || true)
		cap=$(echo "${last_iter}" | grep -oP 'cap_res=\K[0-9.]+' | head -1 || echo "N/A")
		cvt=$(echo "${last_iter}" | grep -oP 'cvt_norm=\K[0-9.]+' | head -1 || echo "N/A")
		fea=$(echo "${last_iter}" | grep -oP 'fea=\K[0-9.-]+' | head -1 || echo "N/A")

		echo "${dataset},${mode},${dead},${total},${cap},${cvt},${fea}" >> "${SUMMARY_CSV}"
		printf "  %-25s %-8s  dead=%s/%s  cap=%s cvt=%s fea=%s\n" \
			"${dataset}" "${mode}" "${dead}" "${total}" "${cap}" "${cvt}" "${fea}"
	done
}

echo ""
echo "--- Video datasets ---"
collect_summary "${OUT_VIDEO}"

echo ""
echo "Summary CSV: ${SUMMARY_CSV}"
echo ""
echo "ALL INFERENCE EXPERIMENTS COMPLETE."
