#!/bin/bash
# ============================================================
# Train Missing Ablation Experiments
# 1. 6_pyscene_fixed_long (dataset ready)
# 2. 6_tnv2_fixed_short   (regenerate dataset first)
# ============================================================

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$PYTHONPATH:$(pwd)

PYTHON="/srv/conda/envs/serverai/sam/bin/python"
OUT_ROOT="runs/ablation_final_reorg"
EPOCHS=60
BASE_ARGS="--diversity_weight 0.3 --rec_weight 0.2 --frechet_weight 0.0"

echo "============================================================"
echo "Training Missing Ablation Experiments"
echo "============================================================"

# ============================================================
# Step 1: Regenerate TNv2 Fixed Short dataset
# ============================================================
echo ""
echo ">> Step 1: Regenerating TNv2 Fixed Short dataset..."
TNSHORT_TRAIN="data/sakuga_dataset_v11_tnv2_short"
TNSHORT_TEST="data/sakuga_dataset_v11_tnv2_short_test"

# Remove broken data
rm -rf "$TNSHORT_TRAIN"
rm -rf "$TNSHORT_TEST"

echo "   Generating Train Set..."
$PYTHON -m scripts.precompute_script.prepare_rl_dataset_v11 \
    --video_dir "data/samples/Sakuga" \
    --out_dir "$TNSHORT_TRAIN" \
    --backend "transnetv2" \
    --min_scene_len 30 \
    --max_scene_len 100 \
    --force_split \
    --device cuda

echo "   Generating Test Set..."
$PYTHON -m scripts.precompute_script.prepare_rl_dataset_v11 \
    --video_dir "data/samples/Sakuga_test" \
    --out_dir "$TNSHORT_TEST" \
    --backend "transnetv2" \
    --min_scene_len 30 \
    --max_scene_len 100 \
    --force_split \
    --device cuda

TNSHORT_SCENES=$(ls "$TNSHORT_TRAIN" 2>/dev/null | wc -l)
echo "   TNv2 Fixed Short: ${TNSHORT_SCENES} scenes generated"

# ============================================================
# Step 2: Train PyScene Fixed Long
# ============================================================
echo ""
echo ">> Step 2: Training 6_pyscene_fixed_long..."
$PYTHON -m src.pipeline.train_rl_dsn_v11_final \
    --save_dir "$OUT_ROOT/6_pyscene_fixed_long" \
    --dataset_root "data/sakuga_dataset_v11_pyscene_long" \
    --val_root "data/sakuga_dataset_v11_pyscene_long_test" \
    --epochs $EPOCHS \
    --device cuda \
    $BASE_ARGS 2>&1 | tee "$OUT_ROOT/6_pyscene_fixed_long.log"

echo "   [$([ ${PIPESTATUS[0]} -eq 0 ] && echo OK || echo FAIL)] 6_pyscene_fixed_long"

# ============================================================
# Step 3: Train TNv2 Fixed Short (if data was generated)
# ============================================================
echo ""
echo ">> Step 3: Training 6_tnv2_fixed_short..."
if [ "$TNSHORT_SCENES" -gt 5 ]; then
    # Remove old broken training output
    rm -rf "$OUT_ROOT/6_tnv2_fixed_short"
    
    $PYTHON -m src.pipeline.train_rl_dsn_v11_final \
        --save_dir "$OUT_ROOT/6_tnv2_fixed_short" \
        --dataset_root "$TNSHORT_TRAIN" \
        --val_root "$TNSHORT_TEST" \
        --epochs $EPOCHS \
        --device cuda \
        $BASE_ARGS 2>&1 | tee "$OUT_ROOT/6_tnv2_fixed_short.log"
    
    echo "   [$([ ${PIPESTATUS[0]} -eq 0 ] && echo OK || echo FAIL)] 6_tnv2_fixed_short"
else
    echo "   [SKIP] Not enough scenes (${TNSHORT_SCENES}). Dataset generation may have failed."
fi

# ============================================================
# Step 4: Run Evaluation on new experiments
# ============================================================
echo ""
echo ">> Step 4: Evaluating new experiments..."
$PYTHON scripts/batch_eval_ablation.py --force

echo ""
echo "============================================================"
echo "Missing Ablation Training Complete!"
echo "Now run: $PYTHON scripts/gen_ablation_table_v2.py > ablation_table.tex"
echo "============================================================"
