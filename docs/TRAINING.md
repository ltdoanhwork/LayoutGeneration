# DSN Training Scripts

## Baseline Training (Recommended)

Stable configuration for production training:

```bash
./scripts/train_baseline.sh
```

**Configuration:**
- 20 epochs
- Validation every 2 epochs on 10 videos
- Balanced reward weights
- Motion flow enabled
- Learning rate: 1e-4
- Budget ratio: 0.06 (6% of frames)

**Expected time:** ~2-3 hours (depends on dataset size)

**Outputs:**
- Checkpoints: `outputs/dsn_runs/baseline_v1/`
- Validation: `outputs/val_runs/baseline_v1/`
- TensorBoard: `runs/dsn_baseline_v1/`

---

## Quick Test

Fast training for debugging and quick experiments:

```bash
./scripts/train_quick.sh
```

**Configuration:**
- 5 epochs
- Validation every 5 epochs on 5 videos
- Same hyperparameters as baseline

**Expected time:** ~30-45 minutes

---

## Custom Training

For custom experiments, use the Python module directly:

```bash
python -m src.pipeline.train_rl_dsn \
  --dataset_root outputs/sakuga_dataset \
  --save_dir outputs/dsn_runs/custom \
  --epochs 10 \
  --device cuda \
  --lr 1e-4 \
  --feat_dim 512 \
  --enc_hidden 256 \
  --lstm_hidden 128 \
  --budget_ratio 0.06 \
  --Bmin 3 \
  --Bmax 15 \
  --w_div 1.0 \
  --w_rep 1.0 \
  --w_rec 0.5 \
  --val_videos_dir data/samples/Sakuga \
  --val_output_dir outputs/val_runs/custom \
  --validate_every 2 \
  --eval_embedder clip_vitb32 \
  --log_dir runs/custom
```

---

## Hyperparameter Guide

### Learning Rate
- **Default: 1e-4** - Stable for most cases
- **1e-3** - Faster convergence but less stable
- **5e-5** - More stable but slower

### Reward Weights
- **w_div (1.0)** - Diversity (higher = more diverse keyframes)
- **w_rep (1.0)** - Representation (coverage of feature space)
- **w_rec (0.5)** - Reconstruction error
- **w_fd (0.2)** - Frechet distance
- **w_ms (0.2)** - Multi-scale SWD color
- **w_motion (0.2)** - Motion-based selection

### Budget
- **budget_ratio (0.06)** - Target 6% of frames
- **Bmin (3)** - Minimum keyframes per scene
- **Bmax (15)** - Maximum keyframes per scene
- **budget_penalty (0.05)** - Penalty for violating budget

### Model Architecture
- **feat_dim (512)** - Must match embedder (CLIP ViT-B/32 = 512)
- **enc_hidden (256)** - Encoder hidden size
- **lstm_hidden (128)** - LSTM hidden size
- **dropout (0.3)** - Regularization

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/dsn_baseline_v1
```

**Available visualizations:**
- **SCALARS**: Training/validation metrics over time
- **IMAGES**: Selected vs rejected keyframes
- **HISTOGRAMS**: Probability and reward distributions
- **TEXT**: Hyperparameters

### Validation Results

```bash
# View summary
python -m eval.print_summary \
  --summary outputs/val_runs/baseline_v1/ep20/summary_results.json

# Create plots
python -m eval.visualize_validation \
  --val_output_dir outputs/val_runs/baseline_v1 \
  --output_dir outputs/val_runs/baseline_v1/plots
```

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--eval_max_videos` (default: 10)
- Use smaller batch in validation
- Disable motion: `--use_motion 0`

### Training Too Slow
- Reduce `--ms_swd_scales` (default: 3)
- Reduce `--ms_swd_dirs` (default: 16)
- Increase `--validate_every` (default: 2)

### Poor Performance
- Check if embedder matches training data
- Increase `--epochs` (default: 20)
- Adjust reward weights
- Enable LPIPS diversity: `--use_lpips_div 1`

---

## Best Practices

1. **Always use screen/tmux** for long training:
   ```bash
   screen -S dsn_training
   ./scripts/train_baseline.sh
   # Detach: Ctrl+A, D
   # Reattach: screen -r dsn_training
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Check logs regularly**:
   ```bash
   tail -f runs/dsn_baseline_v1/events.out.tfevents.*
   ```

4. **Save important runs**:
   ```bash
   cp -r outputs/dsn_runs/baseline_v1 outputs/dsn_runs/baseline_v1_backup
   ```
