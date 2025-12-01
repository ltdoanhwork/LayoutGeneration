# Gradient Visualization in TensorBoard

## Overview

Added comprehensive gradient visualization to `train_rl_dsn.py` to monitor gradient flow and identify potential training issues.

## Metrics Added

### 1. Gradient Norm Statistics

**Before Clipping:**
- `gradients/norm_before_clip_mean` - Mean gradient norm across all scenes in epoch
- `gradients/norm_before_clip_std` - Standard deviation of gradient norms
- `gradients/norm_before_clip_max` - Maximum gradient norm (helps identify exploding gradients)

**After Clipping:**
- `gradients/norm_after_clip_mean` - Mean gradient norm after clipping
- `gradients/norm_after_clip_std` - Standard deviation after clipping
- `gradients/norm_after_clip_max` - Maximum gradient norm after clipping

**Clipping Impact:**
- `gradients/clip_ratio` - Ratio of after/before clipping (should be close to 1.0 if clipping rarely activates)

### 2. Gradient Distributions

**Histograms:**
- `gradients/norm_before_clip` - Distribution of gradient norms before clipping
- `gradients/norm_after_clip` - Distribution of gradient norms after clipping
- `gradients/value_distribution` - Distribution of actual gradient values (sampled)

## Usage

### View in TensorBoard

```bash
tensorboard --logdir runs/dsn_track_b_v2_unified/logs --port 6006
```

Navigate to:
- **SCALARS** → `gradients/` - See gradient statistics over time
- **DISTRIBUTIONS** or **HISTOGRAMS** → `gradients/` - See gradient distributions

### Interpreting Results

#### Healthy Gradients
- `norm_before_clip_mean`: 0.1 - 10.0 (reasonable range)
- `clip_ratio`: 0.8 - 1.0 (clipping rarely activates)
- `value_distribution`: Centered around 0, not too wide

#### Warning Signs
⚠️ **Exploding Gradients:**
- `norm_before_clip_max` >> 10.0
- `clip_ratio` < 0.5 (heavy clipping)
- Gradients increasing over time

⚠️ **Vanishing Gradients:**
- `norm_before_clip_mean` < 0.001
- Very narrow `value_distribution`
- Training stalls

⚠️ **Unstable Training:**
- High `norm_before_clip_std` (inconsistent gradients)
- Erratic `clip_ratio` over epochs

## Implementation Details

### Per-Scene Tracking

During each scene's training step:
```python
# 1. Compute gradients
loss.backward()

# 2. Measure gradient norm BEFORE clipping
for p in model.parameters():
    if p.grad is not None:
        grad_norm_before += p.grad.data.norm(2).item() ** 2

# 3. Apply gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# 4. Measure gradient norm AFTER clipping
for p in model.parameters():
    if p.grad is not None:
        grad_norm_after += p.grad.data.norm(2).item() ** 2
```

### Epoch Summary

At the end of each epoch:
- Aggregate all gradient norms from scenes
- Compute statistics (mean, std, max)
- Log to TensorBoard
- Print summary to console

## Example Console Output

```
[Epoch 5] meanR=2.1234 | sel_ratio=0.0600 | entropy=0.0850 | mean_prob=0.0612 | budget_gap=0.150
  Motion: selected=0.3421, rejected=0.2114, ratio=1.62
  Gradients: before_clip=2.3456, after_clip=2.1234, clip_ratio=0.905
```

## Benefits

1. **Early Detection**: Identify gradient issues before they cause training failure
2. **Hyperparameter Tuning**: Adjust `max_grad_norm` based on actual gradient magnitudes
3. **Model Comparison**: Compare gradient flow between baseline and advanced models
4. **Debugging**: Understand why training might be unstable or slow

## Related Files

- [`train_rl_dsn.py`](file:///home/serverai/ltdoanh/LayoutGeneration/src/pipeline/train_rl_dsn.py) - Main implementation
- Training logs: `runs/*/logs/` - TensorBoard event files
