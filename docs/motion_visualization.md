# Motion Visualization in TensorBoard

## Overview

Added comprehensive motion-related visualizations to the training pipeline when `--use_raft_motion 1` is enabled.

## Motion Metrics Tracked

### 1. Motion Magnitude Statistics
- **`motion/mean_magnitude`**: Average motion magnitude across all frames
- **`motion/std_magnitude`**: Standard deviation of motion magnitudes
- **`motion/max_magnitude`**: Maximum motion magnitude observed
- **`motion/magnitude_distribution`**: Histogram of motion magnitudes

### 2. Motion Feature Statistics
- **`motion/feature_means`**: Histogram of mean values per motion dimension (128 dims)
- **`motion/feature_stds`**: Histogram of std values per motion dimension

### 3. Motion vs Selection Analysis
- **`motion/selected_mean`**: Average motion at selected keyframes
- **`motion/rejected_mean`**: Average motion at rejected frames
- **`motion/selection_ratio`**: Ratio of selected/rejected motion (>1 means DSN prefers high-motion frames)
- **`motion/selected_magnitude`**: Histogram of motion at selected frames
- **`motion/rejected_magnitude`**: Histogram of motion at rejected frames

## Interpretation Guide

### High Selection Ratio (>1.5)
DSN is learning to select high-motion frames → Good for action-heavy content

### Low Selection Ratio (<0.8)
DSN prefers low-motion frames → May indicate static/composition-focused selection

### Balanced Ratio (~1.0)
Motion is not the dominant factor → DSN uses appearance + motion equally

## Viewing in TensorBoard

```bash
tensorboard --logdir runs/dsn_raft_motion/logs --port 6006
```

Navigate to:
- **SCALARS** → `motion/` for time-series plots
- **DISTRIBUTIONS** → `motion/` for histograms
- **HISTOGRAMS** → `motion/` for detailed distributions

## Example Analysis

```
Epoch 10:
  motion/selected_mean: 0.45
  motion/rejected_mean: 0.28
  motion/selection_ratio: 1.61
```

**Interpretation**: DSN learned to prefer frames with 61% higher motion → Effective for dynamic content
