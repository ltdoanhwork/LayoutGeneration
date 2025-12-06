# Anime IQA Optimization V3 - Technical Documentation

> **Version**: 3.0  
> **Status**: Production Ready  
> **Upgrade from**: V1 Premium → V3 Enhanced

---

## Overview

Version 3 represents a significant advancement in anime video summarization using deep reinforcement learning with Anime-CLIP-IQA optimization. This system combines state-of-the-art academic techniques to simultaneously optimize:

- **Anime Quality** (Look, Sakuga, Story via CLIP-IQA)
- **Reconstruction Error** (frame coverage)
- **Frechet Distance** (distribution matching)

### What's New in V3

| Feature | V1 Premium | V3 Enhanced |
|---------|------------|-------------|
| **Quality Assessment** | Custom impl | **torchmetrics CLIP-IQA** ✨ |
| **Curriculum Learning** | 2-stage | **3-stage progressive** ✨ |
| **Reward Normalization** | None | **Running statistics** ✨ |
| **Temporal Consistency** | None | **Smoothness rewards** ✨ |
| **Hard Negative Mining** | None | **Adaptive margins** ✨ |
| **Quality Calibration** | None | **Cross-video normalization** ✨ |

---

## Theory & Motivation

### Multi-Objective Optimization

Video summarization inherently involves conflicting objectives:
- **Diversity**: Select varied frames → may sacrifice quality
- **Quality**: Select best frames → may be redundant
- **Coverage**: Represent all scenes → may include low-quality frames

V3 addresses this using:
1. **Reward Normalization**: Z-score normalize all components to prevent scale imbalances
2. **Curriculum Learning**: Gradually shift focus from coverage → quality → narrative
3. **Adaptive Weighting**: Automatically balance components based on variance

### Curriculum Learning Philosophy

**Stage 1 (Epochs 1-5): Foundation**
- Focus: Learn basic summarization (diversity + representativeness)
- Aesthetic Weight: 10% of final
- Goal: Establish stable selection policy

**Stage 2 (Epochs 6-12): Aesthetic Quality**
- Focus: Ramp up Look and Sakuga rewards
- Aesthetic Weight: 10% → 100% (linear)
- Goal: Learn to identify high-quality frames

**Stage 3 (Epochs 13+): Full Optimization**
- Focus: All objectives including story coherence and temporal flow
- Aesthetic Weight: 100%
- Goal: Optimize all metrics simultaneously

###Hard Negative Mining

Frames just below the quality threshold (within margin) are penalized more heavily:
- **Standard negative**: score < threshold → penalty -1.0
- **Hard negative**: threshold - margin ≤ score < threshold → penalty -1.5

This sharpens the quality boundary and prevents the agent from selecting "almost good" frames.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Loop (V3)                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Load Scene Data                                          │
│     ├─ CLIP features (512-D)                                 │
│     ├─ Anime-CLIP-IQA scores (torchmetrics)                  │
│     └─ RAFT motion features (128-D, optional)                │
│                                                               │
│  2. DSN Forward Pass                                         │
│     ├─ Advanced DSN (Attention + Multi-Scale)                │
│     ├─ Motion Fusion (Cross-Attention)                       │
│     └─ Output: Selection probabilities (T,)                  │
│                                                               │
│  3. Sample Actions (Bernoulli)                               │
│     └─ Binary selection vector                               │
│                                                               │
│  4. Compute Rewards (V3 Enhanced)                            │
│     ├─ Standard: Div + Rep                                   │
│     ├─ Premium Anime: Look + Sakuga + Story                  │
│     │   ├─ Adaptive percentile thresholds                    │
│     │   ├─ Hard negative mining                              │
│     │   └─ Contrastive bonuses                               │
│     ├─ Temporal Smoothness                                   │
│     └─ Quality Variance                                      │
│                                                               │
│  5. Reward Normalization                                     │
│     └─ Z-score normalize using running stats                 │
│                                                               │
│  6. Apply Curriculum Weights                                 │
│     └─ Stage-dependent scaling                               │
│                                                               │
│  7. Policy Gradient Update (REINFORCE)                       │
│     ├─ Advantage: R - baseline                               │
│     ├─ Loss: -advantage * log_prob - entropy_coef * H        │
│     └─ Gradient clipping + EMA weights                       │
│                                                               │
│  8. Validation (Every N Epochs)                              │
│     ├─ Run batch_eval.py on validation set                   │
│     └─ Track RecErr, Frechet, Anime IQA scores               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Loss Functions

### 1. Standard Rewards

**Diversity** (cosine distance):
```
R_div = (1 / K(K-1)) * Σᵢ Σⱼ (1 - cos(fᵢ, fⱼ))  where i ≠ j
```

**Representativeness** (min distance to selected):
```
R_rep = -mean(min_dist(f_all, f_selected))
```

### 2. Premium Anime Rewards (V3)

**Look** (static visual quality):
```
look_score = (sharpness + colorfulness + brightness) / 3
threshold_adaptive = quantile(look_score, p_adapt)  # p_adapt ∈ [0.70, 0.90]
R_look = (n_above - n_below) / n_selected

# Hard negative penalty
R_hn = -0.5 * (n_hard_neg / n_selected)  # where score ∈ [thresh-margin, thresh]

# Contrastive bonus
if mean(look_selected) > mean(look_all) + margin:
    R_look += 0.3
```

**Sakuga** (animation quality):
```
Similar to Look, using sakuga_score attribute
```

**Story** (narrative importance):
```
story_score = (sakuga + cinematic) / 2
R_story = (fraction above threshold)
```

### 3. Temporal Consistency (V3 New)

```
gaps = diff(sorted(selected_indices))
ideal_gap = total_frames / n_selected
gap_variance = var(gaps)

R_temporal = exp(-gap_variance / ideal_gap)  # Higher = smoother
```

### 4. Quality Variance (V3 New)

```
R_quality_var = std(quality_selected) / std(quality_all)
```

### 5. Total Loss

```python
# Compute all components
components = {
    "div": R_div,
    "rep": R_rep,
    "look": R_look,
    "sakuga": R_sakuga,
    "story": R_story,
    "temporal": R_temporal,
    "quality_var": R_qvar
}

# Normalize (Z-score)
for key in components:
    components[key] = (components[key] - running_mean[key]) / running_std[key]

# Apply curriculum weights
weights = get_curriculum_weights(epoch)

# Weighted sum
R_total = sum(w[k] * components[k] for k in components)

# REINFORCE loss
advantage = R_total - baseline
loss = -advantage * log_prob - entropy_coef * H
```

---

## Hyperparameters

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feat_dim` | 512 | CLIP feature dimension |
| `enc_hidden` | 256 | Encoder hidden size |
| `lstm_hidden` | 128 | LSTM hidden size |
| `num_attn_heads` | 4 | Attention heads |
| `num_attn_layers` | 2 | Attention layers |
| `num_scales` | 3 | Multi-scale temporal pooling |
| `dropout` | 0.3 | Dropout rate |

### Premium Reward System (V3)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `percentile_threshold` | 0.75 | [0.6, 0.9] | Base quality percentile |
| `contrastive_margin` | 0.15 | [0.1, 0.3] | Bonus threshold gap |
| `hard_negative_margin` | 0.05 | [0.02, 0.1] | Hard negative range |
| `temporal_weight` | 0.5 | [0.0, 1.0] | Temporal smoothness weight |
| `use_curriculum` | True | bool | Enable 3-stage curriculum |
| `use_quality_calibration` | True | bool | Cross-video normalization |

### Reward Weights

| Component | Stage 1 | Stage 2 | Stage 3 | Description |
|-----------|---------|---------|---------|-------------|
| `w_div` | 1.0 | 0.5 | 0.5 | Diversity weight |
| `w_rep` | 1.0 | 0.5 | 0.5 | Representativeness weight |
| `w_anime_look` | 0.2 | 0.1→2.0 | 2.0 | Look quality weight |
| `w_anime_sakuga` | 0.2 | 0.1→2.0 | 2.0 | Sakuga weight |
| `w_anime_story` | 0.1 | 0.05→1.0 | 1.0 | Story weight |
| `w_temporal` | 0.0 | 0.0 | 0.5 | Temporal smoothness |
| `w_quality_var` | 0.0 | 0.0 | 0.2 | Quality variance |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 20 | Total training epochs |
| `batch_size` | 4 | Gradient accumulation batch |
| `lr` | 1e-4 | Learning rate |
| `weight_decay` | 0.0 | L2 regularization |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `entropy_coef` | 0.01 | Entropy bonus |
| `baseline_momentum` | 0.9 | REINFORCE baseline EMA |

### Budget

| Parameter | Default | Description |
|-----------|---------|-------------|
| `budget_ratio` | 0.06 | Target keyframe ratio |
| `Bmin` | 3 | Minimum keyframes |
| `Bmax` | 15 | Maximum keyframes |
| `budget_penalty` | 0.05 | Penalty for deviation |

---

## Training Guide

### Step 1: Prerequisites

```bash
# Install dependencies
pip install torchmetrics[multimodal]
pip install lpips
pip install opencv-python
pip install tensorboard

# Verify anime CLIP-IQA works
python -m src.models.anime_clipiqa_v3
```

### Step 2: Prepare Dataset

Ensure your dataset follows the structure:
```
data/sakuga_dataset_100_samples/
├─ video_001/
│  ├─ scene_001/
│  │  ├─ features.npy          # CLIP features (T, 512)
│  │  ├─ frames/               # Extracted frames
│  │  ├─ motion_raft.npy       # RAFT motion (T, 128), optional
│  │  └─ anime_attrs.npy       # Anime CLIP-IQA (T, 6)
│  └─ ...
└─ ...
```

### Step 3: Run V3 Training

```bash
bash scripts/rl/train_dsn_v3_anime_premium.sh
```

Monitor training:
```bash
tensorboard --logdir runs/dsn_anime_v3/logs
```

### Step 4: Evaluation

Validation runs automatically every 2 epochs. Check results:
```bash
# View metrics over epochs
python -m eval.visualize_validation \
    --val_dir runs/dsn_anime_v3/val_runs \
    --out_dir runs/dsn_anime_v3/val_runs/plots
```

---

## Monitoring & Debugging

### Key Metrics to Watch

**Training (TensorBoard)**:
1. `train/mean_reward`: Should increase over time
2. `aesthetic/mean_look_selected`: Should increase (target: >0.7)
3. `aesthetic/mean_sakuga_selected`: Should increase (target: >0.65)
4. `curriculum/*`: Check weight scheduling
5. `train/entropy`: Should decrease but stay > 0.01 (prevents collapse)

**Validation**:
1. `RecErr_mean`: Should decrease (lower is better)
2. `Frechet_mean`: Should decrease (lower is better)
3. `LPIPS_PerceptualGap`: Selected vs all (higher is better)

### Common Issues

**Issue**: Reward collapse (all zeros)
- **Cause**: Percentile threshold too high
- **Fix**: Lower `percentile_threshold` to 0.70

**Issue**: No improvement in aesthetics
- **Cause**: Curriculum stuck in Stage 1
- **Fix**: Check `total_epochs` matches actual epochs

**Issue**: High variance in metrics
- **Cause**: Not enough reward normalization samples
- **Fix**: Warmup period needs 10+ scenes before normalization kicks in

**Issue**: Selection too clustered
- **Cause**: Temporal weight too low
- **Fix**: Increase `w_temporal` or `temporal_weight`

---

## Comparison: V1 vs V3

| Aspect | V1 Premium | V3 Enhanced | Improvement |
|--------|------------|-------------|-------------|
| **CLIP-IQA** | Custom impl | torchmetrics (standard) | More reliable |
| **Curriculum** | 2-stage (basic) | 3-stage (progressive) | Better convergence |
| **Reward Handling** | Raw values | Normalized (Z-score) | Balanced gradients |
| **Temporal** | None | Smoothness reward | Better flow |
| **Hard Negatives** | None | Margin-based penalty | Sharper quality boundary |
| **Quality Calibration** | Per-video | Cross-video running stats | Consistent across videos |
| **Training Stability** | Moderate | High (EMA, clipping, norm) | Fewer crashes |
| **Expected RecErr** | ~0.135 | ~0.110 | **~18% better** |
| **Expected Frechet** | ~12.5 | ~10.0 | **~20% better** |
| **Expected Look Score** | ~0.62 | ~0.72 | **+16% quality** |

---

## Advanced Usage

### Custom Anime Prompts

Edit `src/models/anime_clipiqa_v3.py`:
```python
ANIME_PROMPTS = {
    # Add your custom prompts (positive, negative)
    "my_quality": ("High quality anime.", "Low quality anime."),
    # ...
}
```

### Ablation Studies

Test individual V3 components:
```bash
# Disable curriculum
bash scripts/rl/train_dsn_v3_anime_premium.sh --use_curriculum 0

# Disable quality calibration
bash scripts/rl/train_dsn_v3_anime_premium.sh --use_quality_calibration 0

# Disable temporal smoothness
bash scripts/rl/train_dsn_v3_anime_premium.sh --w_temporal 0.0
```

### Backward Compatibility

Run V1 training (no code changes needed):
```bash
bash scripts/rl/train_dsn_multi_anime_premium.sh  # V1 script (unchanged)
```

---

## Troubleshooting

### Installation Issues

**Error**: `ModuleNotFoundError: No module named 'torchmetrics.multimodal'`
```bash
pip install torchmetrics[multimodal] --upgrade
```

**Error**: Component compatibility mismatch
```bash
pip install torch>=2.0 torchmetrics>=1.0 transformers>=4.10
```

### Runtime Issues

**Error**: CUDA out of memory during CLIP-IQA
- **Fix**: Reduce batch size or use smaller CLIP model:
  ```python
  iqa = AnimeClipIQA(model_name="openai/clip-vit-base-patch32")  # Smaller than large-patch14
  ```

**Error**: Evaluation hangs
- **Fix**: Check `--eval_max_videos` limit and `--num_workers`

---

## Citation

If you use this system, please cite:

```bibtex
@misc{anime_iqa_v3_2025,
  title={Anime Video Summarization with Multi-Objective Reinforcement Learning and CLIP-IQA},
  author={Your Name},
  year={2025},
  note={Version 3 Enhancement}
}
```

---

## Changelog

### V3.0 (2025-12-06)
- ✨ Integrated torchmetrics CLIP-IQA for standardized quality assessment
- ✨ Implemented 3-stage progressive curriculum learning
- ✨ Added reward normalization using running statistics
- ✨ Introduced temporal consistency rewards
- ✨ Implemented hard negative mining with adaptive margins
- ✨ Added cross-video quality calibration
- 📝 Comprehensive documentation and training guide

### V1.0 (Previous)
- Initial premium anime optimization with percentile rewards
- Basic 2-stage curriculum
- Custom anime-CLIP-IQA implementation

---

**For questions or issues, please refer to the implementation plan or training scripts.**
