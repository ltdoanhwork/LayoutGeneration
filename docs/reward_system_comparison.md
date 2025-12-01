# Reward System Comparison: V1 vs V2

## Overview

This document compares the old (V1) and new (V2) reward systems for anime keyframe selection, highlighting the critical improvements made.

## Problem in V1: Double-Counting

### V1 Architecture (❌ DEPRECATED)

The old system had anime IQA rewards contributing **twice**:

```python
# In reward_combo (rewards.py):
R_anime = anime_iqa_emphasis(anime_scores, sel_idx)  # First time
R += w_anime * R_anime

# In train_rl_dsn.py (Track B):
R_look = (sharp + color + bright) / 3  # Second time!
R_sakuga = sakuga * motion
R_story = beat_coverage
R += w_look * R_look + w_sakuga * R_sakuga + w_story * R_story
```

**Issues:**
- Same `anime_scores` data used twice → inflated gradients
- `R_look` and anime_iqa_emphasis both use sharpness/color → conflicting signals
- No scale normalization → `R_ms ≈ 0.02` vs `R_div ≈ 1.0` → arbitrary weighting

## Solution in V2: Unified Triad

### V2 Architecture (✅ CURRENT)

Single unified function with three clear components:

```python
def anime_reward(attrs_all, sel_idx, motion=None):
    """
    Returns: {"look": R_look, "sakuga": R_sakuga, "story": R_story}
    """
    # R_look: Normalized z-score
    look_all = (sharp + color + bright) / 3
    R_look = (look_sel.mean() - look_all.mean()) / look_all.std()
    
    # R_sakuga: Combined sakuga + motion
    if motion:
        combined = 0.5 * sakuga + 0.5 * motion_norm
    R_sakuga = (combined_sel.mean() - combined_all.mean()) / combined_all.std()
    
    # R_story: Beat coverage ratio
    beat = 0.5 * sakuga + 0.5 * cinematic
    threshold = beat.mean() + 0.5 * beat.std()
    R_story = (beat_sel > threshold).mean()
```

**Improvements:**
- ✅ No double-counting: anime data used **once**
- ✅ Clear semantics: Look (aesthetic), Sakuga (dynamic), Story (narrative)
- ✅ Normalized scales: z-scores → balanced gradients
- ✅ Separate control: `w_anime_look`, `w_anime_sakuga`, `w_anime_story`

## Mathematical Formulation

### Look Component
$$
R_{\text{look}} = \frac{\mu_{\text{look}}^{\text{sel}} - \mu_{\text{look}}^{\text{all}}}{\sigma_{\text{look}}^{\text{all}}}
$$
where $\text{look} = \frac{\text{sharpness} + \text{color} + \text{brightness}}{3}$

### Sakuga Component
$$
R_{\text{sakuga}} = \frac{\mu_{\text{combined}}^{\text{sel}} - \mu_{\text{combined}}^{\text{all}}}{\sigma_{\text{combined}}^{\text{all}}}
$$
where $\text{combined} = 0.5 \cdot \text{sakuga}_{\text{score}} + 0.5 \cdot \text{motion}_{\text{norm}}$

### Story Component
$$
R_{\text{story}} = \frac{1}{|S|} \sum_{i \in S} \mathbb{1}[\text{beat}_i > \text{threshold}]
$$
where $\text{beat} = 0.5 \cdot \text{sakuga} + 0.5 \cdot \text{cinematic}$

## Training Scripts

### V1 (Archived)
```bash
# OLD - DO NOT USE
--w_anime 0.5     # Gets double-counted!
--w_look 0.5      # Plus these...
--w_sakuga 0.7
--w_story 0.0
```

### V2 (Current)
```bash
# NEW - Use this
--w_anime_look 0.3
--w_anime_sakuga 0.4
--w_anime_story 0.2
```

## Running Comparisons

```bash
# Quick comparison (10 epochs)
bash scripts/dsn_bash_script/compare_v1_v2.sh

# View results
tensorboard --logdir runs/reward_comparison --port 6006
```

## Expected Improvements

1. **Gradient Stability**: No conflicting signals from double-counting
2. **Faster Convergence**: Balanced reward scales → consistent updates
3. **Better Interpretability**: Clear Look-Sakuga-Story metrics in TensorBoard
4. **Higher Quality**: Proper weighting of aesthetic vs dynamic vs narrative aspects

## Migration Guide

**Old code:**
```python
R = reward_combo(..., w_anime=0.5, ...)
# Track B manually adds R_look, R_sakuga, R_story
```

**New code:**
```python
R = reward_combo(
    ..., 
    w_anime_look=0.3,
    w_anime_sakuga=0.4, 
    w_anime_story=0.2
)
# No Track B needed - unified in reward_combo
```

## Files Changed

- `src/rl/rewards.py`: New `anime_reward()`, updated `reward_combo()`
- `src/rl/rewards_v1_old.py`: Backup of old implementation
- `src/pipeline/train_rl_dsn.py`: Removed Track B, new arguments
- `scripts/dsn_bash_script/train_track_b.sh`: Updated weights
- `scripts/dsn_bash_script/train_track_d_anime.sh`: Updated weights
- `scripts/dsn_bash_script/compare_v1_v2.sh`: New comparison script

## References

For detailed implementation, see:
- [Implementation Plan](file:///home/serverai/.gemini/antigravity/brain/08a0e703-396d-4222-88d5-482a220f1dc8/implementation_plan.md)
- [Code: rewards.py](file:///home/serverai/ltdoanh/LayoutGeneration/src/rl/rewards.py)
