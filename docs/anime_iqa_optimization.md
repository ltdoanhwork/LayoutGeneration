# Premium Anime-CLIP-IQA Optimization

## Overview

This module implements an academic-grade optimization strategy to maximize Anime-CLIP-IQA scores (Look, Sakuga, Story) in the DSN video summarization model. It leverages multi-video training with a sophisticated reward system designed to force the model to select only the highest-quality frames.

## Key Innovations

### 1. Percentile-Based Rewards
Instead of using raw aesthetic scores (which can be noisy or scale-dependent), we use a **relative ranking** approach.
- For each video, we compute the **75th percentile** (or other threshold) of aesthetic scores.
- **Reward**: +1.0 for selecting a frame *above* this threshold.
- **Penalty**: -1.0 for selecting a frame *below* this threshold.
- **Effect**: This forces the agent to learn a policy that filters out "average" frames and only picks the "elite" frames, regardless of the video's absolute quality baseline.

### 2. Curriculum Learning
We implement a dynamic weighting schedule to stabilize training:
- **Early Training (Epochs 0-5)**: Low weights for aesthetic rewards. The model focuses on basic summarization tasks (diversity, representativeness) to learn a stable policy.
- **Mid-Late Training (Epochs 5+)**: Aesthetic reward weights linearly ramp up to their maximum values (e.g., 2.0).
- **Effect**: Prevents the agent from collapsing into a degenerate policy (e.g., picking only one "perfect" frame) early on.

### 3. Contrastive Reward Boost
We add a bonus reward if the *mean aesthetic score* of the selected summary is significantly higher (margin > 0.2) than the mean score of the entire video. This explicitly optimizes for **summarization gain**.

## Usage

### Training

Run the premium training script:

```bash
bash scripts/rl/train_dsn_multi_anime_premium.sh
```

### Configuration

Key parameters in `train_dsn_multi_anime_premium.sh`:

- `W_ANIME_LOOK`: Weight for static aesthetic quality (Sharpness, Color, Brightness).
- `W_ANIME_SAKUGA`: Weight for dynamic animation quality.
- `percentile_threshold`: The top fraction of frames to target (default 0.75).
- `use_curriculum`: Enable dynamic weight ramping (1=On).

## Metrics

Monitor these metrics in TensorBoard to verify improvement:

- `aesthetic/mean_look_selected`: Average "Look" score of selected keyframes. Should increase over epochs.
- `aesthetic/mean_sakuga_selected`: Average "Sakuga" score of selected keyframes.
- `curriculum/w_look`: The current weight of the Look reward.
