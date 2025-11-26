# Anime-CLIP-IQA Integration Tracks

This document explains the three experimental tracks (A, B, C) designed to integrate **Anime-CLIP-IQA** attributes into the Deep Summarization Network (DSN) pipeline.

The goal is to enhance the summarization quality by leveraging anime-specific aesthetic and semantic attributes (e.g., "sharpness", "sakuga", "cinematic") computed via CLIP.

---

## Overview

We define **Anime-CLIP-IQA attributes** as a set of scores derived from CLIP by comparing frames against specific prompt pairs (e.g., "A sharp anime frame" vs "A blurry anime frame").

We have three strategies to use these attributes:

| Track | Name | Input Features | Reward Function | Philosophy |
| :--- | :--- | :--- | :--- | :--- |
| **A** | **Feature-only** | **CLIP + Anime Attrs** | Standard | *Give the model "eyes" to see the attributes, let it figure out if they are important.* |
| **B** | **Reward-only** | Standard CLIP | **Standard + Anime Rewards** | *Explicitly tell the model what is "good" (reward shaping), but don't change what it sees.* |
| **C** | **Combined** | **CLIP + Anime Attrs** | **Standard + Anime Rewards** | *Full integration: The model sees the attributes AND is incentivized to optimize for them.* |

---

## Track A: Feature-only (Input Augmentation)

In this track, we treat the Anime-CLIP-IQA attributes as **additional sensory information** for the DSN agent.

*   **Mechanism**:
    *   The precomputed attributes (Shape: `T x K`, where K=6) are concatenated with the standard CLIP visual features (Shape: `T x 512`).
    *   The input dimension to the DSN becomes `512 + K`.
    *   The **Reward Function** remains exactly the same as the baseline (Reconstruction Error, Diversity, Representative, etc.).
*   **Hypothesis**:
    *   By explicitly providing "sharpness" or "action" scores as input, the policy network can learn to associate these features with the standard objectives (e.g., maybe "sharp" frames help minimize reconstruction error better).
    *   This is a "soft" integration; the model is not *forced* to pick high-score frames unless they help the baseline tasks.
*   **Script**: `scripts/bash_script/train_track_a.sh`
*   **Key Flags**: `--use_anime_attrs 1`, `--use_anime_reward 0`

## Track B: Reward-only (Reward Shaping)

In this track, we use the attributes to **guide the learning process** through Reinforcement Learning rewards.

*   **Mechanism**:
    *   The input features remain standard CLIP (`T x 512`). The model does *not* see the attribute scores directly in its input vector.
    *   We add new terms to the **Reward Function**:
        *   **R_look**: Reward for high aesthetic scores (Sharpness, Colorfulness, Brightness).
        *   **R_sakuga**: Reward for high action/dynamic scores (Sakuga, Motion).
        *   **R_story**: Reward for covering "beat" moments (high Sakuga/Cinematic scores).
    *   Total Reward = `R_baseline + w_look * R_look + w_sakuga * R_sakuga + w_story * R_story`.
*   **Hypothesis**:
    *   Even without seeing the scores explicitly, the model will learn (via RL trial-and-error) that selecting certain types of frames (which happen to have high attributes) yields higher rewards.
    *   This explicitly enforces a specific "taste" or "style" for the summary (e.g., "prefer action scenes").
*   **Script**: `scripts/bash_script/train_track_b.sh`
*   **Key Flags**: `--use_anime_attrs 0`, `--use_anime_reward 1`

## Track C: Combined (Full Integration)

This track combines both approaches for maximum effect.

*   **Mechanism**:
    *   **Inputs**: The model sees the attributes (`512 + K` dim).
    *   **Rewards**: The model is rewarded for selecting high-attribute frames.
*   **Hypothesis**:
    *   This should be the most effective method. Providing the attributes as input makes the learning problem easier (the model doesn't have to infer "sakuga-ness" from raw CLIP features; it's given directly), while the reward ensures the policy actually optimizes for them.
*   **Script**: `scripts/bash_script/train_track_c.sh`
*   **Key Flags**: `--use_anime_attrs 1`, `--use_anime_reward 1`

---

## Preprocessing

All tracks require the offline preprocessing step to generate the attribute files (`anime_attrs.npy`) for each scene.

*   **Script**: `scripts/prepare_anime_attrs.py`
*   **Output**: Saves `anime_attrs.npy` in each scene directory.
