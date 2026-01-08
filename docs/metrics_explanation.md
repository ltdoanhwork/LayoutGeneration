# Evaluation Metrics Explanation

This document provides a comprehensive explanation of the evaluation metrics used to assess the DSN model's performance. It covers the mathematical formulation, the intuition, and the implementation source for each metric.

## 1. Representativeness Metrics
*Goal: How well does the summary cover the content of the original video?*

### 1.1. RecErr (Reconstruction Error)
*   **What it is**: The "Feature Reconstruction Error". It measures how well the selected keyframes can "reconstruct" the semantic feature space of the entire video.
*   **Formula**: 
    $$ \text{RecErr} = \frac{1}{N} \sum_{i=1}^{N} \min_{k \in \mathcal{K}} d(f_i, f_k) $$
    Where $f_i$ is the feature of frame $i$, $\mathcal{K}$ is the set of keyframes, and $d$ is Cosine Distance ($1 - \text{sim}$).
*   **Intuition**: For every single frame in the video, we find its "closest match" in your summary. If the RecErr is **low**, it means every part of the video is semantically close to at least one keyframe (no missing content).
*   **Source**: `eval/metrics.py` -> `reconstruction_error`

### 1.2. Frechet Distance
*   **What it is**: A distributional distance metric (inspired by FID for GANs). It assumes features follow a Gaussian distribution.
*   **Formula**: 
    $$ \text{FD} = ||\mu_{all} - \mu_{keys}||^2 + \text{Tr}(\Sigma_{all} + \Sigma_{keys} - 2(\Sigma_{all}\Sigma_{keys})^{1/2}) $$
*   **Intuition**: Does the *distribution* of your summary match the *distribution* of the full video?
    *   If RecErr is "Nearest Neighbor", Frechet is "population statistics".
    *   A low FD means the summary has the same mean and variance in feature space as the full video.
*   **Source**: `eval/metrics.py` -> `frechet_distance`

### 1.3. Scene Coverage
*   **What it is**: The percentage of detected scenes that contain at least one keyframe.
*   **Intuition**: Did we miss entire scenes? (e.g., The video has 10 scenes, but we only picked frames from 5 of them). Ideally $100\%$.
*   **Source**: `eval/metrics.py` -> `scene_coverage`

---

## 2. Diversity Metrics
*Goal: Are the keyframes distinct, or did we pick 5 nearly identical frames?*

### 2.1. Redundancy (Mean Cosine)
*   **What it is**: The average cosine similarity between all pairs of selected keyframes.
*   **Intuition**:
    *   **High** redundancy means keyframes look the same.
    *   **Low** redundancy means keyframes are visually distinct.
*   **Source**: `eval/metrics.py` -> `redundancy_cosine`

### 2.2. LPIPS Diversity (Perceptual)
*   **What it is**: Similar to Redundancy, but uses the **LPIPS** (Learned Perceptual Image Patch Similarity) metric instead of Cosine distance.
*   **Intuition**: LPIPS is closer to human perception. Two frames might have different pixels but look perceptually similar. LPIPS catches this better than simple feature distance.
    *   **Higher** is better (more diverse).
*   **Source**: `eval/extra_metrics.py` -> `lpips_diversity`

---

## 3. Aesthetic / Quality Metrics (Anime-Specific)
*Goal: Did we pick the "Sakuga" (good animation) moments?*

### 3.1. MPR (Mean Percentile Rank)
*   **What it is**: The primary reward metric. It calculates where the selected frames rank in terms of quality relative to the *entire* video.
*   **Formula**:
    $$ \text{MPR} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \text{Rank}(q_k) / N $$
    Where $q_k$ is the quality score (e.g., 'Sakuga' score) of keyframe $k$.
*   **Intuition**:
    *   MPR = 0.5: Random selection.
    *   MPR = 1.0: You picked strictly the highest-scoring frames in the video.
    *   MPR = 0.9: You consistently picked valid high-quality moments.
*   **Source**: `src/pipeline/train_rl_dsn_v11_simple.py` and distribution logs.

### 3.2. Top-10% Recall
*   **What it is**: What fraction of the "Top 10% best frames" did you manage to catch?
*   **Intuition**: If the video has 1000 frames, the best 100 are the "Gold Standard". If your summary of 10 frames captures 5 of them, your recall is high given the budget.
*   **Source**: `src/pipeline/train_rl_dsn_v11_simple.py`

### 3.3. MS-SWD (Color)
*   **What it is**: Multi-Scale Sliced Wasserstein Distance on color distribution.
*   **Intuition**: Does the summary capture the full *color palette* of the video?
    *   Sakuga often involves vibrant color shifts (explosions, beams). A low MS-SWD means the summary's color histogram matches the full video's color histogram accurately.
*   **Source**: `eval/metrics.py` -> `ms_swd_color`

### 3.4. Anime Attributes (Mean Scores)
*   **What**: Raw average scores for specific CLIP attributes:
    *   **Sakuga**: Dynamic action.
    *   **Sharpness**: Clarity of line art.
    *   **Cinematic**: Composition quality.
*   **Source**: `eval/extra_metrics.py` -> `compute_anime_attr_stats`

---

## 4. Summary Table

| Metric | Category | Lower is Better? | Ideal Behavior |
| :--- | :--- | :---: | :--- |
| **RecErr** | Representativeness | ✅ | Low (Complete semantic coverage) |
| **Frechet** | Representativeness | ✅ | Low (Matching statistical distribution) |
| **SceneCov** | Representativeness | ❌ | High (No distinct scenes missed) |
| **MPR** | Aesthetic | ❌ | High (>0.7), indicates picking top-tier frames |
| **Top10** | Aesthetic | ❌ | High (Capturing the absolute best moments) |
| **LPIPS_Div**| Diversity | ❌ | High (Keyframes are visually distinct) |
| **MS-SWD** | Diversity / Palette | ✅ | Low (Captures full color range) |

