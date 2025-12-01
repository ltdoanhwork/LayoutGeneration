from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np

from eval.metrics import *
from src.distance_selector.registry import create_metric

# Anime-CLIP-IQA attribute indices for clarity
ATTR_INDEX = {
    "sharpness": 0,
    "colorfulness": 1,
    "brightness": 2,
    "sakuga": 3,
    "cinematic": 4,
    "expression": 5,
}

def cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    S = X @ X.T
    return 1.0 - S

def diversity_reward(feats_sel: np.ndarray) -> float:
    K = feats_sel.shape[0]
    if K < 2: return 0.0
    D = cosine_dist_matrix(feats_sel)
    sum_pair = (np.sum(D) - np.trace(D))
    return float(sum_pair / (K*(K-1)))

def diversity_reward_lpips(frames_sel, lpips_net="alex", device="cuda", max_pairs=2000):
    """
    Perceptual diversity using LPIPS on selected frames (list of HxWx3, BGR).
    Returns average pairwise LPIPS distance.
    """
    import torch, numpy as np
    K = len(frames_sel)
    if K < 2:
        return 0.0

    metric = create_metric("lpips", net=lpips_net, device=device)
    # Preprocess once per frame
    tensors = [metric.preprocess_bgr(fr) for fr in frames_sel]

    # Build all pairs or a random subset for speed
    pairs = []
    for i in range(K):
        for j in range(i+1, K):
            pairs.append((i, j))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(123)
        pairs = [pairs[idx] for idx in rng.choice(len(pairs), size=max_pairs, replace=False)]

    vals = []
    with torch.no_grad():
        for i, j in pairs:
            d = metric.pair_distance(tensors[i], tensors[j])
            vals.append(float(d))
    return float(np.mean(vals)) if vals else 0.0

def representativeness_reward(feats_all: np.ndarray, feats_sel: np.ndarray) -> float:
    if feats_sel.shape[0] == 0: return 0.0
    D_all_sel = 1.0 - (feats_all @ feats_sel.T)
    min_dist = np.min(D_all_sel, axis=1)
    return float(- np.mean(min_dist))

def anime_reward(
    attrs_all: np.ndarray,
    sel_idx: List[int],
    motion: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Unified anime reward with three conceptually distinct components.
    
    Args:
        attrs_all: (T, K) Anime-CLIP IQA scores for all frames
        sel_idx: List of selected frame indices
        motion: (T,) Optional motion magnitudes for each frame
    
    Returns:
        Dict with keys: "look", "sakuga", "story"
        - look: Static aesthetic quality (normalized)
        - sakuga: Dynamic sakuga quality combined with motion (normalized)
        - story: Narrative beat coverage (fraction above threshold)
    """
    if len(sel_idx) == 0 or attrs_all.shape[0] == 0:
        return {"look": 0.0, "sakuga": 0.0, "story": 0.0}
    
    # Filter valid indices
    sel_idx_valid = [i for i in sel_idx if i < len(attrs_all)]
    if len(sel_idx_valid) == 0:
        return {"look": 0.0, "sakuga": 0.0, "story": 0.0}
    
    # Extract attributes using ATTR_INDEX
    sharp_all = attrs_all[:, ATTR_INDEX["sharpness"]]
    color_all = attrs_all[:, ATTR_INDEX["colorfulness"]]
    bright_all = attrs_all[:, ATTR_INDEX["brightness"]]
    sakuga_all = attrs_all[:, ATTR_INDEX["sakuga"]]
    cinema_all = attrs_all[:, ATTR_INDEX["cinematic"]]
    
    # Selected frames
    sharp_sel = sharp_all[sel_idx_valid]
    color_sel = color_all[sel_idx_valid]
    bright_sel = bright_all[sel_idx_valid]
    sakuga_sel = sakuga_all[sel_idx_valid]
    
    # --- R_look: Static aesthetic quality (normalized z-score) ---
    # Mean of (sharp + color + brightness) for selected vs all
    look_all = (sharp_all + color_all + bright_all) / 3.0
    look_sel = (sharp_sel + color_sel + bright_sel) / 3.0
    
    look_mean_all = float(look_all.mean())
    look_std_all = float(look_all.std())
    look_mean_sel = float(look_sel.mean())
    
    # Normalized: how much better is selection compared to global mean
    R_look = (look_mean_sel - look_mean_all) / (look_std_all + 1e-6)
    
    # --- R_sakuga: Dynamic sakuga + motion (normalized) ---
    # Combine sakuga score with motion magnitude if available
    if motion is not None and len(motion) > 0:
        # Normalize motion to [0, 1]
        motion_valid = motion[:len(attrs_all)]
        motion_min = float(motion_valid.min())
        motion_max = float(motion_valid.max())
        motion_range = motion_max - motion_min
        
        if motion_range > 1e-8:
            motion_norm = (motion_valid - motion_min) / motion_range
        else:
            motion_norm = np.zeros_like(motion_valid)
        
        # Combined score: 50% sakuga + 50% motion
        sakuga_combined_all = 0.5 * sakuga_all + 0.5 * motion_norm
        sakuga_combined_sel = sakuga_combined_all[sel_idx_valid]
        
        sakuga_mean_all = float(sakuga_combined_all.mean())
        sakuga_std_all = float(sakuga_combined_all.std())
        sakuga_mean_sel = float(sakuga_combined_sel.mean())
    else:
        # Motion not available, use pure sakuga score
        sakuga_mean_all = float(sakuga_all.mean())
        sakuga_std_all = float(sakuga_all.std())
        sakuga_mean_sel = float(sakuga_sel.mean())
    
    R_sakuga = (sakuga_mean_sel - sakuga_mean_all) / (sakuga_std_all + 1e-6)
    
    # --- R_story: Narrative beat coverage (fraction of selected frames above threshold) ---
    # Beat = combination of sakuga + cinematic scores
    beat_all = 0.5 * sakuga_all + 0.5 * cinema_all
    beat_mean = float(beat_all.mean())
    beat_std = float(beat_all.std())
    
    # Threshold: mean + 0.5 * std (highlights important beats)
    beat_threshold = beat_mean + 0.5 * beat_std
    
    # Coverage: fraction of selected frames that exceed threshold
    beat_sel = beat_all[sel_idx_valid]
    coverage = float((beat_sel > beat_threshold).mean()) if len(beat_sel) > 0 else 0.0
    R_story = coverage
    
    return {
        "look": float(R_look),
        "sakuga": float(R_sakuga),
        "story": float(R_story),
    }

def prob_separation_reward(probs: np.ndarray, sel_idx: List[int]) -> float:
    """
    Reward for separating probabilities of selected vs non-selected frames.
    probs: (T,) or (1, T)
    """
    probs = probs.flatten()
    T = len(probs)
    if len(sel_idx) == 0 or len(sel_idx) == T:
        return 0.0
    
    sel_probs = probs[sel_idx]
    rest_idx = [i for i in range(T) if i not in sel_idx]
    rest_probs = probs[rest_idx]
    
    # We want mean prob of keyframes to be higher than non-keyframes
    return float(sel_probs.mean() - rest_probs.mean())

def reward_combo(
    feats_all: np.ndarray,             # (T,D) normalized
    sel_idx: List[int],                # indices
    frames_all: Optional[List[np.ndarray]] = None,
    motion: Optional[np.ndarray] = None,
    anime_scores: Optional[np.ndarray] = None, # (T, K)
    probs: Optional[np.ndarray] = None,        # (T,)
    w_div: float = 1.0,
    w_rep: float = 1.0,
    w_rec: float = 0.0,
    w_fd: float = 0.0,
    w_ms: float = 0.0,
    w_motion: float = 0.0,
    w_anime_look: float = 0.0,
    w_anime_sakuga: float = 0.0,
    w_anime_story: float = 0.0,
    w_probsep: float = 0.0,
    ms_swd_scales: int = 3,
    ms_swd_dirs: int = 16,
    use_lpips_div: bool = False,
    lpips_net: str = "alex",
    lpips_device: str = "cuda",
    reward_stats: Optional[Dict[str, float]] = None,
    return_components: bool = False,
) -> float:
    if len(sel_idx) == 0:
        return 0.0
    feats_sel = feats_all[sel_idx]
    if use_lpips_div and frames_all is not None:
        frames_sel = [frames_all[i] for i in sel_idx]
        R_div = diversity_reward_lpips(frames_sel, lpips_net=lpips_net, device=lpips_device)
    else:
        R_div = diversity_reward(feats_sel)
    R_rep = representativeness_reward(feats_all, feats_sel)
    R_rec = -reconstruction_error(feats_all, feats_sel) if w_rec!=0 else 0.0
    R_fd  = -frechet_distance(feats_all, feats_sel) if (w_fd!=0 and feats_all.shape[0]>=2 and feats_sel.shape[0]>=2) else 0.0
    R_ms  = 0.0
    if w_ms!=0 and frames_all is not None:
        frames_sel = [frames_all[i] for i in sel_idx]
        ms = ms_swd_color(frames_all, frames_sel, num_scales=ms_swd_scales, num_dirs=ms_swd_dirs)
        if not np.isnan(ms):
            R_ms = -float(ms)  # lower is better

    R_mot = 0.0
    if w_motion!=0 and motion is not None:
        # simple average motion on selected indices
        R_mot = float(np.mean(motion[sel_idx]))

    # Anime rewards (unified look-sakuga-story triad)
    R_anime_look = 0.0
    R_anime_sakuga = 0.0
    R_anime_story = 0.0
    if anime_scores is not None and (w_anime_look != 0.0 or w_anime_sakuga != 0.0 or w_anime_story != 0.0):
        anime_terms = anime_reward(anime_scores, sel_idx, motion=motion)
        R_anime_look = anime_terms["look"]
        R_anime_sakuga = anime_terms["sakuga"]
        R_anime_story = anime_terms["story"]

    R_probsep = 0.0
    if w_probsep != 0.0 and probs is not None:
        R_probsep = prob_separation_reward(probs, sel_idx)

    # Combine all reward components
    components = {
        "div": R_div,
        "rep": R_rep,
        "rec": R_rec,
        "fd": R_fd,
        "ms": R_ms,
        "motion": R_mot,
        "anime_look": R_anime_look,
        "anime_sakuga": R_anime_sakuga,
        "anime_story": R_anime_story,
        "probsep": R_probsep,
    }
    
    # Apply normalization if reward_stats provided
    if reward_stats is not None:
        normalized_components = {}
        for key, val in components.items():
            std_key = f"{key}_std"
            sigma = reward_stats.get(std_key, 1.0)
            normalized_components[key] = val / (sigma + 1e-6)
        components_for_sum = normalized_components
    else:
        components_for_sum = components
    
    # Weighted sum
    R = (
        w_div * components_for_sum["div"]
        + w_rep * components_for_sum["rep"]
        + w_rec * components_for_sum["rec"]
        + w_fd * components_for_sum["fd"]
        + w_ms * components_for_sum["ms"]
        + w_motion * components_for_sum["motion"]
        + w_anime_look * components_for_sum["anime_look"]
        + w_anime_sakuga * components_for_sum["anime_sakuga"]
        + w_anime_story * components_for_sum["anime_story"]
        + w_probsep * components_for_sum["probsep"]
    )
    
    if return_components:
        return float(R), components
    return float(R)
