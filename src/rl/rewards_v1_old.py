"""
OLD REWARD SYSTEM (v1) - BACKUP FOR COMPARISON
This file preserves the original implementation with double-counting issue.
DO NOT USE IN PRODUCTION - for comparison purposes only.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np

from eval.metrics import *
from src.distance_selector.registry import create_metric

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
    tensors = [metric.preprocess_bgr(fr) for fr in frames_sel]

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

def anime_iqa_emphasis_v1(q_all: np.ndarray,
                       sel_idx: List[int],
                       important_ids=(1, 3, 4, 5),
                       lambda_match=0.1) -> float:
    """
    OLD IMPLEMENTATION - Reward based on Anime-CLIP IQA scores.
    This gets double-counted with Track B!
    """
    if len(sel_idx) == 0:
        return 0.0
    q_sel = q_all[sel_idx]
    mu_all = q_all.mean(axis=0)
    mu_sel = q_sel.mean(axis=0)

    # 1) match entire distribution (lightly)
    diff = mu_sel - mu_all
    R_match = -float(np.mean(diff**2))

    # 2) boost important dimensions
    boost = (mu_sel[list(important_ids)] - mu_all[list(important_ids)]).mean()
    
    return lambda_match * R_match + float(boost)

def prob_separation_reward(probs: np.ndarray, sel_idx: List[int]) -> float:
    """
    Reward for separating probabilities of selected vs non-selected frames.
    """
    probs = probs.flatten()
    T = len(probs)
    if len(sel_idx) == 0 or len(sel_idx) == T:
        return 0.0
    
    sel_probs = probs[sel_idx]
    rest_idx = [i for i in range(T) if i not in sel_idx]
    rest_probs = probs[rest_idx]
    
    return float(sel_probs.mean() - rest_probs.mean())

def reward_combo_v1(
    feats_all: np.ndarray,
    sel_idx: List[int],
    frames_all: Optional[List[np.ndarray]] = None,
    motion: Optional[np.ndarray] = None,
    anime_scores: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    w_div: float = 1.0,
    w_rep: float = 1.0,
    w_rec: float = 0.0,
    w_fd: float = 0.0,
    w_ms: float = 0.0,
    w_motion: float = 0.0,
    w_anime: float = 0.0,  # OLD PARAMETER - causes double counting!
    w_probsep: float = 0.0,
    ms_swd_scales: int = 3,
    ms_swd_dirs: int = 16,
    use_lpips_div: bool = False,
    lpips_net: str = "alex",
    lpips_device: str = "cuda",
) -> float:
    """
    OLD reward_combo with double-counting issue.
    anime_iqa_emphasis gets called here, then Track B adds R_look/R_sakuga/R_story again!
    """
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
            R_ms = -float(ms)

    R_mot = 0.0
    if w_motion!=0 and motion is not None:
        R_mot = float(np.mean(motion[sel_idx]))

    # DOUBLE-COUNTING ISSUE: This gets added here...
    R_anime = 0.0
    if w_anime != 0.0 and anime_scores is not None:
        R_anime = anime_iqa_emphasis_v1(anime_scores, sel_idx)

    R_probsep = 0.0
    if w_probsep != 0.0 and probs is not None:
        R_probsep = prob_separation_reward(probs, sel_idx)

    # ...and then Track B adds R_look + R_sakuga + R_story from same data!
    R = ( w_div*R_div + w_rep*R_rep + w_rec*R_rec + w_fd*R_fd + w_ms*R_ms + w_motion*R_mot 
          + w_anime*R_anime + w_probsep*R_probsep )
    return float(R)
