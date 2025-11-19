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

def reward_combo(
    feats_all: np.ndarray,             # (T,D) normalized
    sel_idx: List[int],                # indices
    frames_all: Optional[List[np.ndarray]] = None,
    motion: Optional[np.ndarray] = None,
    w_div: float = 1.0,
    w_rep: float = 1.0,
    w_rec: float = 0.0,
    w_fd: float = 0.0,
    w_ms: float = 0.0,
    w_motion: float = 0.0,
    ms_swd_scales: int = 3,
    ms_swd_dirs: int = 16,
    use_lpips_div: bool = False,
    lpips_net: str = "alex",
    lpips_device: str = "cuda",
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

    R = ( w_div*R_div + w_rep*R_rep + w_rec*R_rec + w_fd*R_fd + w_ms*R_ms + w_motion*R_mot )
    return float(R)
