#!/usr/bin/env python3
"""
Enhanced Reward Functions (Version 3)

This module extends the base reward system with advanced normalization,
variance-aware weighting, and additional reward components for V3.

New Features:
- Reward component normalization using running statistics
- Temporal smoothness rewards
- Perceptual diversity (LPIPS-based)
- Adaptive multi-objective weighting

Author: Version 3 Enhancement
Date: 2025-12-06
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from src.rl.rewards import (
    diversity_reward,
    representativeness_reward,
    cosine_dist_matrix,
    diversity_reward_lpips
)


class RewardNormalizer:
    """
    Reward normalization using running mean/std statistics.
    
    Prevents reward scale imbalances during training by normalizing
    each reward component to zero mean and unit variance.
    """
    
    def __init__(self, components: List[str]):
        """
        Initialize normalizer for specified reward components.
        
        Args:
            components: List of reward component names to track
        """
        self.components = components
        self.stats = {name: {"mean": 0.0, "std": 1.0, "n": 0, "m2": 0.0} 
                     for name in components}
    
    def update(self, rewards: Dict[str, float]):
        """
        Update statistics with new reward values (Welford's algorithm).
        
        Args:
            rewards: Dict mapping component names to reward values
        """
        for name, value in rewards.items():
            if name not in self.stats:
                continue
            
            s = self.stats[name]
            s["n"] += 1
            delta = value - s["mean"]
            s["mean"] += delta / s["n"]
            delta2 = value - s["mean"]
            s["m2"] += delta * delta2
            
            # Update std
            if s["n"] >= 2:
                variance = s["m2"] / (s["n"] - 1)
                s["std"] = max(np.sqrt(variance), 1e-6)  # Avoid division by zero
    
    def normalize(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize reward components to zero mean, unit variance.
        
        Args:
            rewards: Raw reward values
        
        Returns:
            Normalized reward values
        """
        normalized = {}
        for name, value in rewards.items():
            if name in self.stats:
                s = self.stats[name]
                if s["n"] >= 10:  # Only normalize after sufficient samples
                    normalized[name] = (value - s["mean"]) / (s["std"] + 1e-8)
                else:
                    normalized[name] = value  # Not enough data yet
            else:
                normalized[name] = value
        
        return normalized
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics for all components."""
        return {name: {"mean": s["mean"], "std": s["std"], "n": s["n"]} 
                for name, s in self.stats.items()}


def temporal_smoothness_reward(
    sel_idx: List[int],
    total_frames: int,
    penalty_scale: float = 1.0
) -> float:
    """
    Reward for temporally smooth frame selections.
    
    Encourages evenly-distributed selections over time, penalizes
    clustering or large gaps.
    
    Args:
        sel_idx: Selected frame indices (will be sorted)
        total_frames: Total number of frames
        penalty_scale: Scaling factor for penalty
    
    Returns:
        Smoothness reward (higher is better)
    """
    if len(sel_idx) < 2:
        return 0.0
    
    sel_sorted = sorted(sel_idx)
    gaps = np.diff(sel_sorted)
    
    # Ideal gap: uniform spacing
    ideal_gap = total_frames / len(sel_idx)
    
    # Compute deviation from ideal
    gap_std = np.std(gaps)
    
    # Reward lower variance (smoother spacing)
    # Use exponential decay: exp(-std / scale)
    reward = np.exp(-gap_std / (ideal_gap * penalty_scale))
    
    return float(reward)


def quality_variance_reward(
    quality_scores_all: np.ndarray,
    sel_idx: List[int],
    reward_scale: float = 1.0
) -> float:
    """
    Reward for high variance in selected frame quality.
    
    Encourages selecting frames with diverse quality characteristics
    (e.g., both dramatic peaks and stable moments).
    
    Args:
        quality_scores_all: Quality scores for all frames (T,)
        sel_idx: Selected indices
        reward_scale: Scaling factor
    
    Returns:
        Variance reward
    """
    if len(sel_idx) < 2:
        return 0.0
    
    sel_quality = quality_scores_all[sel_idx]
    quality_std = np.std(sel_quality)
    
    # Normalize by overall variance
    overall_std = np.std(quality_scores_all)
    if overall_std < 1e-6:
        return 0.0
    
    # Reward relative variance
    relative_var = quality_std / (overall_std + 1e-6)
    reward = reward_scale * relative_var
    
    return float(reward)


def perceptual_diversity_reward(
    frames_all: List[np.ndarray],
    sel_idx: List[int],
    lpips_net: str = "alex",
    device: str = "cuda",
    max_pairs: int = 1000
) -> float:
    """
    LPIPS-based perceptual diversity reward.
    
    More accurate than cosine similarity for visual diversity.
    
    Args:
        frames_all: All frames (list of HxWx3 BGR numpy arrays)
        sel_idx: Selected indices
        lpips_net: LPIPS network ('alex', 'vgg', or 'squeeze')
        device: Device for LPIPS computation
        max_pairs: Maximum number of pairs to evaluate
    
    Returns:
        Perceptual diversity score
    """
    if len(sel_idx) < 2:
        return 0.0
    
    frames_sel = [frames_all[i] for i in sel_idx if i < len(frames_all)]
    
    if len(frames_sel) < 2:
        return 0.0
    
    # Use existing LPIPS diversity function
    diversity = diversity_reward_lpips(
        frames_sel, 
        lpips_net=lpips_net, 
        device=device, 
        max_pairs=max_pairs
    )
    
    return diversity


def adaptive_pareto_weights(
    reward_components: Dict[str, float],
    target_ratios: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute adaptive weights for multi-objective optimization.
    
    Uses Pareto-optimal weighting to balance conflicting objectives
    (e.g., diversity vs quality).
    
    Args:
        reward_components: Current reward values (unnormalized)
        target_ratios: Optional target ratios for each component
    
    Returns:
        Dict of adaptive weights
    """
    if target_ratios is None:
        # Default: equal importance
        target_ratios = {k: 1.0 for k in reward_components.keys()}
    
    # Compute inverse magnitudes (give more weight to smaller components)
    weights = {}
    for key in reward_components.keys():
        if key in target_ratios:
            magnitude = abs(reward_components[key]) + 1e-6
            weights[key] = target_ratios[key] / magnitude
        else:
            weights[key] = 1.0
    
    # Normalize weights to sum to number of components
    total_weight = sum(weights.values())
    n_components = len(weights)
    
    if total_weight > 1e-6:
        weights = {k: (v / total_weight) * n_components for k, v in weights.items()}
    
    return weights


def reward_combo_v3(
    feats_all: np.ndarray,
    sel_idx: List[int],
    frames_all: Optional[List[np.ndarray]] = None,
    anime_scores: Optional[np.ndarray] = None,
    motion: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    # Standard weights
    w_div: float = 1.0,
    w_rep: float = 1.0,
    w_rec: float = 0.0,
    w_fd: float = 0.0,
    w_ms: float = 0.0,
    w_motion: float = 0.0,
    w_probsep: float = 0.0,
    # V3 enhancements
    w_temporal: float = 0.5,
    w_quality_var: float = 0.2,
    use_perceptual_div: bool = False,
    use_reward_norm: bool = False,
    normalizer: Optional[RewardNormalizer] = None,
    # Advanced options
    lpips_net: str = "alex",
    lpips_device: str = "cuda",
    return_components: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Enhanced reward computation with V3 features.
    
    New Features:
    - Temporal smoothness reward
    - Quality variance reward
    - Optional perceptual diversity (LPIPS)
    - Reward normalization support
    
    Args:
        ... (see reward_combo in rewards.py for base args)
        w_temporal: Weight for temporal smoothness
        w_quality_var: Weight for quality variance
        use_perceptual_div: Use LPIPS for diversity instead of cosine
        use_reward_norm: Apply reward normalization
        normalizer: RewardNormalizer instance (if use_reward_norm=True)
        return_components: Return (total, components_dict) tuple
    
    Returns:
        Total reward or (total, components_dict) if return_components=True
    """
    from src.rl.rewards import reward_combo
    
    # Compute base rewards using existing function
    R_base, components_base = reward_combo(
        feats_all=feats_all,
        sel_idx=sel_idx,
        frames_all=frames_all,
        anime_scores=anime_scores,
        motion=motion,
        probs=probs,
        w_div=w_div,
        w_rep=w_rep,
        w_rec=w_rec,
        w_fd=w_fd,
        w_ms=w_ms,
        w_motion=w_motion,
        w_anime_look=0.0,  # Handled separately in V3
        w_anime_sakuga=0.0,
        w_anime_story=0.0,
        w_probsep=w_probsep,
        use_lpips_div=use_perceptual_div,
        lpips_net=lpips_net,
        lpips_device=lpips_device,
        return_components=True
    )
    
    # Add V3-specific rewards
    components = dict(components_base)
    
    # Temporal smoothness
    if w_temporal > 0:
        R_temp = temporal_smoothness_reward(sel_idx, len(feats_all))
        components["temporal"] = R_temp
    else:
        components["temporal"] = 0.0
    
    # Quality variance (if anime_scores provided)
    if w_quality_var > 0 and anime_scores is not None:
        # Use mean quality as proxy
        quality_all = anime_scores[:, :3].mean(axis=1)  # Average of sharp/color/bright
        R_qvar = quality_variance_reward(quality_all, sel_idx)
        components["quality_var"] = R_qvar
    else:
        components["quality_var"] = 0.0
    
    # Apply normalization if requested
    if use_reward_norm and normalizer is not None:
        # Update normalizer stats
        normalizer.update(components)
        # Normalize components
        components = normalizer.normalize(components)
    
    # Compute total reward
    R_total = (
        components.get("div", 0.0) * w_div +
        components.get("rep", 0.0) * w_rep +
        components.get("rec", 0.0) * w_rec +
        components.get("fd", 0.0) * w_fd +
        components.get("ms", 0.0) * w_ms +
        components.get("motion", 0.0) * w_motion +
        components.get("probsep", 0.0) * w_probsep +
        components.get("temporal", 0.0) * w_temporal +
        components.get("quality_var", 0.0) * w_quality_var
    )
    
    if return_components:
        return float(R_total), components
    return float(R_total)


if __name__ == "__main__":
    # Demo
    print("=== Reward V3 Enhancements Demo ===\n")
    
    # Test reward normalizer
    normalizer = RewardNormalizer(["div", "rep", "temporal"])
    
    # Simulate training
    for i in range(20):
        rewards = {
            "div": np.random.randn() * 0.5 + 1.0,
            "rep": np.random.randn() * 2.0 - 0.5,
            "temporal": np.random.rand()
        }
        normalizer.update(rewards)
        
        if i >= 10:  # After warmup
            norm_rewards = normalizer.normalize(rewards)
            if i == 15:
                print(f"Step {i}:")
                print(f"  Raw: {rewards}")
                print(f"  Normalized: {norm_rewards}")
    
    print(f"\nFinal Stats: {normalizer.get_stats()}\n")
    
    # Test temporal smoothness
    print("Temporal Smoothness:")
    good_selection = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # Evenly spaced
    bad_selection = [0, 1, 2, 50, 51, 52, 90, 91, 92, 93]  # Clustered
    
    R_good = temporal_smoothness_reward(good_selection, 100)
    R_bad = temporal_smoothness_reward(bad_selection, 100)
    
    print(f"  Even spacing: {R_good:.3f}")
    print(f"  Clustered: {R_bad:.3f}")
