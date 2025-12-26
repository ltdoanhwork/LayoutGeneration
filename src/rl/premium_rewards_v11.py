#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Premium Rewards V11: Quality + Temporal Diversity

Key features:
1. Extends V9 reward structure (keeps 2-head architecture)
2. Adds temporal diversity penalty (no clustering)
3. DPP-based diversity reward
4. Combined local + global optimization signal
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, NamedTuple
import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import det, inv
import warnings

# Import V9 base
from src.rl.premium_rewards_v9 import (
    PremiumRewardV9, 
    ConstraintConfig, 
    DPPConfig, 
    TestTimeScalingConfig,
    ConstraintTracker,
    ATTR_NAMES,
    ATTR_INDEX,
)


@dataclass
class DiversityConfig:
    """Configuration for temporal diversity."""
    min_gap_ratio: float = 0.3      # Expected gap as ratio of (T/budget)
    gap_penalty_weight: float = 2.0  # Weight for gap penalty
    dpp_diversity_weight: float = 1.0  # Weight for DPP diversity reward
    

class PremiumRewardV11(PremiumRewardV9):
    """
    V11 Reward System: Quality + Temporal Diversity
    
    Extends V9 with:
    - Temporal gap penalty for clustered selections
    - DPP-based diversity reward
    - Position-aware scoring (using rel_positions)
    """
    
    def __init__(
        self,
        constraint_config: Optional[ConstraintConfig] = None,
        dpp_config: Optional[DPPConfig] = None,
        tts_config: Optional[TestTimeScalingConfig] = None,
        diversity_config: Optional[DiversityConfig] = None,
        **kwargs
    ):
        super().__init__(
            constraint_config=constraint_config,
            dpp_config=dpp_config,
            tts_config=tts_config,
            **kwargs
        )
        self.diversity_config = diversity_config or DiversityConfig()
        
        # Track diversity metrics
        self.diversity_tracker = {
            "min_gap_history": [],
            "mean_gap_history": [],
            "dpp_score_history": [],
        }
    
    def compute_temporal_diversity(
        self,
        sel_idx: List[int],
        T: int,
        budget: int,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute temporal diversity metrics.
        
        Args:
            sel_idx: Selected frame indices
            T: Total frames
            budget: Number of frames to select
            
        Returns:
            diversity_reward: float
            info: dict
        """
        if len(sel_idx) < 2:
            return 0.0, {"min_gap": 0.0, "mean_gap": 0.0}
        
        sorted_idx = sorted(sel_idx)
        gaps = np.diff(sorted_idx)
        
        min_gap = float(np.min(gaps))
        mean_gap = float(np.mean(gaps))
        
        # Expected gap if perfectly distributed
        expected_gap = T / (budget + 1)
        min_expected = expected_gap * self.diversity_config.min_gap_ratio
        
        # Penalty for gaps below minimum
        gap_violations = sum(max(0, min_expected - g) for g in gaps)
        gap_penalty = gap_violations / max(1, len(gaps)) * self.diversity_config.gap_penalty_weight
        
        # Bonus for well-distributed selection
        coverage_score = min(1.0, mean_gap / expected_gap)
        
        diversity_reward = coverage_score - gap_penalty
        
        info = {
            "min_gap": min_gap,
            "mean_gap": mean_gap,
            "expected_gap": expected_gap,
            "gap_penalty": gap_penalty,
            "coverage_score": coverage_score,
        }
        
        return diversity_reward, info
    
    def compute_dpp_diversity_score(
        self,
        feats_all: np.ndarray,
        sel_idx: List[int],
    ) -> float:
        """
        Compute DPP-based diversity score (log-determinant).
        
        Higher value = more diverse selection.
        """
        if len(sel_idx) < 2:
            return 0.0
        
        try:
            feats_sel = feats_all[sel_idx]
            
            # Similarity kernel
            K = feats_sel @ feats_sel.T
            
            # Add small regularization for numerical stability
            K = K + 1e-4 * np.eye(len(sel_idx))
            
            # Log determinant (diversity measure)
            sign, logdet = np.linalg.slogdet(K)
            if sign <= 0:
                return 0.0
            
            # Normalize by selection size
            return logdet / len(sel_idx)
            
        except Exception:
            return 0.0
    
    def compute_reward(
        self,
        feats_all: np.ndarray,
        sel_idx: List[int],
        anime_attrs: Optional[np.ndarray] = None,
        rel_positions: Optional[np.ndarray] = None,  # V11: relative positions
        current_epoch: int = 0,
        update_lagrangian: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute V11 reward: V9 rewards + diversity rewards.
        
        Args:
            feats_all: (T, D) features
            sel_idx: Selected indices
            anime_attrs: (T, 6) quality attributes
            rel_positions: (T,) relative positions in [0, 1]
            current_epoch: Current training epoch
            update_lagrangian: Whether to update Lagrangian multipliers
            
        Returns:
            rewards: Dict with total, anime, constraint_penalty, diversity
            components: Dict with detailed metrics
        """
        T = len(feats_all)
        budget = len(sel_idx)
        
        # 1. Get V9 base rewards (quality + constraints)
        rewards, components = super().compute_reward(
            feats_all=feats_all,
            sel_idx=sel_idx,
            anime_attrs=anime_attrs,
            current_epoch=current_epoch,
            update_lagrangian=update_lagrangian
        )
        
        # 2. Temporal diversity
        div_reward, div_info = self.compute_temporal_diversity(sel_idx, T, budget)
        rewards["diversity"] = div_reward
        components.update({f"div_{k}": v for k, v in div_info.items()})
        
        # 3. DPP diversity (feature-based)
        dpp_score = self.compute_dpp_diversity_score(feats_all, sel_idx)
        rewards["dpp"] = dpp_score * self.diversity_config.dpp_diversity_weight
        components["dpp_score"] = dpp_score
        
        # 4. Position coverage bonus (if rel_positions available)
        if rel_positions is not None and len(sel_idx) > 0:
            sel_positions = rel_positions[sel_idx]
            # Good coverage = positions spread from 0 to 1
            pos_range = float(np.max(sel_positions) - np.min(sel_positions))
            rewards["position_coverage"] = pos_range
            components["position_range"] = pos_range
        
        # 5. Update total reward
        rewards["total"] = (
            rewards.get("anime", 0) + 
            rewards.get("percentile", 0) -
            rewards.get("constraint_penalty", 0) +
            rewards.get("diversity", 0) +
            rewards.get("dpp", 0) +
            rewards.get("position_coverage", 0)
        )
        
        # Track diversity metrics
        self.diversity_tracker["min_gap_history"].append(div_info.get("min_gap", 0))
        self.diversity_tracker["mean_gap_history"].append(div_info.get("mean_gap", 0))
        self.diversity_tracker["dpp_score_history"].append(dpp_score)
        
        return rewards, components
    
    def get_diversity_stats(self, last_n: int = 100) -> Dict[str, float]:
        """Get summary of diversity metrics."""
        def safe_mean(lst):
            if not lst:
                return 0.0
            return float(np.mean(lst[-last_n:]))
        
        return {
            "min_gap_mean": safe_mean(self.diversity_tracker["min_gap_history"]),
            "mean_gap_mean": safe_mean(self.diversity_tracker["mean_gap_history"]),
            "dpp_score_mean": safe_mean(self.diversity_tracker["dpp_score_history"]),
        }
