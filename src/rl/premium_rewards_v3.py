#!/usr/bin/env python3
"""
Premium Anime Reward System (Version 3)

This module implements state-of-the-art reward mechanisms for anime video summarization,
incorporating:
1. Adaptive percentile-based rewards with quality calibration
2. Multi-stage curriculum learning (3 training phases)
3. Contrastive learning with hard negative mining
4. Temporal consistency rewards
5. Reward normalization using running statistics

Key Improvements over V1:
- Adaptive thresholds based on video quality distribution
- More sophisticated curriculum with 3 distinct phases
- Hard negative mining to penalize near-miss selections
- Temporal smoothness reward for narrative flow
- Quality calibration across videos of different base quality

Author: Version 3 Enhancement
Date: 2025-12-06
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np


class RunningStats:
    """
    Efficient running statistics tracker for reward normalization.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from current mean
        
    def update(self, value: float):
        """Update statistics with new value."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2
    
    def get_mean(self) -> float:
        """Get current mean."""
        return self.mean if self.n > 0 else 0.0
    
    def get_std(self) -> float:
        """Get current standard deviation."""
        if self.n < 2:
            return 1.0  # Return 1.0 to avoid division by zero
        variance = self.m2 / (self.n - 1)
        return np.sqrt(variance)
    
    def get_stats(self) -> Tuple[float, float]:
        """Get (mean, std) tuple."""
        return self.get_mean(), self.get_std()


class PremiumAnimeRewardV3:
    """
    Premium Anime Reward System Version 3.
    
    Features:
    - Adaptive percentile thresholds based on video quality
    - Multi-stage curriculum (foundation → aesthetic → narrative)
    - Hard negative mining for sharper quality boundaries
    - Temporal consistency rewards
    - Cross-video quality calibration
    
    Usage:
        >>> reward_system = PremiumAnimeRewardV3(total_epochs=20)
        >>> # Get curriculum weights for current epoch
        >>> weights = reward_system.get_curriculum_weights(epoch=10)
        >>> # Compute rewards
        >>> rewards = reward_system.compute_reward(attrs_all, sel_idx, current_epoch=10)
    """
    
    def __init__(
        self,
        percentile_threshold: float = 0.75,
        contrastive_margin: float = 0.15,
        hard_negative_margin: float = 0.05,
        temporal_weight: float = 0.5,
        use_curriculum: bool = True,
        use_quality_calibration: bool = True,
        total_epochs: int = 20
    ):
        """
        Initialize Premium Anime Reward System V3.
        
        Args:
            percentile_threshold: Base percentile for quality threshold (0.0-1.0)
            contrastive_margin: Minimum gap between selected and mean for bonus
            hard_negative_margin: Margin for hard negative mining
            temporal_weight: Weight for temporal consistency reward
            use_curriculum: Enable multi-stage curriculum learning
            use_quality_calibration: Calibrate rewards across videos
            total_epochs: Total training epochs for curriculum scheduling
        """
        self.percentile_threshold = percentile_threshold
        self.contrastive_margin = contrastive_margin
        self.hard_negative_margin = hard_negative_margin
        self.temporal_weight = temporal_weight
        self.use_curriculum = use_curriculum
        self.use_quality_calibration = use_quality_calibration
        self.total_epochs = total_epochs
        
        # Running statistics for quality calibration
        self.look_stats = RunningStats()
        self.sakuga_stats = RunningStats()
        self.story_stats = RunningStats()
        
        # Curriculum stages (proportion of total epochs)
        self.stage1_end = int(0.25 * total_epochs)  # Epochs 1-5: Foundation
        self.stage2_end = int(0.60 * total_epochs)  # Epochs 6-12: Aesthetic
        # Stage 3: Epochs 13+: Full optimization
        
    def get_adaptive_percentile(
        self, 
        scores: np.ndarray, 
        base_percentile: float = 0.75
    ) -> float:
        """
        Compute adaptive percentile threshold based on score distribution.
        
        For videos with high variance, we can be more selective (higher percentile).
        For videos with low variance, we relax the threshold slightly.
        
        Args:
            scores: Quality scores for all frames
            base_percentile: Base percentile value
        
        Returns:
            Adjusted percentile value
        """
        if len(scores) < 2:
            return base_percentile
        
        # Compute coefficient of variation (CV = std / mean)
        mean_score = float(scores.mean())
        std_score = float(scores.std())
        cv = std_score / (abs(mean_score) + 1e-6)
        
        # High variance (CV > 0.3): Be more selective (increase percentile)
        # Low variance (CV < 0.1): Relax threshold (decrease percentile)
        if cv > 0.3:
            adjustment = 0.05
        elif cv < 0.1:
            adjustment = -0.05
        else:
            adjustment = 0.0
        
        adjusted = np.clip(base_percentile + adjustment, 0.5, 0.95)
        return adjusted
    
    def get_curriculum_weights(self, current_epoch: int) -> Dict[str, float]:
        """
        Get curriculum weights for current epoch based on 3-stage curriculum.
        
        Stage 1 (Epochs 1-5): Foundation
          - Focus on basic summarization (diversity, representativeness)
          - Low aesthetic weights (10% of final)
          
        Stage 2 (Epochs 6-12): Aesthetic Quality
          - Ramp up Look and Sakuga rewards
          - Medium weights (10% → 100% linearly)
          
        Stage 3 (Epochs 13+): Full Optimization
          - All rewards at full strength
          - Add temporal consistency
        
        Args:
            current_epoch: Current training epoch (1-indexed)
        
        Returns:
            Dict with scaling factors for each reward component
        """
        if not self.use_curriculum:
            return {
                "look": 1.0,
                "sakuga": 1.0,
                "story": 1.0,
                "temporal": 1.0,
            }
        
        weights = {}
        
        # Stage 1: Foundation (low aesthetic focus)
        if current_epoch <= self.stage1_end:
            weights["look"] = 0.1
            weights["sakuga"] = 0.1
            weights["story"] = 0.05
            weights["temporal"] = 0.0
        
        # Stage 2: Aesthetic Rampup
        elif current_epoch <= self.stage2_end:
            progress = (current_epoch - self.stage1_end) / (self.stage2_end - self.stage1_end)
            weights["look"] = 0.1 + 0.9 * progress
            weights["sakuga"] = 0.1 + 0.9 * progress
            weights["story"] = 0.05 + 0.45 * progress  # Story grows slower
            weights["temporal"] = 0.0
        
        # Stage 3: Full Optimization
        else:
            weights["look"] = 1.0
            weights["sakuga"] = 1.0
            weights["story"] = 1.0
            weights["temporal"] = 1.0
        
        return weights
    
    def compute_hard_negative_penalty(
        self,
        selected_scores: np.ndarray,
        threshold: float,
        margin: float
    ) -> float:
        """
        Compute penalty for selecting frames just below the threshold.
        
        Hard negative mining: frames in range [threshold - margin, threshold)
        are penalized more heavily to sharpen the quality boundary.
        
        Args:
            selected_scores: Scores of selected frames
            threshold: Quality threshold
            margin: Margin for hard negatives
        
        Returns:
            Penalty value (negative)
        """
        if len(selected_scores) == 0:
            return 0.0
        
        # Find hard negatives: scores in [threshold - margin, threshold)
        hard_negatives = (selected_scores >= (threshold - margin)) & (selected_scores < threshold)
        num_hard_neg = hard_negatives.sum()
        
        if num_hard_neg == 0:
            return 0.0
        
        # Additional penalty for hard negatives (beyond standard penalty)
        penalty = -0.5 * (num_hard_neg / len(selected_scores))
        return float(penalty)
    
    def compute_temporal_consistency(
        self,
        sel_idx: List[int],
        total_frames: int
    ) -> float:
        """
        Compute temporal consistency reward.
        
        Rewards smooth, evenly-distributed selections over time.
        Penalizes clustered or jumpy selections.
        
        Args:
            sel_idx: Selected frame indices (assumed sorted)
            total_frames: Total number of frames in video
        
        Returns:
            Temporal consistency score [0, 1]
        """
        if len(sel_idx) < 2:
            return 0.0
        
        sel_idx_sorted = sorted(sel_idx)
        
        # Compute gaps between consecutive selections
        gaps = np.diff(sel_idx_sorted)
        
        # Ideal gap: evenly distributed
        ideal_gap = total_frames / len(sel_idx)
        
        # Measure deviation from ideal (lower is better)
        gap_variance = np.var(gaps) if len(gaps) > 0 else 0.0
        
        # Normalize to [0, 1], higher is better
        # Use exponential decay: exp(-variance / scale)
        scale = max(ideal_gap * 0.5, 1.0)
        consistency = np.exp(-gap_variance / scale)
        
        return float(consistency)
    
    def compute_reward(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
        current_epoch: int = 0,
        motion: Optional[np.ndarray] = None,
        indices_mapping: Optional[np.ndarray] = None,  # For ATTR_INDEX compatibility
    ) -> Dict[str, float]:
        """
        Compute premium rewards with all V3 enhancements.
        
        Args:
            attrs_all: (T, K) Anime-CLIP IQA scores for all frames
                       Expected columns: [sharpness, colorfulness, brightness, 
                                         sakuga, cinematic, expression]
            sel_idx: List of selected frame indices
            current_epoch: Current training epoch
            motion: Optional (T,) motion magnitudes for sakuga enhancement
            indices_mapping: Optional ATTR_INDEX dict for compatibility
        
        Returns:
            Dict with keys: "look", "sakuga", "story", "temporal", "total"
        """
        if len(sel_idx) == 0 or attrs_all.shape[0] == 0:
            return {
                "look": 0.0,
                "sakuga": 0.0,
                "story": 0.0,
                "temporal": 0.0,
                "total": 0.0
            }
        
        sel_idx_valid = [i for i in sel_idx if i < len(attrs_all)]
        if len(sel_idx_valid) == 0:
            return {
                "look": 0.0,
                "sakuga": 0.0,
                "story": 0.0,
                "temporal": 0.0,
                "total": 0.0
            }
        
        # Extract scores (assume standard order)
        sharp_all = attrs_all[:, 0]
        color_all = attrs_all[:, 1]
        bright_all = attrs_all[:, 2]
        sakuga_all = attrs_all[:, 3]
        cinema_all = attrs_all[:, 4]
        
        # Compute composite scores
        look_scores = (sharp_all + color_all + bright_all) / 3.0
        sakuga_scores = sakuga_all
        story_scores = (sakuga_all + cinema_all) / 2.0
        
        # --- Adaptive Percentile Thresholds ---
        look_percentile = self.get_adaptive_percentile(look_scores, self.percentile_threshold)
        sakuga_percentile = self.get_adaptive_percentile(sakuga_scores, self.percentile_threshold)
        story_percentile = self.get_adaptive_percentile(story_scores, self.percentile_threshold)
        
        look_thresh = float(np.quantile(look_scores, look_percentile))
        sakuga_thresh = float(np.quantile(sakuga_scores, sakuga_percentile))
        story_thresh = float(np.quantile(story_scores, story_percentile))
        
        # Selected scores
        sel_look = look_scores[sel_idx_valid]
        sel_sakuga = sakuga_scores[sel_idx_valid]
        sel_story = story_scores[sel_idx_valid]
        
        # --- Core Reward: Percentile-based ---
        def compute_component(selected, threshold):
            good = (selected >= threshold).sum()
            bad = (selected < threshold).sum()
            total = len(selected)
            if total == 0:
                return 0.0
            return float((good - bad) / total)
        
        R_look_base = compute_component(sel_look, look_thresh)
        R_sakuga_base = compute_component(sel_sakuga, sakuga_thresh)
        R_story_base = compute_component(sel_story, story_thresh)
        
        # --- Hard Negative Penalty ---
        R_look_hn = self.compute_hard_negative_penalty(sel_look, look_thresh, self.hard_negative_margin)
        R_sakuga_hn = self.compute_hard_negative_penalty(sel_sakuga, sakuga_thresh, self.hard_negative_margin)
        R_story_hn = self.compute_hard_negative_penalty(sel_story, story_thresh, self.hard_negative_margin)
        
        R_look = R_look_base + R_look_hn
        R_sakuga = R_sakuga_base + R_sakuga_hn
        R_story = R_story_base + R_story_hn
        
        # --- Contrastive Bonus ---
        if sel_look.mean() > look_scores.mean() + self.contrastive_margin:
            R_look += 0.3
        
        if sel_sakuga.mean() > sakuga_scores.mean() + self.contrastive_margin:
            R_sakuga += 0.3
        
        # --- Temporal Consistency ---
        R_temporal = self.compute_temporal_consistency(sel_idx_valid, len(attrs_all))
        
        # --- Quality Calibration (Normalization) ---
        if self.use_quality_calibration:
            # Update running stats
            self.look_stats.update(sel_look.mean())
            self.sakuga_stats.update(sel_sakuga.mean())
            self.story_stats.update(sel_story.mean())
            
            # Z-score normalization (if enough samples)
            if self.look_stats.n > 10:
                look_mean, look_std = self.look_stats.get_stats()
                sakuga_mean, sakuga_std = self.sakuga_stats.get_stats()
                
                # Normalize reward signals
                R_look = (R_look - 0.0) / max(look_std, 0.5)  # Center around 0
                R_sakuga = (R_sakuga - 0.0) / max(sakuga_std, 0.5)
        
        # --- Apply Curriculum Weights ---
        weights = self.get_curriculum_weights(current_epoch)
        
        R_look_weighted = R_look * weights["look"]
        R_sakuga_weighted = R_sakuga * weights["sakuga"]
        R_story_weighted = R_story * weights["story"]
        R_temporal_weighted = R_temporal * self.temporal_weight * weights["temporal"]
        
        # Total reward
        R_total = R_look_weighted + R_sakuga_weighted + R_story_weighted + R_temporal_weighted
        
        return {
            "look": float(R_look),
            "sakuga": float(R_sakuga),
            "story": float(R_story),
            "temporal": float(R_temporal),
            "look_weighted": float(R_look_weighted),
            "sakuga_weighted": float(R_sakuga_weighted),
            "story_weighted": float(R_story_weighted),
            "temporal_weighted": float(R_temporal_weighted),
            "total": float(R_total),
        }
    
    def get_curriculum_stage(self, current_epoch: int) -> str:
        """Get current curriculum stage name."""
        if current_epoch <= self.stage1_end:
            return "Foundation"
        elif current_epoch <= self.stage2_end:
            return "Aesthetic"
        else:
            return "Full Optimization"


# Backward compatibility alias
PremiumAnimeReward = PremiumAnimeRewardV3


if __name__ == "__main__":
    # Demo
    print("=== Premium Anime Reward V3 Demo ===\n")
    
    # Create dummy data
    np.random.seed(42)
    T = 100
    attrs = np.random.rand(T, 6).astype(np.float32)
    sel_idx = sorted(np.random.choice(T, size=8, replace=False).tolist())
    
    print(f"Video frames: {T}")
    print(f"Selected frames: {len(sel_idx)}")
    print(f"Selected indices: {sel_idx}\n")
    
    # Initialize reward system
    reward_sys = PremiumAnimeRewardV3(total_epochs=20)
    
    # Test across epochs
    for epoch in [1, 5, 10, 15, 20]:
        rewards = reward_sys.compute_reward(attrs, sel_idx, current_epoch=epoch)
        stage = reward_sys.get_curriculum_stage(epoch)
        weights = reward_sys.get_curriculum_weights(epoch)
        
        print(f"Epoch {epoch:2d} ({stage}):")
        print(f"  Weights: Look={weights['look']:.2f}, Sakuga={weights['sakuga']:.2f}, "
              f"Story={weights['story']:.2f}, Temporal={weights['temporal']:.2f}")
        print(f"  Rewards: Look={rewards['look']:.3f}, Sakuga={rewards['sakuga']:.3f}, "
              f"Story={rewards['story']:.3f}, Temporal={rewards['temporal']:.3f}")
        print(f"  Total Weighted: {rewards['total']:.3f}\n")
