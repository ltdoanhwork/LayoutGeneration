#!/usr/bin/env python3
"""
Premium Anime Reward System (Version 4)

Key Improvements over V3:
1. CONTINUOUS quality rewards instead of binary percentile-based
2. PER-ATTRIBUTE optimization for all 6 CLIP-IQA dimensions
3. Quality improvement metrics that MATCH evaluation metrics
4. Top-k coverage bonus for selecting highest quality frames
5. Miss penalty for NOT selecting high-quality frames

This version ensures training rewards ALIGN with evaluation metrics.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np


# Anime-CLIP-IQA attribute indices (matches prepare_anime_attrs.py)
ATTR_INDEX = {
    "sharpness": 0,
    "colorfulness": 1,
    "brightness": 2,
    "sakuga": 3,
    "cinematic": 4,
    "expression": 5,
}

ATTR_NAMES = ["sharpness", "colorfulness", "brightness", "sakuga", "cinematic", "expression"]


class QualityTracker:
    """
    Track quality improvement during training for logging.
    """
    def __init__(self):
        self.improvements = {name: [] for name in ATTR_NAMES}
        self.top_k_recalls = []
        self.top_k_precisions = []
    
    def update(self, improvements: Dict[str, float], top_k_recall: float, top_k_precision: float):
        for name, val in improvements.items():
            if name in self.improvements:
                self.improvements[name].append(val)
        self.top_k_recalls.append(top_k_recall)
        self.top_k_precisions.append(top_k_precision)
    
    def get_summary(self) -> Dict[str, float]:
        summary = {}
        for name, vals in self.improvements.items():
            if vals:
                summary[f"quality_improvement_{name}"] = float(np.mean(vals))
        if self.top_k_recalls:
            summary["top_k_recall_mean"] = float(np.mean(self.top_k_recalls))
        if self.top_k_precisions:
            summary["top_k_precision_mean"] = float(np.mean(self.top_k_precisions))
        return summary
    
    def reset(self):
        self.improvements = {name: [] for name in ATTR_NAMES}
        self.top_k_recalls = []
        self.top_k_precisions = []


class PremiumAnimeRewardV4:
    """
    Premium Anime Reward System Version 4 - Aligned with Evaluation Metrics.
    
    Key Design Principles:
    1. Rewards should DIRECTLY correspond to evaluation metrics
    2. Model should learn to maximize ABSOLUTE quality, not just relative
    3. Per-attribute rewards to avoid losing signal through aggregation
    4. Top-k coverage to ensure best frames are selected
    """
    
    def __init__(
        self,
        # Per-attribute weights (can tune to emphasize specific qualities)
        attr_weights: Optional[Dict[str, float]] = None,
        # Top-k settings
        top_k_ratio: float = 0.1,  # Consider top 10% as high-quality
        top_k_recall_weight: float = 1.0,  # Weight for recall of top-k frames
        top_k_precision_weight: float = 0.5,  # Weight for precision
        # Baseline reward settings
        improvement_scale: float = 3.0,  # Scale factor for quality improvement rewards
        use_continuous: bool = True,  # Use continuous vs binary rewards
        # Curriculum settings
        use_curriculum: bool = True,
        total_epochs: int = 60,
    ):
        # Default equal weights for all attributes
        self.attr_weights = attr_weights or {
            "sharpness": 1.0,
            "colorfulness": 1.0,
            "brightness": 0.5,  # Less important
            "sakuga": 2.0,  # More important for anime
            "cinematic": 1.5,
            "expression": 1.5,
        }
        
        self.top_k_ratio = top_k_ratio
        self.top_k_recall_weight = top_k_recall_weight
        self.top_k_precision_weight = top_k_precision_weight
        self.improvement_scale = improvement_scale
        self.use_continuous = use_continuous
        self.use_curriculum = use_curriculum
        self.total_epochs = total_epochs
        
        # Curriculum stages
        self.stage1_end = int(0.20 * total_epochs)  # Foundation
        self.stage2_end = int(0.50 * total_epochs)  # Quality focus
        # Stage 3: Full optimization
        
        # Tracker for logging
        self.tracker = QualityTracker()
    
    def get_curriculum_weights(self, current_epoch: int) -> Dict[str, float]:
        """
        Get curriculum weights based on training stage.
        
        Stage 1 (Foundation): Low quality weights, focus on basic selection
        Stage 2 (Quality): Ramp up quality rewards
        Stage 3 (Full): All rewards at full strength
        """
        if not self.use_curriculum:
            return {
                "quality": 1.0,
                "top_k": 1.0,
                "per_attr": 1.0,
            }
        
        if current_epoch <= self.stage1_end:
            # Stage 1: Foundation - low quality focus
            return {
                "quality": 0.2,
                "top_k": 0.1,
                "per_attr": 0.1,
            }
        elif current_epoch <= self.stage2_end:
            # Stage 2: Ramp up
            progress = (current_epoch - self.stage1_end) / (self.stage2_end - self.stage1_end)
            return {
                "quality": 0.2 + 0.8 * progress,
                "top_k": 0.1 + 0.9 * progress,
                "per_attr": 0.1 + 0.9 * progress,
            }
        else:
            # Stage 3: Full
            return {
                "quality": 1.0,
                "top_k": 1.0,
                "per_attr": 1.0,
            }
    
    def compute_per_attribute_improvement(
        self,
        attrs_all: np.ndarray,  # (T, 6)
        sel_idx: List[int],
    ) -> Dict[str, float]:
        """
        Compute quality improvement for each attribute.
        
        Improvement = mean(selected) - mean(all)
        
        This DIRECTLY corresponds to what evaluation metrics measure!
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return {name: 0.0 for name in ATTR_NAMES}
        
        sel_idx_valid = [i for i in sel_idx if i < len(attrs_all)]
        if len(sel_idx_valid) == 0:
            return {name: 0.0 for name in ATTR_NAMES}
        
        improvements = {}
        for name, idx in ATTR_INDEX.items():
            attr_all = attrs_all[:, idx]
            attr_sel = attr_all[sel_idx_valid]
            
            mean_all = float(np.mean(attr_all))
            mean_sel = float(np.mean(attr_sel))
            
            # Direct improvement: higher selected mean = better
            improvement = mean_sel - mean_all
            improvements[name] = improvement
        
        return improvements
    
    def compute_top_k_coverage(
        self,
        attrs_all: np.ndarray,  # (T, 6)
        sel_idx: List[int],
    ) -> Tuple[float, float, List[int]]:
        """
        Compute how well we cover the top-k quality frames.
        
        Returns:
            recall: Fraction of top-k frames that were selected
            precision: Fraction of selected frames that are in top-k
            top_k_indices: Indices of top-k quality frames
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return 0.0, 0.0, []
        
        T = len(attrs_all)
        k = max(1, int(T * self.top_k_ratio))
        
        # Compute aggregate quality (weighted sum of attributes)
        aggregate_quality = np.zeros(T)
        for name, idx in ATTR_INDEX.items():
            weight = self.attr_weights.get(name, 1.0)
            aggregate_quality += weight * attrs_all[:, idx]
        
        # Find top-k indices
        top_k_indices = np.argsort(aggregate_quality)[-k:].tolist()
        top_k_set = set(top_k_indices)
        sel_set = set(sel_idx)
        
        # Recall: How many of top-k did we select?
        selected_top_k = len(top_k_set & sel_set)
        recall = selected_top_k / len(top_k_set) if top_k_set else 0.0
        
        # Precision: How many of selected are in top-k?
        precision = selected_top_k / len(sel_set) if sel_set else 0.0
        
        return float(recall), float(precision), top_k_indices
    
    def compute_continuous_quality_reward(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
    ) -> float:
        """
        Continuous quality reward based on improvement.
        
        Unlike V3's binary (above/below threshold), this rewards
        proportionally to how much better the selection is.
        """
        improvements = self.compute_per_attribute_improvement(attrs_all, sel_idx)
        
        # Weighted sum of improvements
        total_improvement = 0.0
        for name, imp in improvements.items():
            weight = self.attr_weights.get(name, 1.0)
            total_improvement += weight * imp
        
        # Average across attributes
        num_attrs = len(improvements)
        avg_improvement = total_improvement / num_attrs if num_attrs > 0 else 0.0
        
        return avg_improvement * self.improvement_scale
    
    def compute_reward(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
        current_epoch: int = 0,
        motion: Optional[np.ndarray] = None,
        return_details: bool = False,
    ) -> Dict[str, float]:
        """
        Compute premium rewards with V4 enhancements.
        
        Returns dict with:
        - quality_improvement: Continuous quality reward
        - top_k_recall: Recall on top-k quality frames
        - top_k_precision: Precision on top-k frames
        - per_attr_*: Per-attribute improvement rewards
        - total: Weighted sum of all rewards
        """
        if len(sel_idx) == 0 or attrs_all.shape[0] == 0:
            result = {
                "quality_improvement": 0.0,
                "top_k_recall": 0.0,
                "top_k_precision": 0.0,
                "total": 0.0,
            }
            for name in ATTR_NAMES:
                result[f"per_attr_{name}"] = 0.0
            return result
        
        sel_idx_valid = [i for i in sel_idx if i < len(attrs_all)]
        if len(sel_idx_valid) == 0:
            result = {
                "quality_improvement": 0.0,
                "top_k_recall": 0.0,
                "top_k_precision": 0.0,
                "total": 0.0,
            }
            for name in ATTR_NAMES:
                result[f"per_attr_{name}"] = 0.0
            return result
        
        # --- Curriculum weights ---
        curriculum = self.get_curriculum_weights(current_epoch)
        
        # --- Per-attribute improvements ---
        improvements = self.compute_per_attribute_improvement(attrs_all, sel_idx_valid)
        
        # --- Continuous quality reward ---
        R_quality = self.compute_continuous_quality_reward(attrs_all, sel_idx_valid)
        
        # --- Top-k coverage ---
        top_k_recall, top_k_precision, top_k_indices = self.compute_top_k_coverage(attrs_all, sel_idx_valid)
        
        # --- Per-attribute weighted rewards ---
        R_per_attr = 0.0
        per_attr_rewards = {}
        for name, imp in improvements.items():
            weight = self.attr_weights.get(name, 1.0)
            reward = imp * weight * self.improvement_scale
            per_attr_rewards[name] = reward
            R_per_attr += reward
        R_per_attr /= len(improvements) if improvements else 1.0
        
        # --- Top-k reward ---
        R_top_k = (
            self.top_k_recall_weight * top_k_recall + 
            self.top_k_precision_weight * top_k_precision
        )
        
        # --- Apply curriculum ---
        R_quality_weighted = R_quality * curriculum["quality"]
        R_top_k_weighted = R_top_k * curriculum["top_k"]
        R_per_attr_weighted = R_per_attr * curriculum["per_attr"]
        
        # --- Total reward ---
        R_total = R_quality_weighted + R_top_k_weighted + R_per_attr_weighted
        
        # --- Update tracker ---
        self.tracker.update(improvements, top_k_recall, top_k_precision)
        
        # --- Build result ---
        result = {
            "quality_improvement": float(R_quality),
            "quality_improvement_weighted": float(R_quality_weighted),
            "top_k_recall": float(top_k_recall),
            "top_k_precision": float(top_k_precision),
            "top_k_reward": float(R_top_k),
            "top_k_reward_weighted": float(R_top_k_weighted),
            "per_attr_total": float(R_per_attr),
            "per_attr_total_weighted": float(R_per_attr_weighted),
            "total": float(R_total),
        }
        
        # Add per-attribute details
        for name in ATTR_NAMES:
            result[f"improvement_{name}"] = float(improvements.get(name, 0.0))
            result[f"reward_{name}"] = float(per_attr_rewards.get(name, 0.0))
        
        if return_details:
            result["top_k_indices"] = top_k_indices
            result["curriculum_weights"] = curriculum
        
        return result
    
    def get_curriculum_stage(self, current_epoch: int) -> str:
        """Get human-readable curriculum stage name."""
        if current_epoch <= self.stage1_end:
            return "Foundation"
        elif current_epoch <= self.stage2_end:
            return "Quality Focus"
        else:
            return "Full Optimization"
    
    def get_tracker_summary(self) -> Dict[str, float]:
        """Get summary of tracking metrics."""
        return self.tracker.get_summary()
    
    def reset_tracker(self):
        """Reset tracker for new epoch."""
        self.tracker.reset()


def compute_quality_metrics_for_eval(
    attrs_all: np.ndarray,
    sel_idx: List[int],
    attr_weights: Optional[Dict[str, float]] = None,
    top_k_ratio: float = 0.1,
) -> Dict[str, float]:
    """
    Compute quality metrics for evaluation (standalone function).
    
    Returns metrics that MATCH training rewards:
    - Per-attribute improvements
    - Top-k recall and precision
    - Aggregate quality improvement
    """
    reward_sys = PremiumAnimeRewardV4(
        attr_weights=attr_weights,
        top_k_ratio=top_k_ratio,
        use_curriculum=False,  # No curriculum for eval
    )
    
    return reward_sys.compute_reward(
        attrs_all=attrs_all,
        sel_idx=sel_idx,
        current_epoch=100,  # High epoch = full optimization
        return_details=True,
    )


if __name__ == "__main__":
    # Demo
    print("=== Premium Anime Reward V4 Demo ===\n")
    
    np.random.seed(42)
    T = 100
    attrs = np.random.rand(T, 6).astype(np.float32)
    
    # Create some high-quality frames (top 10%)
    high_quality_idx = list(range(90, 100))
    for idx in high_quality_idx:
        attrs[idx] = attrs[idx] * 0.5 + 0.5  # Boost quality
    
    print(f"Video frames: {T}")
    print(f"High-quality frames (ground truth): {high_quality_idx}\n")
    
    # Test 1: Select random frames
    random_sel = sorted(np.random.choice(T, size=10, replace=False).tolist())
    print(f"Random selection: {random_sel}")
    
    reward_sys = PremiumAnimeRewardV4(total_epochs=60)
    
    rewards_random = reward_sys.compute_reward(attrs, random_sel, current_epoch=50)
    print(f"  Quality improvement: {rewards_random['quality_improvement']:.4f}")
    print(f"  Top-k recall: {rewards_random['top_k_recall']:.4f}")
    print(f"  Top-k precision: {rewards_random['top_k_precision']:.4f}")
    print(f"  Total reward: {rewards_random['total']:.4f}\n")
    
    # Test 2: Select high-quality frames
    good_sel = high_quality_idx[:8] + [50, 60]  # 8 high-quality + 2 random
    print(f"Good selection (8 high-quality + 2 random): {sorted(good_sel)}")
    
    rewards_good = reward_sys.compute_reward(attrs, good_sel, current_epoch=50)
    print(f"  Quality improvement: {rewards_good['quality_improvement']:.4f}")
    print(f"  Top-k recall: {rewards_good['top_k_recall']:.4f}")
    print(f"  Top-k precision: {rewards_good['top_k_precision']:.4f}")
    print(f"  Total reward: {rewards_good['total']:.4f}\n")
    
    # Test 3: Curriculum
    print("=== Curriculum Test ===")
    for epoch in [1, 10, 30, 50]:
        stage = reward_sys.get_curriculum_stage(epoch)
        weights = reward_sys.get_curriculum_weights(epoch)
        print(f"Epoch {epoch:2d} ({stage}): quality={weights['quality']:.2f}, top_k={weights['top_k']:.2f}")
