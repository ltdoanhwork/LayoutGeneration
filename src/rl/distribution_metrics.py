#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution-Aware Anime IQA Metrics

This module provides metrics that evaluate keyframe selection relative to the
video's quality distribution, rather than using absolute scores.

Key metrics:
1. Percentile Rank: Average percentile of selected frames (0.5=random, 1.0=perfect)
2. Top-K Coverage: Fraction of top-K% frames selected
3. Z-Score Improvement: Normalized quality improvement
4. Threshold Metrics: Fraction of frames above P90, P75, etc.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

# Anime-CLIP-IQA attribute indices
ATTR_NAMES = ["sharpness", "colorfulness", "brightness", "sakuga", "cinematic", "expression"]
ATTR_INDEX = {name: i for i, name in enumerate(ATTR_NAMES)}


@dataclass
class DistributionMetricsResult:
    """Container for distribution-aware metrics results"""
    # Percentile-based metrics
    mean_percentile_rank: float = 0.0
    median_percentile_rank: float = 0.0
    min_percentile_rank: float = 0.0
    max_percentile_rank: float = 0.0
    
    # Top-K coverage
    top_10_coverage: float = 0.0  # Fraction of top 10% that was selected
    top_25_coverage: float = 0.0  # Fraction of top 25% that was selected
    top_10_precision: float = 0.0  # Fraction of selected that are in top 10%
    top_25_precision: float = 0.0  # Fraction of selected that are in top 25%
    
    # Z-Score metrics
    zscore_improvement: float = 0.0  # (mean_sel - mean_all) / std_all
    
    # Above-threshold metrics
    above_median_ratio: float = 0.0  # Fraction of selected above median
    above_p75_ratio: float = 0.0     # Fraction of selected above 75th percentile
    above_p90_ratio: float = 0.0     # Fraction of selected above 90th percentile
    
    # Missed frames penalty
    missed_top_k: int = 0  # Number of top-K frames not selected
    
    # Per-attribute percentile ranks (optional)
    per_attr_percentile: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for JSON serialization"""
        d = {
            "mean_percentile_rank": self.mean_percentile_rank,
            "median_percentile_rank": self.median_percentile_rank,
            "min_percentile_rank": self.min_percentile_rank,
            "max_percentile_rank": self.max_percentile_rank,
            "top_10_coverage": self.top_10_coverage,
            "top_25_coverage": self.top_25_coverage,
            "top_10_precision": self.top_10_precision,
            "top_25_precision": self.top_25_precision,
            "zscore_improvement": self.zscore_improvement,
            "above_median_ratio": self.above_median_ratio,
            "above_p75_ratio": self.above_p75_ratio,
            "above_p90_ratio": self.above_p90_ratio,
            "missed_top_k": self.missed_top_k,
        }
        for k, v in self.per_attr_percentile.items():
            d[f"percentile_{k}"] = v
        return d


class DistributionAwareMetrics:
    """
    Compute distribution-aware metrics for keyframe selection.
    
    These metrics evaluate selection quality relative to the video's
    quality distribution, making them comparable across different videos.
    """
    
    def __init__(
        self,
        top_k_ratio: float = 0.1,
        attr_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            top_k_ratio: Ratio of frames considered "top quality" (default: 10%)
            attr_weights: Weights for aggregating attributes (default: equal weights)
        """
        self.top_k_ratio = top_k_ratio
        self.attr_weights = attr_weights or {
            "sharpness": 1.0,
            "colorfulness": 1.0,
            "brightness": 0.5,
            "sakuga": 2.0,
            "cinematic": 1.5,
            "expression": 1.5,
        }
    
    def compute_aggregate_quality(self, attrs: np.ndarray) -> np.ndarray:
        """
        Compute weighted aggregate quality score for each frame.
        
        Args:
            attrs: (T, 6) anime IQA attributes
            
        Returns:
            (T,) aggregate quality scores
        """
        T = len(attrs)
        quality = np.zeros(T)
        
        for name, idx in ATTR_INDEX.items():
            weight = self.attr_weights.get(name, 1.0)
            quality += weight * attrs[:, idx]
        
        return quality
    
    def compute_percentile_ranks(self, values: np.ndarray) -> np.ndarray:
        """
        Compute percentile rank for each value (0 = worst, 1 = best).
        
        Args:
            values: (T,) quality values
            
        Returns:
            (T,) percentile ranks in [0, 1]
        """
        T = len(values)
        if T <= 1:
            return np.ones(T)
        
        # Rank-based percentile (handles ties properly)
        ranks = np.argsort(np.argsort(values))  # 0 = lowest, T-1 = highest
        percentiles = ranks / (T - 1)
        return percentiles
    
    def compute_percentile_metrics(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
    ) -> Dict[str, float]:
        """
        Compute percentile-based metrics.
        
        Args:
            attrs_all: (T, 6) all frame attributes
            sel_idx: List of selected frame indices
            
        Returns:
            Dict with percentile metrics
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return {
                "mean_percentile_rank": 0.0,
                "median_percentile_rank": 0.0,
                "min_percentile_rank": 0.0,
                "max_percentile_rank": 0.0,
            }
        
        T = len(attrs_all)
        sel_idx = [i for i in sel_idx if 0 <= i < T]
        if len(sel_idx) == 0:
            return {
                "mean_percentile_rank": 0.0,
                "median_percentile_rank": 0.0,
                "min_percentile_rank": 0.0,
                "max_percentile_rank": 0.0,
            }
        
        # Aggregate quality
        quality = self.compute_aggregate_quality(attrs_all)
        percentiles = self.compute_percentile_ranks(quality)
        
        # Get percentiles of selected frames
        sel_percentiles = percentiles[sel_idx]
        
        return {
            "mean_percentile_rank": float(np.mean(sel_percentiles)),
            "median_percentile_rank": float(np.median(sel_percentiles)),
            "min_percentile_rank": float(np.min(sel_percentiles)),
            "max_percentile_rank": float(np.max(sel_percentiles)),
        }
    
    def compute_topk_metrics(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
        k_ratio: float = 0.1,
    ) -> Dict[str, float]:
        """
        Compute top-K coverage and precision metrics.
        
        Coverage = |selected ∩ top-K| / |top-K|
        Precision = |selected ∩ top-K| / |selected|
        
        Args:
            attrs_all: (T, 6) all frame attributes
            sel_idx: Selected indices
            k_ratio: Fraction of frames to consider as "top"
            
        Returns:
            Dict with coverage and precision metrics
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return {
                f"top_{int(k_ratio*100)}_coverage": 0.0,
                f"top_{int(k_ratio*100)}_precision": 0.0,
                "missed_top_k": 0,
            }
        
        T = len(attrs_all)
        sel_idx = [i for i in sel_idx if 0 <= i < T]
        if len(sel_idx) == 0:
            k = max(1, int(T * k_ratio))
            return {
                f"top_{int(k_ratio*100)}_coverage": 0.0,
                f"top_{int(k_ratio*100)}_precision": 0.0,
                "missed_top_k": k,
            }
        
        # Get top-K indices by aggregate quality
        quality = self.compute_aggregate_quality(attrs_all)
        k = max(1, int(T * k_ratio))
        top_k_indices = set(np.argsort(quality)[-k:])
        
        sel_set = set(sel_idx)
        intersection = top_k_indices & sel_set
        
        coverage = len(intersection) / len(top_k_indices) if top_k_indices else 0.0
        precision = len(intersection) / len(sel_set) if sel_set else 0.0
        missed = len(top_k_indices - sel_set)
        
        return {
            f"top_{int(k_ratio*100)}_coverage": float(coverage),
            f"top_{int(k_ratio*100)}_precision": float(precision),
            "missed_top_k": int(missed),
        }
    
    def compute_zscore_metrics(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
    ) -> Dict[str, float]:
        """
        Compute z-score normalized quality improvement.
        
        Z-score = (mean_selected - mean_all) / std_all
        
        This is scale-invariant and comparable across videos.
        
        Args:
            attrs_all: (T, 6) all frame attributes
            sel_idx: Selected indices
            
        Returns:
            Dict with z-score improvement
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return {"zscore_improvement": 0.0}
        
        T = len(attrs_all)
        sel_idx = [i for i in sel_idx if 0 <= i < T]
        if len(sel_idx) == 0:
            return {"zscore_improvement": 0.0}
        
        # Aggregate quality
        quality = self.compute_aggregate_quality(attrs_all)
        
        mean_all = float(np.mean(quality))
        std_all = float(np.std(quality))
        mean_sel = float(np.mean(quality[sel_idx]))
        
        if std_all < 1e-8:
            # No variance - can't compute meaningful z-score
            zscore = 0.0 if abs(mean_sel - mean_all) < 1e-8 else 1.0
        else:
            zscore = (mean_sel - mean_all) / std_all
        
        return {"zscore_improvement": zscore}
    
    def compute_threshold_metrics(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
    ) -> Dict[str, float]:
        """
        Compute metrics based on threshold crossings.
        
        How many selected frames are above median, P75, P90?
        
        Args:
            attrs_all: (T, 6) all frame attributes
            sel_idx: Selected indices
            
        Returns:
            Dict with above-threshold ratios
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return {
                "above_median_ratio": 0.0,
                "above_p75_ratio": 0.0,
                "above_p90_ratio": 0.0,
            }
        
        T = len(attrs_all)
        sel_idx = [i for i in sel_idx if 0 <= i < T]
        if len(sel_idx) == 0:
            return {
                "above_median_ratio": 0.0,
                "above_p75_ratio": 0.0,
                "above_p90_ratio": 0.0,
            }
        
        # Aggregate quality
        quality = self.compute_aggregate_quality(attrs_all)
        sel_quality = quality[sel_idx]
        
        # Compute percentiles
        p50 = np.percentile(quality, 50)
        p75 = np.percentile(quality, 75)
        p90 = np.percentile(quality, 90)
        
        K = len(sel_idx)
        above_median = np.sum(sel_quality >= p50) / K
        above_p75 = np.sum(sel_quality >= p75) / K
        above_p90 = np.sum(sel_quality >= p90) / K
        
        return {
            "above_median_ratio": float(above_median),
            "above_p75_ratio": float(above_p75),
            "above_p90_ratio": float(above_p90),
        }
    
    def compute_per_attribute_percentile(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
    ) -> Dict[str, float]:
        """
        Compute mean percentile rank for each attribute separately.
        
        Args:
            attrs_all: (T, 6) all frame attributes
            sel_idx: Selected indices
            
        Returns:
            Dict mapping attribute name to mean percentile
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return {f"percentile_{name}": 0.0 for name in ATTR_NAMES}
        
        T = len(attrs_all)
        sel_idx = [i for i in sel_idx if 0 <= i < T]
        if len(sel_idx) == 0:
            return {f"percentile_{name}": 0.0 for name in ATTR_NAMES}
        
        result = {}
        for name, idx in ATTR_INDEX.items():
            attr_values = attrs_all[:, idx]
            percentiles = self.compute_percentile_ranks(attr_values)
            sel_percentiles = percentiles[sel_idx]
            result[f"percentile_{name}"] = float(np.mean(sel_percentiles))
        
        return result
    
    def compute_all_metrics(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
        include_per_attr: bool = True,
    ) -> DistributionMetricsResult:
        """
        Compute all distribution-aware metrics.
        
        Args:
            attrs_all: (T, 6) all frame attributes
            sel_idx: Selected indices
            include_per_attr: Whether to include per-attribute percentiles
            
        Returns:
            DistributionMetricsResult with all metrics
        """
        result = DistributionMetricsResult()
        
        # Percentile metrics
        percentile = self.compute_percentile_metrics(attrs_all, sel_idx)
        result.mean_percentile_rank = percentile["mean_percentile_rank"]
        result.median_percentile_rank = percentile["median_percentile_rank"]
        result.min_percentile_rank = percentile["min_percentile_rank"]
        result.max_percentile_rank = percentile["max_percentile_rank"]
        
        # Top-K metrics (10% and 25%)
        top10 = self.compute_topk_metrics(attrs_all, sel_idx, k_ratio=0.1)
        top25 = self.compute_topk_metrics(attrs_all, sel_idx, k_ratio=0.25)
        result.top_10_coverage = top10["top_10_coverage"]
        result.top_10_precision = top10["top_10_precision"]
        result.top_25_coverage = top25["top_25_coverage"]
        result.top_25_precision = top25["top_25_precision"]
        result.missed_top_k = top10["missed_top_k"]
        
        # Z-score metrics
        zscore = self.compute_zscore_metrics(attrs_all, sel_idx)
        result.zscore_improvement = zscore["zscore_improvement"]
        
        # Threshold metrics
        threshold = self.compute_threshold_metrics(attrs_all, sel_idx)
        result.above_median_ratio = threshold["above_median_ratio"]
        result.above_p75_ratio = threshold["above_p75_ratio"]
        result.above_p90_ratio = threshold["above_p90_ratio"]
        
        # Per-attribute percentiles
        if include_per_attr:
            result.per_attr_percentile = self.compute_per_attribute_percentile(attrs_all, sel_idx)
        
        return result
    
    def get_selection_distribution_data(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
    ) -> Dict[str, Any]:
        """
        Get data needed for distribution visualization.
        
        Args:
            attrs_all: (T, 6) all frame attributes
            sel_idx: Selected indices
            
        Returns:
            Dict with arrays for visualization
        """
        T = len(attrs_all)
        sel_idx_valid = [i for i in sel_idx if 0 <= i < T]
        
        quality = self.compute_aggregate_quality(attrs_all)
        percentiles = self.compute_percentile_ranks(quality)
        
        # Create selection mask
        sel_mask = np.zeros(T, dtype=bool)
        sel_mask[sel_idx_valid] = True
        
        return {
            "quality_all": quality.tolist(),
            "percentiles_all": percentiles.tolist(),
            "quality_selected": quality[sel_idx_valid].tolist() if sel_idx_valid else [],
            "percentiles_selected": percentiles[sel_idx_valid].tolist() if sel_idx_valid else [],
            "frame_indices_selected": sel_idx_valid,
            "selection_mask": sel_mask.tolist(),
            "total_frames": T,
            "num_selected": len(sel_idx_valid),
            "attrs_all": attrs_all.tolist(),
        }


def compute_distribution_metrics_for_eval(
    attrs_all: np.ndarray,
    sel_idx: List[int],
    attr_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Convenience function for evaluation.
    
    Args:
        attrs_all: (T, 6) all frame attributes
        sel_idx: Selected indices
        attr_weights: Optional attribute weights
        
    Returns:
        Flat dict of all metrics
    """
    metrics = DistributionAwareMetrics(attr_weights=attr_weights)
    result = metrics.compute_all_metrics(attrs_all, sel_idx, include_per_attr=True)
    return result.to_dict()


def compute_per_scene_distribution_metrics(
    attrs_all: np.ndarray,
    sel_idx: List[int],
    scene_boundaries: List[Tuple[int, int]],
    attr_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute metrics averaged across scenes.
    
    This evaluates how well the model picks frames relative to each scene's 
    local distribution, which aligns with the per-scene training approach.
    
    Args:
        attrs_all: (T, 6) all frame attributes
        sel_idx: Global indices of selected frames
        scene_boundaries: List of (start_idx, end_idx) in Global space
        attr_weights: Optional attribute weights
        
    Returns:
        Dict with averaged per-scene metrics
    """
    metrics_computer = DistributionAwareMetrics(attr_weights=attr_weights)
    sel_set = set(sel_idx)
    
    scene_results = []
    
    for start, end in scene_boundaries:
        if end <= start:
            continue
            
        # Extract scene attributes
        scene_attrs = attrs_all[start:end+1]
        
        # Get selected indices within this scene (and make them local)
        scene_sel_local = [i - start for i in sel_idx if start <= i <= end]
        
        if len(scene_sel_local) == 0:
            continue
            
        # Compute metrics for this scene
        scene_res = metrics_computer.compute_all_metrics(scene_attrs, scene_sel_local, include_per_attr=False)
        scene_results.append(scene_res.to_dict())
        
    if not scene_results:
        return {}
        
    # Average across scenes
    agg = {}
    for key in scene_results[0].keys():
        values = [r[key] for r in scene_results if key in r]
        agg[f"local_{key}"] = float(np.mean(values))
        
    return agg


if __name__ == "__main__":
    # Demo/test
    print("=== Distribution-Aware Metrics Demo ===\n")
    
    np.random.seed(42)
    T = 100
    
    # Create synthetic data with some high-quality frames
    attrs = np.random.rand(T, 6).astype(np.float32) * 0.5  # Base quality 0-0.5
    
    # Make last 20 frames high quality
    high_quality_idx = list(range(80, 100))
    for idx in high_quality_idx:
        attrs[idx] = 0.5 + np.random.rand(6) * 0.5  # Quality 0.5-1.0
    
    print(f"Video frames: {T}")
    print(f"High-quality frames: {high_quality_idx[:5]}... (indices 80-99)\n")
    
    metrics = DistributionAwareMetrics()
    
    # Test 1: Random selection
    random_sel = sorted(np.random.choice(T, size=10, replace=False).tolist())
    print(f"Random selection: {random_sel[:5]}...")
    result_random = metrics.compute_all_metrics(attrs, random_sel)
    print(f"  Mean percentile: {result_random.mean_percentile_rank:.3f}")
    print(f"  Top-10 coverage: {result_random.top_10_coverage:.3f}")
    print(f"  Z-score improvement: {result_random.zscore_improvement:.3f}")
    print(f"  Above P90 ratio: {result_random.above_p90_ratio:.3f}\n")
    
    # Test 2: Good selection (mostly high-quality frames)
    good_sel = sorted(high_quality_idx[:8] + [50, 60])  # 8 high-quality + 2 random
    print(f"Good selection: {good_sel[:5]}...")
    result_good = metrics.compute_all_metrics(attrs, good_sel)
    print(f"  Mean percentile: {result_good.mean_percentile_rank:.3f}")
    print(f"  Top-10 coverage: {result_good.top_10_coverage:.3f}")
    print(f"  Z-score improvement: {result_good.zscore_improvement:.3f}")
    print(f"  Above P90 ratio: {result_good.above_p90_ratio:.3f}\n")
    
    # Test 3: Perfect selection (all top-10 frames)
    perfect_sel = list(range(90, 100))  # Top 10 frames
    print(f"Perfect selection: {perfect_sel[:5]}...")
    result_perfect = metrics.compute_all_metrics(attrs, perfect_sel)
    print(f"  Mean percentile: {result_perfect.mean_percentile_rank:.3f}")
    print(f"  Top-10 coverage: {result_perfect.top_10_coverage:.3f}")
    print(f"  Z-score improvement: {result_perfect.zscore_improvement:.3f}")
    print(f"  Above P90 ratio: {result_perfect.above_p90_ratio:.3f}\n")
    
    print("✅ Demo completed successfully!")
