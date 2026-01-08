#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for distribution-aware metrics.

Run with: python -m pytest tests/test_distribution_metrics.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.distribution_metrics import (
    DistributionAwareMetrics,
    DistributionMetricsResult,
    compute_distribution_metrics_for_eval,
)


class TestDistributionAwareMetrics:
    """Test suite for DistributionAwareMetrics class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample anime attrs data for testing."""
        np.random.seed(42)
        T = 100
        # 6 attributes: sharpness, colorfulness, brightness, sakuga, cinematic, expression
        attrs = np.random.rand(T, 6).astype(np.float32) * 0.5  # Low quality baseline
        
        # Make last 20 frames high quality
        for i in range(80, 100):
            attrs[i] = 0.5 + np.random.rand(6) * 0.5
        
        return attrs
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return DistributionAwareMetrics()
    
    def test_compute_percentile_ranks(self, metrics, sample_data):
        """Test percentile rank computation."""
        quality = metrics.compute_aggregate_quality(sample_data)
        percentiles = metrics.compute_percentile_ranks(quality)
        
        # Should be in [0, 1]
        assert percentiles.min() >= 0.0
        assert percentiles.max() <= 1.0
        
        # High quality frames (80-99) should have high percentiles
        high_quality_percentiles = percentiles[80:100].mean()
        low_quality_percentiles = percentiles[0:80].mean()
        assert high_quality_percentiles > low_quality_percentiles
    
    def test_perfect_selection(self, metrics, sample_data):
        """Test metrics with perfect selection (top 10 frames)."""
        # Select top 10 frames (should be mostly from 80-99)
        quality = metrics.compute_aggregate_quality(sample_data)
        top_10_idx = sorted(np.argsort(quality)[-10:].tolist())
        
        result = metrics.compute_all_metrics(sample_data, top_10_idx)
        
        # Perfect selection should have high percentile rank
        assert result.mean_percentile_rank > 0.9
        
        # Top-10 coverage should be 100% (we selected exactly top 10)
        assert result.top_10_coverage == 1.0
        
        # Z-score improvement should be positive and significant
        assert result.zscore_improvement > 0
    
    def test_random_selection(self, metrics, sample_data):
        """Test metrics with random selection."""
        np.random.seed(123)
        random_sel = sorted(np.random.choice(100, size=10, replace=False).tolist())
        
        result = metrics.compute_all_metrics(sample_data, random_sel)
        
        # Random selection should have percentile around 0.5
        assert 0.3 < result.mean_percentile_rank < 0.7
        
        # Top-10 coverage should be low for random selection
        assert result.top_10_coverage < 0.5
    
    def test_worst_selection(self, metrics, sample_data):
        """Test metrics with worst selection (bottom 10 frames)."""
        quality = metrics.compute_aggregate_quality(sample_data)
        bottom_10_idx = sorted(np.argsort(quality)[:10].tolist())
        
        result = metrics.compute_all_metrics(sample_data, bottom_10_idx)
        
        # Worst selection should have low percentile rank
        assert result.mean_percentile_rank < 0.15
        
        # Top-10 coverage should be 0
        assert result.top_10_coverage == 0.0
        
        # Z-score improvement should be negative
        assert result.zscore_improvement < 0
    
    def test_empty_selection(self, metrics, sample_data):
        """Test metrics with empty selection."""
        result = metrics.compute_all_metrics(sample_data, [])
        
        assert result.mean_percentile_rank == 0.0
        assert result.top_10_coverage == 0.0
        assert result.zscore_improvement == 0.0
    
    def test_single_frame_video(self, metrics):
        """Test metrics with single-frame video."""
        single_frame_attrs = np.random.rand(1, 6).astype(np.float32)
        
        result = metrics.compute_all_metrics(single_frame_attrs, [0])
        
        # Should handle single frame without error
        assert result.mean_percentile_rank == 1.0
    
    def test_topk_metrics(self, metrics, sample_data):
        """Test top-K coverage and precision metrics."""
        # Select all frames 80-89 (10 frames from top)
        sel_idx = list(range(80, 90))
        
        top10 = metrics.compute_topk_metrics(sample_data, sel_idx, k_ratio=0.1)
        top25 = metrics.compute_topk_metrics(sample_data, sel_idx, k_ratio=0.25)
        
        # Coverage should be high for both
        assert top10["top_10_coverage"] > 0.5
        assert top25["top_25_coverage"] > 0.3
    
    def test_zscore_metrics(self, metrics, sample_data):
        """Test z-score computation."""
        # Select high quality frames
        sel_idx = list(range(90, 100))
        
        zscore = metrics.compute_zscore_metrics(sample_data, sel_idx)
        
        # Should be positive and significant
        assert zscore["zscore_improvement"] > 1.0
    
    def test_per_attribute_percentile(self, metrics, sample_data):
        """Test per-attribute percentile computation."""
        sel_idx = list(range(90, 100))
        
        per_attr = metrics.compute_per_attribute_percentile(sample_data, sel_idx)
        
        # Should have all 6 attributes
        assert len(per_attr) == 6
        
        # All should be high (> 0.5) for high-quality selection
        for name, val in per_attr.items():
            assert val > 0.5, f"{name} percentile too low: {val}"
    
    def test_threshold_metrics(self, metrics, sample_data):
        """Test above-threshold ratio metrics."""
        # Select high quality frames
        sel_idx = list(range(90, 100))
        
        threshold = metrics.compute_threshold_metrics(sample_data, sel_idx)
        
        # High quality selection should mostly be above median
        assert threshold["above_median_ratio"] > 0.8
        
        # And many above P75
        assert threshold["above_p75_ratio"] > 0.5
    
    def test_result_to_dict(self, metrics, sample_data):
        """Test DistributionMetricsResult.to_dict()."""
        sel_idx = list(range(90, 100))
        result = metrics.compute_all_metrics(sample_data, sel_idx, include_per_attr=True)
        
        d = result.to_dict()
        
        # Should include all expected keys
        expected_keys = [
            "mean_percentile_rank",
            "zscore_improvement", 
            "top_10_coverage",
            "above_p90_ratio",
        ]
        
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"
        
        # Should include per-attribute percentiles
        assert "percentile_sakuga" in d
    
    def test_visualization_data(self, metrics, sample_data):
        """Test get_selection_distribution_data()."""
        sel_idx = list(range(90, 100))
        
        data = metrics.get_selection_distribution_data(sample_data, sel_idx)
        
        assert "quality_all" in data
        assert "quality_selected" in data
        assert "frame_indices_selected" in data
        assert len(data["quality_all"]) == 100
        assert len(data["quality_selected"]) == 10


class TestConvenienceFunction:
    """Test the convenience function for evaluation."""
    
    def test_compute_distribution_metrics_for_eval(self):
        """Test standalone evaluation function."""
        np.random.seed(42)
        attrs = np.random.rand(50, 6).astype(np.float32)
        sel_idx = [40, 42, 45, 47, 49]  # Top-end frames
        
        metrics = compute_distribution_metrics_for_eval(attrs, sel_idx)
        
        assert isinstance(metrics, dict)
        assert "mean_percentile_rank" in metrics
        assert "zscore_improvement" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
