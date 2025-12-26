#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Premium Rewards V9: Quality-Focused Keyframe Selection

Key improvements over V8:
1. INCREASED quality weighting (anime_scale: 3.0 → 5.0)
2. NEW: Explicit percentile-based reward for selecting high-percentile frames
3. REMOVED: Motion features dependency
4. REDUCED: Constraint penalties (focus on quality over constraints)

Goal: Maximize Mean Percentile Rank and Top-K Coverage of selected keyframes.

Metrics to optimize:
- Mean Percentile Rank: > 0.70 (0.5 = random)
- Top-10% Coverage: > 20%
- Z-Score Improvement: > 0.5
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, NamedTuple
import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import det, inv
import warnings

# Anime-CLIP-IQA attribute indices
ATTR_NAMES = ["sharpness", "colorfulness", "brightness", "sakuga", "cinematic", "expression"]
ATTR_INDEX = {name: i for i, name in enumerate(ATTR_NAMES)}


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ConstraintConfig:
    """Configuration for constraint thresholds (tau values)"""
    # Constraint thresholds - constraint is satisfied if value <= threshold
    rec_err_threshold: float = 0.35      # tau: max acceptable RecErr
    coverage_threshold: float = 0.3      # tau_c: max gap variance (lower = better coverage)
    diversity_threshold: float = 0.25    # tau_k: min pairwise distance (higher = better)
    
    # Lagrangian learning
    lambda_lr: float = 0.01              # Learning rate for dual variables
    lambda_max: float = 10.0             # Cap on multipliers
    lambda_init: float = 1.0             # Initial multiplier values


@dataclass
class DPPConfig:
    """Configuration for DPP selection"""
    beta: float = 1.0                    # Distance scaling in kernel
    quality_power: float = 1.0           # Power for quality scores
    candidate_ratio: float = 0.3         # Top p% as candidates in stage A
    use_greedy_approx: bool = True       # Use greedy DPP (faster)


@dataclass
class TestTimeScalingConfig:
    """Configuration for test-time scaling"""
    n_samples: int = 8                   # Number of summaries to sample
    temperature: float = 1.2             # Sampling temperature
    top_p: float = 0.9                   # Nucleus sampling threshold
    constraint_penalty: float = 100.0    # Penalty for constraint violations


# ============================================================================
# Constraint Tracking
# ============================================================================

@dataclass
class ConstraintTracker:
    """Track constraint satisfaction and violations"""
    rec_err_history: List[float] = field(default_factory=list)
    coverage_history: List[float] = field(default_factory=list)  
    diversity_history: List[float] = field(default_factory=list)
    
    # Violation rates (binary: 1 if violated, 0 if satisfied)
    rec_err_violations: List[int] = field(default_factory=list)
    coverage_violations: List[int] = field(default_factory=list)
    diversity_violations: List[int] = field(default_factory=list)
    
    # Lagrangian multipliers history
    lambda_rec_history: List[float] = field(default_factory=list)
    lambda_cov_history: List[float] = field(default_factory=list)
    lambda_div_history: List[float] = field(default_factory=list)
    
    # Quality metrics
    anime_score_history: List[float] = field(default_factory=list)
    quantile_score_history: List[float] = field(default_factory=list)
    top_k_recall_history: List[float] = field(default_factory=list)
    
    def log_constraints(
        self, 
        rec_err: float, 
        coverage: float, 
        diversity: float,
        config: ConstraintConfig
    ):
        self.rec_err_history.append(rec_err)
        self.coverage_history.append(coverage)
        self.diversity_history.append(diversity)
        
        # Check violations
        self.rec_err_violations.append(1 if rec_err > config.rec_err_threshold else 0)
        self.coverage_violations.append(1 if coverage > config.coverage_threshold else 0)
        self.diversity_violations.append(1 if diversity < config.diversity_threshold else 0)
    
    def log_lambdas(self, lambda_rec: float, lambda_cov: float, lambda_div: float):
        self.lambda_rec_history.append(lambda_rec)
        self.lambda_cov_history.append(lambda_cov)
        self.lambda_div_history.append(lambda_div)
    
    def log_quality(self, anime_score: float, quantile_score: float, top_k_recall: float):
        self.anime_score_history.append(anime_score)
        self.quantile_score_history.append(quantile_score)
        self.top_k_recall_history.append(top_k_recall)
    
    def get_summary(self, last_n: int = 100) -> Dict[str, float]:
        """Get summary statistics for logging"""
        def safe_mean(lst):
            if not lst: return 0.0
            return float(np.mean(lst[-last_n:]))
        
        def violation_rate(lst):
            if not lst: return 0.0
            return float(np.mean(lst[-last_n:]))
        
        return {
            # Constraint values
            "rec_err_mean": safe_mean(self.rec_err_history),
            "coverage_mean": safe_mean(self.coverage_history),
            "diversity_mean": safe_mean(self.diversity_history),
            # Violation rates
            "rec_err_violation_rate": violation_rate(self.rec_err_violations),
            "coverage_violation_rate": violation_rate(self.coverage_violations),
            "diversity_violation_rate": violation_rate(self.diversity_violations),
            # Lambda values
            "lambda_rec": safe_mean(self.lambda_rec_history),
            "lambda_cov": safe_mean(self.lambda_cov_history),
            "lambda_div": safe_mean(self.lambda_div_history),
            # Quality
            "anime_score_mean": safe_mean(self.anime_score_history),
            "quantile_score_mean": safe_mean(self.quantile_score_history),
            "top_k_recall_mean": safe_mean(self.top_k_recall_history),
        }


# ============================================================================
# Lagrangian Optimizer
# ============================================================================

class LagrangianOptimizer:
    """
    Dual gradient descent for constraint multipliers.
    
    Updates λ based on constraint violations:
    λ_new = λ_old + lr * (constraint_value - threshold)
    """
    
    def __init__(self, config: ConstraintConfig):
        self.config = config
        # Initialize multipliers
        self.lambda_rec = config.lambda_init
        self.lambda_cov = config.lambda_init
        self.lambda_div = config.lambda_init
    
    def update(
        self, 
        rec_err: float, 
        coverage_gap: float, 
        diversity: float
    ) -> Dict[str, float]:
        """
        Update multipliers based on constraint violations.
        
        Args:
            rec_err: Current reconstruction error
            coverage_gap: Gap variance (lower = better coverage)
            diversity: Mean pairwise distance (higher = better)
        
        Returns:
            Dict with updated lambda values
        """
        cfg = self.config
        
        # Gradient for each constraint (violation = positive gradient)
        grad_rec = rec_err - cfg.rec_err_threshold
        grad_cov = coverage_gap - cfg.coverage_threshold
        grad_div = cfg.diversity_threshold - diversity  # Note: flipped for min constraint
        
        # Update with gradient ascent (maximize Lagrangian = dual descent)
        self.lambda_rec = np.clip(
            self.lambda_rec + cfg.lambda_lr * grad_rec, 
            0.0, cfg.lambda_max
        )
        self.lambda_cov = np.clip(
            self.lambda_cov + cfg.lambda_lr * grad_cov,
            0.0, cfg.lambda_max
        )
        self.lambda_div = np.clip(
            self.lambda_div + cfg.lambda_lr * grad_div,
            0.0, cfg.lambda_max
        )
        
        return {
            "lambda_rec": self.lambda_rec,
            "lambda_cov": self.lambda_cov,
            "lambda_div": self.lambda_div,
        }
    
    def get_penalty(
        self, 
        rec_err: float, 
        coverage_gap: float, 
        diversity: float
    ) -> float:
        """
        Compute Lagrangian penalty for constraints.
        
        Returns:
            Total penalty (negative reward for violations)
        """
        cfg = self.config
        
        # Constraint violations (positive if violated)
        viol_rec = max(0, rec_err - cfg.rec_err_threshold)
        viol_cov = max(0, coverage_gap - cfg.coverage_threshold)
        viol_div = max(0, cfg.diversity_threshold - diversity)
        
        # Weighted penalty
        penalty = (
            self.lambda_rec * viol_rec +
            self.lambda_cov * viol_cov +
            self.lambda_div * viol_div
        )
        
        return penalty
    
    def state_dict(self) -> Dict[str, float]:
        return {
            "lambda_rec": self.lambda_rec,
            "lambda_cov": self.lambda_cov,
            "lambda_div": self.lambda_div,
        }
    
    def load_state_dict(self, state: Dict[str, float]):
        self.lambda_rec = state.get("lambda_rec", self.config.lambda_init)
        self.lambda_cov = state.get("lambda_cov", self.config.lambda_init)
        self.lambda_div = state.get("lambda_div", self.config.lambda_init)


# ============================================================================
# DPP Selector
# ============================================================================

class DPPSelector:
    """
    Determinantal Point Process for diverse subset selection.
    
    Kernel: K_ij = q_i * q_j * exp(-β * d(f_i, f_j))
    
    where q_i is quality score and d is cosine distance.
    """
    
    def __init__(self, config: DPPConfig):
        self.config = config
    
    def compute_kernel(
        self, 
        quality_scores: np.ndarray,  # (N,)
        features: np.ndarray,        # (N, D)
    ) -> np.ndarray:
        """
        Compute DPP kernel matrix.
        
        Args:
            quality_scores: Quality score for each item
            features: Feature vectors (normalized)
        
        Returns:
            (N, N) kernel matrix
        """
        N = len(quality_scores)
        cfg = self.config
        
        # Quality diagonal (raised to power for emphasis)
        # Clip to avoid numerical issues
        q = np.power(np.clip(quality_scores, 1e-6, 1.0), cfg.quality_power)
        
        # Normalize features if not already
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        features = features / norms
        
        # Similarity matrix (cosine similarity)
        S = features @ features.T  # (N, N)
        
        # Clip similarity to valid range
        S = np.clip(S, -1.0, 1.0)
        
        # Distance = 1 - similarity
        D = 1.0 - S
        
        # Kernel: K_ij = q_i * q_j * exp(-β * d_ij)
        K = np.outer(q, q) * np.exp(-cfg.beta * D)
        
        # Ensure symmetric
        K = (K + K.T) / 2
        
        # Add regularization for numerical stability
        K = K + 1e-6 * np.eye(N)
        
        return K
    
    def select_greedy(
        self, 
        kernel: np.ndarray,  # (N, N)
        k: int
    ) -> List[int]:
        """
        Greedy MAP inference for DPP.
        
        Iteratively selects items that maximize log det(K_S).
        """
        N = kernel.shape[0]
        k = min(k, N)
        
        selected = []
        remaining = set(range(N))
        
        for _ in range(k):
            best_idx = -1
            best_gain = -np.inf
            
            for idx in remaining:
                # Compute gain from adding idx
                S_new = selected + [idx]
                K_S = kernel[np.ix_(S_new, S_new)]
                
                try:
                    # Log determinant
                    sign, logdet = np.linalg.slogdet(K_S)
                    if sign > 0:
                        gain = logdet
                    else:
                        gain = -np.inf
                except:
                    gain = -np.inf
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        return sorted(selected)
    
    def select(
        self,
        quality_scores: np.ndarray,
        features: np.ndarray,
        k: int
    ) -> List[int]:
        """
        Select k diverse items using DPP.
        
        Args:
            quality_scores: (N,) quality scores
            features: (N, D) normalized features
            k: Number of items to select
        
        Returns:
            List of selected indices
        """
        if len(quality_scores) <= k:
            return list(range(len(quality_scores)))
        
        kernel = self.compute_kernel(quality_scores, features)
        return self.select_greedy(kernel, k)


# ============================================================================
# Two-Stage Selection
# ============================================================================

class TwoStageSelector:
    """
    Stage A: Candidate mining (top p% by quality)
    Stage B: DPP selection from candidates
    """
    
    def __init__(self, dpp_config: DPPConfig):
        self.dpp_config = dpp_config
        self.dpp_selector = DPPSelector(dpp_config)
    
    def select(
        self,
        quality_scores: np.ndarray,  # (T,) per-frame quality
        features: np.ndarray,         # (T, D) normalized features
        k: int,                       # Target number of keyframes
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Two-stage selection.
        
        Returns:
            selected_indices: Final selected frame indices
            info: Statistics about selection
        """
        T = len(quality_scores)
        
        # Stage A: Candidate mining (top p%)
        n_candidates = max(k * 2, int(T * self.dpp_config.candidate_ratio))
        n_candidates = min(n_candidates, T)
        
        # Get top candidates by quality
        candidate_indices = np.argsort(quality_scores)[-n_candidates:]
        
        # Stage B: DPP selection from candidates
        candidate_quality = quality_scores[candidate_indices]
        candidate_features = features[candidate_indices]
        
        # Select k from candidates using DPP
        dpp_selection = self.dpp_selector.select(
            candidate_quality, 
            candidate_features, 
            k
        )
        
        # Map back to original indices
        selected = [int(candidate_indices[i]) for i in dpp_selection]
        
        # Compute info
        info = {
            "n_candidates": n_candidates,
            "candidate_mean_quality": float(np.mean(candidate_quality)),
            "selected_mean_quality": float(np.mean(quality_scores[selected])),
            "dpp_diversity": self._compute_diversity(features[selected]) if len(selected) > 1 else 0.0,
        }
        
        return sorted(selected), info
    
    def _compute_diversity(self, features: np.ndarray) -> float:
        """Compute mean pairwise cosine distance"""
        S = features @ features.T
        iu = np.triu_indices(len(features), k=1)
        return float(np.mean(1.0 - S[iu]))


# ============================================================================
# Quantile/Rank Rewards
# ============================================================================

class QuantileRewardComputer:
    """
    Compute rewards based on percentile ranks rather than mean/std normalization.
    
    More robust to distribution shape and scale variations.
    """
    
    def __init__(
        self,
        top_percentile: float = 0.01,   # Top 1% gets maximum bonus
        high_percentile: float = 0.10,  # Top 10% gets good bonus
        scale: float = 3.0,             # Overall reward scale
    ):
        self.top_percentile = top_percentile
        self.high_percentile = high_percentile
        self.scale = scale
    
    def compute_quantile_scores(
        self, 
        values: np.ndarray,  # (T,)
        sel_idx: List[int]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute quantile-based reward.
        
        Args:
            values: Per-frame quality values
            sel_idx: Selected frame indices
        
        Returns:
            reward: Total quantile reward
            info: Detailed breakdown
        """
        T = len(values)
        if len(sel_idx) == 0 or T == 0:
            return 0.0, {}
        
        sel_idx_valid = [i for i in sel_idx if i < T]
        if len(sel_idx_valid) == 0:
            return 0.0, {}
        
        # Compute percentile rank for each frame
        ranks = np.argsort(np.argsort(values))  # 0 = worst, T-1 = best
        percentiles = ranks / (T - 1) if T > 1 else np.ones(T)  # [0, 1]
        
        # Get percentiles of selected frames
        sel_percentiles = percentiles[sel_idx_valid]
        
        # Count frames in each tier
        n_top = np.sum(sel_percentiles >= (1 - self.top_percentile))
        n_high = np.sum(sel_percentiles >= (1 - self.high_percentile))
        K = len(sel_idx_valid)
        
        # Tier ratios
        top_ratio = n_top / K
        high_ratio = n_high / K
        
        # Mean percentile (should be > 0.5 if selecting quality frames)
        mean_percentile = float(np.mean(sel_percentiles))
        
        # Reward components
        # 1. Base: how much above average (0.5)
        base_reward = (mean_percentile - 0.5) * 2  # [-1, 1]
        
        # 2. Top bonus: extra reward for catching top 1%
        top_bonus = top_ratio * 2.0  # Up to 2.0
        
        # 3. High bonus: reward for top 10%
        high_bonus = high_ratio * 1.0  # Up to 1.0
        
        # Total
        total_reward = (base_reward + top_bonus + high_bonus) * self.scale
        
        info = {
            "mean_percentile": mean_percentile,
            "top_1_percent_ratio": top_ratio,
            "top_10_percent_ratio": high_ratio,
            "base_reward": base_reward,
            "top_bonus": top_bonus,
            "high_bonus": high_bonus,
        }
        
        return float(total_reward), info
    
    def compute_per_attribute_quantile(
        self,
        attrs: np.ndarray,  # (T, 6)
        sel_idx: List[int]
    ) -> Dict[str, float]:
        """Compute quantile score for each attribute"""
        results = {}
        
        for name, idx in ATTR_INDEX.items():
            attr_values = attrs[:, idx]
            reward, _ = self.compute_quantile_scores(attr_values, sel_idx)
            results[f"quantile_{name}"] = reward
        
        return results


# ============================================================================
# Test-Time Scaling
# ============================================================================

class TestTimeScaler:
    """
    Sample multiple summaries from policy → rerank by true objective.
    
    Implements:
    1. Temperature-scaled sampling
    2. Top-p nucleus sampling
    3. Objective-based reranking with constraint penalties
    """
    
    def __init__(self, config: TestTimeScalingConfig):
        self.config = config
    
    def sample_summary(
        self,
        probs: np.ndarray,  # (T,) selection probabilities
        k: int,
    ) -> List[int]:
        """Sample k frames with temperature and nucleus sampling"""
        T = len(probs)
        cfg = self.config
        
        # Temperature scaling
        logits = np.log(probs + 1e-8)
        scaled_logits = logits / cfg.temperature
        scaled_probs = np.exp(scaled_logits)
        scaled_probs = scaled_probs / scaled_probs.sum()
        
        # Nucleus (top-p) sampling
        sorted_idx = np.argsort(scaled_probs)[::-1]
        cumsum = np.cumsum(scaled_probs[sorted_idx])
        cutoff_idx = np.searchsorted(cumsum, cfg.top_p) + 1
        
        # Zero out non-nucleus probabilities
        nucleus_mask = np.zeros(T, dtype=bool)
        nucleus_mask[sorted_idx[:cutoff_idx]] = True
        
        sampling_probs = np.where(nucleus_mask, scaled_probs, 0.0)
        sampling_probs = sampling_probs / sampling_probs.sum()
        
        # Sample k frames
        k = min(k, T)
        try:
            selected = np.random.choice(T, size=k, replace=False, p=sampling_probs)
        except:
            # Fallback to top-k if sampling fails
            selected = np.argsort(probs)[-k:]
        
        return sorted(selected.tolist())
    
    def compute_summary_score(
        self,
        sel_idx: List[int],
        features: np.ndarray,
        anime_attrs: Optional[np.ndarray],
        constraint_config: ConstraintConfig,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute objective score for a summary.
        
        Args:
            sel_idx: Selected frame indices
            features: (T, D) normalized features
            anime_attrs: (T, 6) anime attributes
            constraint_config: Constraint thresholds
        
        Returns:
            score: Total score (higher = better)
            info: Detailed breakdown
        """
        cfg = self.config
        T = len(features)
        
        if len(sel_idx) == 0:
            return -np.inf, {}
        
        sel_idx = [i for i in sel_idx if i < T]
        if len(sel_idx) == 0:
            return -np.inf, {}
        
        feats_sel = features[sel_idx]
        
        # === Compute constraints ===
        # 1. RecErr
        S = features @ feats_sel.T
        min_dist = 1.0 - np.max(S, axis=1)
        rec_err = float(np.mean(min_dist))
        
        # 2. Diversity
        if len(feats_sel) >= 2:
            S_sel = feats_sel @ feats_sel.T
            iu = np.triu_indices(len(feats_sel), k=1)
            diversity = float(np.mean(1.0 - S_sel[iu]))
        else:
            diversity = 0.0
        
        # 3. Coverage (gap variance)
        sorted_idx = sorted(sel_idx)
        if len(sorted_idx) > 1:
            gaps = np.diff(sorted_idx)
            ideal_gap = T / len(sel_idx)
            coverage_gap = float(np.std(gaps) / max(ideal_gap, 1.0))
        else:
            coverage_gap = 0.0
        
        # === Constraint penalties ===
        penalty = 0.0
        if rec_err > constraint_config.rec_err_threshold:
            penalty += cfg.constraint_penalty * (rec_err - constraint_config.rec_err_threshold)
        if coverage_gap > constraint_config.coverage_threshold:
            penalty += cfg.constraint_penalty * (coverage_gap - constraint_config.coverage_threshold)
        if diversity < constraint_config.diversity_threshold:
            penalty += cfg.constraint_penalty * (constraint_config.diversity_threshold - diversity)
        
        # === Primary objective: Anime quality ===
        anime_score = 0.0
        if anime_attrs is not None:
            attrs_sel = anime_attrs[sel_idx]
            mean_all = np.mean(anime_attrs, axis=0)
            mean_sel = np.mean(attrs_sel, axis=0)
            anime_score = float(np.sum(mean_sel - mean_all))  # Total improvement
        
        # Total score
        total_score = anime_score - penalty
        
        info = {
            "anime_score": anime_score,
            "rec_err": rec_err,
            "diversity": diversity,
            "coverage_gap": coverage_gap,
            "penalty": penalty,
        }
        
        return total_score, info
    
    def select_best_summary(
        self,
        probs: np.ndarray,
        features: np.ndarray,
        anime_attrs: Optional[np.ndarray],
        k: int,
        constraint_config: ConstraintConfig,
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Sample N summaries and return the best one.
        
        Returns:
            best_selection: Best frame selection
            info: Statistics about the search
        """
        cfg = self.config
        
        candidates = []
        scores = []
        
        for i in range(cfg.n_samples):
            sel = self.sample_summary(probs, k)
            score, _ = self.compute_summary_score(
                sel, features, anime_attrs, constraint_config
            )
            candidates.append(sel)
            scores.append(score)
        
        # Select best
        best_idx = np.argmax(scores)
        best_selection = candidates[best_idx]
        best_score = scores[best_idx]
        
        # Compute final info
        _, best_info = self.compute_summary_score(
            best_selection, features, anime_attrs, constraint_config
        )
        
        info = {
            "n_samples": cfg.n_samples,
            "best_score": best_score,
            "mean_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            **best_info
        }
        
        return best_selection, info


# ============================================================================
# Main Reward System
# ============================================================================

class PremiumRewardV9:
    """
    V9 Quality-Focused RL Reward System.
    
    Key principles:
    1. Anime quality is PRIMARY objective (increased weight)
    2. NEW: Direct percentile-based reward for maximizing Mean Percentile Rank
    3. RecErr/Coverage/Diversity are soft constraints (reduced penalty)
    4. Quantile rewards for robust outlier detection
    5. No motion features required
    """
    
    def __init__(
        self,
        # Constraint config
        constraint_config: Optional[ConstraintConfig] = None,
        # DPP config
        dpp_config: Optional[DPPConfig] = None,
        # Test-time scaling config
        tts_config: Optional[TestTimeScalingConfig] = None,
        # Reward scales - V9: INCREASED quality focus
        anime_scale: float = 5.0,       # V8: 3.0 → V9: 5.0
        quantile_scale: float = 3.0,    # V8: 2.0 → V9: 3.0
        percentile_scale: float = 2.0,  # NEW in V9
        # Training settings
        use_curriculum: bool = True,
        total_epochs: int = 60,
    ):
        self.constraint_config = constraint_config or ConstraintConfig()
        self.dpp_config = dpp_config or DPPConfig()
        self.tts_config = tts_config or TestTimeScalingConfig()
        
        self.anime_scale = anime_scale
        self.quantile_scale = quantile_scale
        self.percentile_scale = percentile_scale  # NEW in V9
        self.use_curriculum = use_curriculum
        self.total_epochs = total_epochs
        
        # Components
        self.lagrangian = LagrangianOptimizer(self.constraint_config)
        self.two_stage_selector = TwoStageSelector(self.dpp_config)
        self.quantile_computer = QuantileRewardComputer(scale=quantile_scale)
        self.tts_scaler = TestTimeScaler(self.tts_config)
        
        # Tracker
        self.tracker = ConstraintTracker()
        
        # Curriculum stages - V9: shorter warmup, faster quality focus
        self.warmup_epochs = int(0.05 * total_epochs)  # V8: 10% → V9: 5%
    
    def get_curriculum_weights(self, current_epoch: int) -> Dict[str, float]:
        """Get curriculum weights - V9: Higher anime weight throughout"""
        if not self.use_curriculum:
            return {"constraint": 0.5, "anime": 1.5}  # V9: favor anime
        
        # Warmup: still focus some on constraints
        if current_epoch <= self.warmup_epochs:
            progress = current_epoch / max(1, self.warmup_epochs)
            return {
                "constraint": 1.5 - progress * 0.5,  # 1.5 → 1.0 (V8: 2.0 → 1.0)
                "anime": 1.0 + 0.5 * progress,       # 1.0 → 1.5 (V8: 0.5 → 1.0)
            }
        
        # After warmup: full optimization
        return {"constraint": 1.0, "anime": 1.0}
    
    def compute_constraints(
        self,
        feats_all: np.ndarray,
        sel_idx: List[int],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute constraint values.
        
        Returns:
            values: Constraint values (rec_err, coverage_gap, diversity)
            violations: Bool dict of which constraints are violated
        """
        if len(sel_idx) == 0 or len(feats_all) == 0:
            return {}, {}
        
        sel_idx = [i for i in sel_idx if i < len(feats_all)]
        if len(sel_idx) == 0:
            return {}, {}
        
        feats_sel = feats_all[sel_idx]
        T = len(feats_all)
        cfg = self.constraint_config
        
        # 1. RecErr
        S = feats_all @ feats_sel.T
        min_dist = 1.0 - np.max(S, axis=1)
        rec_err = float(np.mean(min_dist))
        
        # 2. Diversity
        if len(feats_sel) >= 2:
            S_sel = feats_sel @ feats_sel.T
            iu = np.triu_indices(len(feats_sel), k=1)
            diversity = float(np.mean(1.0 - S_sel[iu]))
        else:
            diversity = 0.0
        
        # 3. Coverage (gap variance normalized by ideal gap)
        sorted_idx = sorted(sel_idx)
        if len(sorted_idx) > 1:
            gaps = np.diff(sorted_idx)
            ideal_gap = T / len(sel_idx)
            coverage_gap = float(np.std(gaps) / max(ideal_gap, 1.0))
        else:
            coverage_gap = 0.0
        
        values = {
            "rec_err": rec_err,
            "coverage_gap": coverage_gap,
            "diversity": diversity,
        }
        
        violations = {
            "rec_err_violated": rec_err > cfg.rec_err_threshold,
            "coverage_violated": coverage_gap > cfg.coverage_threshold,
            "diversity_violated": diversity < cfg.diversity_threshold,
        }
        
        return values, violations
    
    def compute_percentile_reward(
        self,
        attrs_all: np.ndarray,  # (T, 6)
        sel_idx: List[int],
    ) -> Tuple[float, Dict[str, float]]:
        """
        V9 NEW: Compute reward for selecting high-percentile frames.
        
        This directly optimizes for Mean Percentile Rank metric.
        
        Args:
            attrs_all: (T, 6) anime attributes for all frames
            sel_idx: Selected frame indices
            
        Returns:
            reward: Percentile-based reward (higher = better selection)
            info: Detailed metrics
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return 0.0, {}
        
        sel_idx = [i for i in sel_idx if i < len(attrs_all)]
        if len(sel_idx) == 0:
            return 0.0, {}
        
        T = len(attrs_all)
        
        # Compute aggregate quality per frame
        avg_quality = np.mean(attrs_all, axis=1)  # (T,)
        
        # Compute percentile rank for each frame (0 = worst, 1 = best)
        percentile_ranks = np.array([np.sum(avg_quality <= q) / T for q in avg_quality])
        
        # Get percentile ranks of selected frames
        sel_percentiles = percentile_ranks[sel_idx]
        
        # Key metrics
        mean_percentile = float(np.mean(sel_percentiles))
        median_percentile = float(np.median(sel_percentiles))
        min_percentile = float(np.min(sel_percentiles))
        
        # Top-K coverage
        k10 = max(1, int(T * 0.10))  # Top 10%
        k25 = max(1, int(T * 0.25))  # Top 25%
        top10_idx = set(np.argsort(avg_quality)[-k10:])
        top25_idx = set(np.argsort(avg_quality)[-k25:])
        
        top10_coverage = len(set(sel_idx) & top10_idx) / k10
        top25_coverage = len(set(sel_idx) & top25_idx) / k25
        
        # Reward components:
        # 1. Mean percentile reward (main signal): (mean_perc - 0.5) * 2 to get [-1, 1]
        base_reward = (mean_percentile - 0.5) * 2
        
        # 2. Top-10 coverage bonus
        top10_bonus = top10_coverage * 2.0
        
        # 3. Top-25 coverage bonus  
        top25_bonus = top25_coverage * 1.0
        
        # 4. Penalty for having any low-percentile frame
        low_penalty = max(0, 0.2 - min_percentile) * 1.0  # V9.1: Softer penalty (0.2 threshold, 1.0 weight)
        
        # 5. V9 NEW: Temporal spread penalty (avoid edge-biased selection)
        # Penalize if selected frames are clustered at beginning/end
        sel_positions = np.array(sel_idx) / max(T - 1, 1)  # Normalize to [0, 1]
        mean_position = np.mean(sel_positions)
        position_spread = np.std(sel_positions) if len(sel_positions) > 1 else 0
        
        # Ideal: mean_position ≈ 0.5 (center), spread ≈ 0.3 (well distributed)
        center_deviation = abs(mean_position - 0.5)
        spread_penalty = 0.0
        
        # V9.1 update: Increase penalty significantly (2.0 -> 5.0)
        # Penalize if mean is far from center (edge-biased)
        if center_deviation > 0.15:  # More than 15% off center
            spread_penalty += center_deviation * 5.0
        
        # Penalize if spread is too low (clustered)
        if position_spread < 0.15:  # Very clustered
            spread_penalty += (0.15 - position_spread) * 5.0
        
        # Total reward (with spread penalty)
        total_reward = (base_reward + top10_bonus + top25_bonus - low_penalty - spread_penalty) * self.percentile_scale
        
        info = {
            "mean_percentile": mean_percentile,
            "median_percentile": median_percentile,
            "min_percentile": min_percentile,
            "top10_coverage": top10_coverage,
            "top25_coverage": top25_coverage,
            "percentile_base_reward": base_reward,
            "percentile_top10_bonus": top10_bonus,
            "percentile_top25_bonus": top25_bonus,
            "percentile_low_penalty": low_penalty,
            "temporal_mean_position": mean_position,           # NEW
            "temporal_spread": position_spread,                 # NEW
            "temporal_spread_penalty": spread_penalty,          # NEW
        }
        
        return float(total_reward), info
    
    def compute_anime_reward(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
        use_quantile: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute anime quality reward (PRIMARY objective).
        
        Uses quantile-based rewards for robust outlier hunting.
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return 0.0, {}
        
        sel_idx = [i for i in sel_idx if i < len(attrs_all)]
        if len(sel_idx) == 0:
            return 0.0, {}
        
        # Mean quality for selected vs all
        avg_quality = np.mean(attrs_all, axis=1)
        
        # Quantile-based reward
        if use_quantile:
            quantile_reward, quantile_info = self.quantile_computer.compute_quantile_scores(
                avg_quality, sel_idx
            )
            per_attr_quantile = self.quantile_computer.compute_per_attribute_quantile(
                attrs_all, sel_idx
            )
        else:
            quantile_reward = 0.0
            quantile_info = {}
            per_attr_quantile = {}
        
        # Standard improvement reward
        mean_all = float(np.mean(avg_quality))
        mean_sel = float(np.mean(avg_quality[sel_idx]))
        std_all = float(np.std(avg_quality)) + 1e-6
        improvement = (mean_sel - mean_all) / std_all
        
        # Top-k recall
        T = len(attrs_all)
        k = max(1, int(T * 0.1))  # Top 10%
        top_k_indices = set(np.argsort(avg_quality)[-k:])
        top_k_recall = len(top_k_indices & set(sel_idx)) / k
        
        # Combined reward
        total_reward = (
            quantile_reward * 0.6 + 
            improvement * self.anime_scale * 0.4
        )
        
        info = {
            "improvement": improvement,
            "top_k_recall": top_k_recall,
            "quantile_reward": quantile_reward,
            **quantile_info,
            **per_attr_quantile,
        }
        
        return float(total_reward), info
    
    def compute_reward(
        self,
        feats_all: np.ndarray,
        sel_idx: List[int],
        anime_attrs: Optional[np.ndarray] = None,
        motion: Optional[np.ndarray] = None,
        current_epoch: int = 0,
        update_lagrangian: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute constrained reward.
        
        Primary objective: Anime quality
        Constraints: RecErr, Coverage, Diversity (enforced via Lagrangian)
        
        Returns:
            rewards: Dict with 'anime', 'constraint_penalty', 'total'
            components: Detailed breakdown
        """
        components = {}
        
        # Get curriculum weights
        weights = self.get_curriculum_weights(current_epoch)
        
        # === Compute constraints ===
        constraint_values, violations = self.compute_constraints(feats_all, sel_idx)
        
        if constraint_values:
            # Log to tracker
            self.tracker.log_constraints(
                constraint_values["rec_err"],
                constraint_values["coverage_gap"],
                constraint_values["diversity"],
                self.constraint_config
            )
            
            # Compute Lagrangian penalty
            penalty = self.lagrangian.get_penalty(
                constraint_values["rec_err"],
                constraint_values["coverage_gap"],
                constraint_values["diversity"],
            )
            
            # Update Lagrangian multipliers
            if update_lagrangian:
                lambdas = self.lagrangian.update(
                    constraint_values["rec_err"],
                    constraint_values["coverage_gap"],
                    constraint_values["diversity"],
                )
                self.tracker.log_lambdas(
                    lambdas["lambda_rec"],
                    lambdas["lambda_cov"],
                    lambdas["lambda_div"],
                )
            
            components.update(constraint_values)
            components.update(violations)
            components["constraint_penalty"] = penalty
        else:
            penalty = 0.0
            components["constraint_penalty"] = 0.0
        
        # === Compute anime reward (PRIMARY objective) ===
        if anime_attrs is not None:
            anime_reward, anime_info = self.compute_anime_reward(anime_attrs, sel_idx)
            components.update({f"anime_{k}": v for k, v in anime_info.items()})
            
            # V9 NEW: Compute percentile reward
            percentile_reward, percentile_info = self.compute_percentile_reward(anime_attrs, sel_idx)
            components.update({f"perc_{k}": v for k, v in percentile_info.items()})
            
            # Log to tracker
            self.tracker.log_quality(
                anime_reward,
                anime_info.get("quantile_reward", 0.0),
                anime_info.get("top_k_recall", 0.0),
            )
        else:
            anime_reward = 0.0
            percentile_reward = 0.0
        
        # === Combine === V9: Include percentile reward
        constraint_term = -penalty * weights["constraint"]
        anime_term = anime_reward * weights["anime"]
        percentile_term = percentile_reward * weights["anime"]  # Use anime weight for percentile too
        total_reward = anime_term + percentile_term + constraint_term
        
        rewards = {
            "anime": anime_term,
            "percentile": percentile_term,  # V9 NEW
            "constraint_penalty": penalty,
            "constraint_term": constraint_term,
            "total": total_reward,
        }
        
        components["weights_constraint"] = weights["constraint"]
        components["weights_anime"] = weights["anime"]
        
        return rewards, components
    
    def select_with_dpp(
        self,
        features: np.ndarray,
        anime_attrs: Optional[np.ndarray],
        k: int,
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Two-stage selection using DPP.
        
        Args:
            features: (T, D) normalized features
            anime_attrs: (T, 6) anime attributes
            k: Number of frames to select
        
        Returns:
            selected: List of selected indices
            info: Selection statistics
        """
        # Compute quality scores
        if anime_attrs is not None:
            quality = np.mean(anime_attrs, axis=1)
        else:
            quality = np.ones(len(features))
        
        # Two-stage selection
        selected, info = self.two_stage_selector.select(
            quality, features, k
        )
        
        return selected, info
    
    def select_with_test_time_scaling(
        self,
        probs: np.ndarray,
        features: np.ndarray,
        anime_attrs: Optional[np.ndarray],
        k: int,
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Test-time scaling: sample multiple summaries → select best.
        
        Args:
            probs: (T,) selection probabilities from policy
            features: (T, D) normalized features
            anime_attrs: (T, 6) anime attributes
            k: Number of frames to select
        
        Returns:
            best_selection: Best frame selection
            info: Search statistics
        """
        return self.tts_scaler.select_best_summary(
            probs, features, anime_attrs, k, self.constraint_config
        )
    
    def get_tracker_summary(self) -> Dict[str, float]:
        """Get summary from tracker for TensorBoard logging"""
        return self.tracker.get_summary()
    
    def get_lagrangian_state(self) -> Dict[str, float]:
        """Get Lagrangian multiplier state for checkpointing"""
        return self.lagrangian.state_dict()
    
    def load_lagrangian_state(self, state: Dict[str, float]):
        """Load Lagrangian multiplier state from checkpoint"""
        self.lagrangian.load_state_dict(state)


# ============================================================================
# Factory function
# ============================================================================

def create_reward_system_v8(
    rec_err_threshold: float = 0.35,
    coverage_threshold: float = 0.3,
    diversity_threshold: float = 0.25,
    lambda_lr: float = 0.01,
    dpp_beta: float = 1.0,
    candidate_ratio: float = 0.3,
    tts_n_samples: int = 8,
    tts_temperature: float = 1.2,
    anime_scale: float = 3.0,
    quantile_scale: float = 2.0,
    use_curriculum: bool = True,
    total_epochs: int = 60,
) -> PremiumRewardV8:
    """Factory function to create V8 reward system with custom config"""
    
    constraint_config = ConstraintConfig(
        rec_err_threshold=rec_err_threshold,
        coverage_threshold=coverage_threshold,
        diversity_threshold=diversity_threshold,
        lambda_lr=lambda_lr,
    )
    
    dpp_config = DPPConfig(
        beta=dpp_beta,
        candidate_ratio=candidate_ratio,
    )
    
    tts_config = TestTimeScalingConfig(
        n_samples=tts_n_samples,
        temperature=tts_temperature,
    )
    
    return PremiumRewardV8(
        constraint_config=constraint_config,
        dpp_config=dpp_config,
        tts_config=tts_config,
        anime_scale=anime_scale,
        quantile_scale=quantile_scale,
        use_curriculum=use_curriculum,
        total_epochs=total_epochs,
    )
