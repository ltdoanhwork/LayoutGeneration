"""
Premium Rewards V5: Dual-Objective Optimization (RecErr + Anime Quality)

Key improvements over V4:
1. Enhanced RecErr rewards with coverage and diversity bonuses
2. Continuous quality rewards for both RecErr and Anime
3. Per-attribute optimization for all metrics
4. Dual-objective balance with adaptive weighting
5. Comprehensive tracking for all metrics
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

# Anime-CLIP-IQA attribute indices
ATTR_NAMES = ["sharpness", "colorfulness", "brightness", "sakuga", "cinematic", "expression"]
ATTR_INDEX = {name: i for i, name in enumerate(ATTR_NAMES)}


@dataclass
class DualObjectiveTracker:
    """Track both RecErr and Anime quality improvements"""
    
    # RecErr-related
    rec_err_history: List[float] = field(default_factory=list)
    frechet_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    coverage_history: List[float] = field(default_factory=list)
    
    # Anime-related
    per_attr_history: Dict[str, List[float]] = field(default_factory=lambda: {n: [] for n in ATTR_NAMES})
    quality_improvement_history: List[float] = field(default_factory=list)
    top_k_recall_history: List[float] = field(default_factory=list)
    top_k_precision_history: List[float] = field(default_factory=list)
    
    def log_rec_metrics(self, rec_err: float, frechet: float, diversity: float, coverage: float):
        self.rec_err_history.append(rec_err)
        self.frechet_history.append(frechet)
        self.diversity_history.append(diversity)
        self.coverage_history.append(coverage)
    
    def log_anime_metrics(self, improvements: Dict[str, float], top_k_recall: float, top_k_precision: float):
        for name, val in improvements.items():
            if name in self.per_attr_history:
                self.per_attr_history[name].append(val)
        total_improvement = sum(improvements.values()) / len(improvements) if improvements else 0.0
        self.quality_improvement_history.append(total_improvement)
        self.top_k_recall_history.append(top_k_recall)
        self.top_k_precision_history.append(top_k_precision)
    
    def get_summary(self, last_n: int = 100) -> Dict[str, float]:
        """Get summary statistics for logging"""
        def safe_mean(lst):
            if not lst:
                return 0.0
            return float(np.mean(lst[-last_n:]))
        
        summary = {
            "rec_err_mean": safe_mean(self.rec_err_history),
            "frechet_mean": safe_mean(self.frechet_history),
            "diversity_mean": safe_mean(self.diversity_history),
            "coverage_mean": safe_mean(self.coverage_history),
            "quality_improvement": safe_mean(self.quality_improvement_history),
            "top_k_recall": safe_mean(self.top_k_recall_history),
            "top_k_precision": safe_mean(self.top_k_precision_history),
        }
        
        for name in ATTR_NAMES:
            summary[f"attr_{name}"] = safe_mean(self.per_attr_history.get(name, []))
        
        return summary


class PremiumRewardV5:
    """
    V5 Reward System: Dual-Objective Optimization
    
    Improves BOTH:
    - RecErr/Frechet (reconstruction quality)
    - Anime quality (CLIP-IQA attributes)
    
    Key features:
    1. Continuous RecErr rewards (not just binary)
    2. Coverage-based bonuses for RecErr
    3. Diversity rewards for better representation
    4. Per-attribute anime quality optimization
    5. Adaptive weighting based on training stage
    """
    
    def __init__(
        self,
        # RecErr settings
        rec_err_scale: float = 3.0,
        frechet_scale: float = 2.0,
        diversity_weight: float = 1.0,
        coverage_weight: float = 1.0,
        # Anime settings
        anime_scale: float = 2.5,
        top_k_ratio: float = 0.1,
        per_attr_weights: Optional[Dict[str, float]] = None,
        # Training settings
        use_curriculum: bool = True,
        total_epochs: int = 60,
        use_adaptive_weighting: bool = True,
    ):
        self.rec_err_scale = rec_err_scale
        self.frechet_scale = frechet_scale
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight
        
        self.anime_scale = anime_scale
        self.top_k_ratio = top_k_ratio
        self.per_attr_weights = per_attr_weights or {name: 1.0 for name in ATTR_NAMES}
        
        self.use_curriculum = use_curriculum
        self.total_epochs = total_epochs
        self.use_adaptive_weighting = use_adaptive_weighting
        
        # Curriculum stages
        self.stage1_end = int(0.2 * total_epochs)  # RecErr focus
        self.stage2_end = int(0.5 * total_epochs)  # Balance
        self.stage3_end = total_epochs              # Full optimization
        
        # Tracker
        self.tracker = DualObjectiveTracker()
        
        # Running statistics for normalization
        self.rec_err_running_mean = 0.5
        self.rec_err_running_std = 0.2
        self.anime_running_mean = 0.0
        self.anime_running_std = 0.1
        self.ema_alpha = 0.01
    
    def get_curriculum_weights(self, current_epoch: int) -> Dict[str, float]:
        """Get weights for RecErr vs Anime based on curriculum stage"""
        if not self.use_curriculum:
            return {"rec": 1.0, "anime": 1.0}
        
        # Stage 1: Focus on RecErr (foundation)
        if current_epoch <= self.stage1_end:
            progress = current_epoch / max(1, self.stage1_end)
            return {
                "rec": 1.0,
                "anime": 0.2 + 0.3 * progress,  # Gradually introduce anime
            }
        
        # Stage 2: Balance both objectives
        elif current_epoch <= self.stage2_end:
            progress = (current_epoch - self.stage1_end) / max(1, self.stage2_end - self.stage1_end)
            return {
                "rec": 1.0,
                "anime": 0.5 + 0.5 * progress,  # Ramp up to full
            }
        
        # Stage 3: Full optimization
        else:
            return {"rec": 1.0, "anime": 1.0}
    
    # ==================== RecErr Rewards ====================
    
    def compute_rec_err_reward(
        self,
        feats_all: np.ndarray,  # (T, D) normalized features
        sel_idx: List[int],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute continuous RecErr reward
        
        Returns:
            reward: Total RecErr reward
            components: Individual components for logging
        """
        if len(sel_idx) == 0 or len(feats_all) == 0:
            return 0.0, {}
        
        sel_idx_valid = [i for i in sel_idx if i < len(feats_all)]
        if len(sel_idx_valid) == 0:
            return 0.0, {}
        
        feats_sel = feats_all[sel_idx_valid]
        
        # 1. RecErr: Mean nearest-neighbor distance (lower is better)
        # Cosine distance = 1 - cosine_sim
        S = feats_all @ feats_sel.T  # (T, K)
        min_dist = 1.0 - np.max(S, axis=1)  # (T,)
        rec_err = float(np.mean(min_dist))
        
        # Convert to reward: lower RecErr = higher reward
        # Normalize and invert
        rec_err_reward = -rec_err * self.rec_err_scale
        
        # 2. Diversity: Mean pairwise distance among selected (higher is better)
        if len(feats_sel) >= 2:
            S_sel = feats_sel @ feats_sel.T
            iu = np.triu_indices(len(feats_sel), k=1)
            pairwise_dist = 1.0 - S_sel[iu]
            diversity = float(np.mean(pairwise_dist))
        else:
            diversity = 0.0
        
        diversity_reward = diversity * self.diversity_weight
        
        # 3. Coverage: Temporal spread of selected frames
        T = len(feats_all)
        if T > 1:
            sorted_idx = sorted(sel_idx_valid)
            # Ideal spacing
            ideal_spacing = T / len(sel_idx_valid)
            actual_gaps = np.diff(sorted_idx)
            gap_variance = float(np.std(actual_gaps)) if len(actual_gaps) > 0 else 0.0
            # Lower variance = more uniform coverage = higher reward
            coverage = 1.0 / (1.0 + gap_variance / ideal_spacing)
        else:
            coverage = 1.0
        
        coverage_reward = coverage * self.coverage_weight
        
        # 4. Representativeness: How well selected frames represent all frames
        # Already captured in rec_err, but add explicit bonus
        rep_bonus = 0.0
        if rec_err < 0.3:  # Good reconstruction
            rep_bonus = 0.5 * (0.3 - rec_err)
        
        # Total reward
        total_reward = rec_err_reward + diversity_reward + coverage_reward + rep_bonus
        
        components = {
            "rec_err": rec_err,
            "rec_err_reward": rec_err_reward,
            "diversity": diversity,
            "diversity_reward": diversity_reward,
            "coverage": coverage,
            "coverage_reward": coverage_reward,
            "rep_bonus": rep_bonus,
        }
        
        # Log to tracker
        self.tracker.log_rec_metrics(rec_err, 0.0, diversity, coverage)
        
        return float(total_reward), components
    
    def compute_frechet_reward(
        self,
        feats_all: np.ndarray,
        sel_idx: List[int],
        eps: float = 1e-6,
    ) -> float:
        """
        Compute Frechet distance-based reward
        Lower Frechet distance = higher reward
        """
        if len(sel_idx) < 2 or len(feats_all) < 2:
            return 0.0
        
        sel_idx_valid = [i for i in sel_idx if i < len(feats_all)]
        if len(sel_idx_valid) < 2:
            return 0.0
        
        feats_sel = feats_all[sel_idx_valid]
        
        # Frechet distance between all frames and selected frames
        mu1 = np.mean(feats_all, axis=0)
        mu2 = np.mean(feats_sel, axis=0)
        
        diff = mu1 - mu2
        diff2 = float(np.dot(diff, diff))
        
        # Covariance matrices
        try:
            S1 = np.cov(feats_all, rowvar=False) + np.eye(feats_all.shape[1]) * eps
            S2 = np.cov(feats_sel, rowvar=False) + np.eye(feats_sel.shape[1]) * eps
            
            cov_prod = S1 @ S2
            cov_prod = (cov_prod + cov_prod.T) * 0.5
            
            eigvals, eigvecs = np.linalg.eigh(cov_prod)
            eigvals[eigvals < 0] = 0.0
            sqrt_cov_prod = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            trace = np.trace(S1 + S2 - 2.0 * sqrt_cov_prod)
            frechet = float(diff2 + trace)
        except:
            frechet = diff2
        
        # Convert to reward (negative, scaled)
        frechet_reward = -frechet * self.frechet_scale / 1000.0  # Scale down
        
        return frechet_reward
    
    # ==================== Anime Quality Rewards ====================
    
    def compute_per_attribute_improvement(
        self,
        attrs_all: np.ndarray,  # (T, 6)
        sel_idx: List[int],
    ) -> Dict[str, float]:
        """Compute quality improvement for each attribute"""
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
            std_all = float(np.std(attr_all)) + 1e-6
            
            # Normalized improvement
            improvement = (mean_sel - mean_all) / std_all
            improvements[name] = improvement * self.per_attr_weights.get(name, 1.0)
        
        return improvements
    
    def compute_top_k_coverage(
        self,
        attrs_all: np.ndarray,
        sel_idx: List[int],
    ) -> Tuple[float, float]:
        """
        Compute how well selected frames cover top-k quality frames
        
        Returns:
            recall: Fraction of top-k frames that were selected
            precision: Fraction of selected frames that are in top-k
        """
        T = len(attrs_all)
        k = max(1, int(T * self.top_k_ratio))
        
        # Average quality score
        avg_quality = np.mean(attrs_all, axis=1)
        
        # Find top-k frame indices
        top_k_indices = set(np.argsort(avg_quality)[-k:])
        
        # Calculate recall and precision
        sel_set = set(sel_idx)
        
        true_positives = len(top_k_indices & sel_set)
        
        recall = true_positives / k if k > 0 else 0.0
        precision = true_positives / len(sel_set) if len(sel_set) > 0 else 0.0
        
        return float(recall), float(precision)
    
    def compute_anime_reward(
        self,
        attrs_all: np.ndarray,  # (T, 6)
        sel_idx: List[int],
        motion: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total anime quality reward
        
        Returns:
            reward: Total anime reward
            components: Individual components for logging  
        """
        if len(sel_idx) == 0 or len(attrs_all) == 0:
            return 0.0, {}
        
        # Per-attribute improvements
        improvements = self.compute_per_attribute_improvement(attrs_all, sel_idx)
        
        # Top-k coverage
        top_k_recall, top_k_precision = self.compute_top_k_coverage(attrs_all, sel_idx)
        
        # Combine attribute improvements
        attr_reward = sum(improvements.values())
        
        # Top-k bonus
        top_k_reward = (top_k_recall + top_k_precision) * 0.5
        
        # Motion bonus (if available)
        motion_bonus = 0.0
        if motion is not None and len(motion) > 0:
            sel_idx_valid = [i for i in sel_idx if i < len(motion)]
            if sel_idx_valid:
                motion_sel = motion[sel_idx_valid]
                motion_all = motion
                motion_improvement = (np.mean(motion_sel) - np.mean(motion_all))
                motion_std = np.std(motion_all) + 1e-6
                motion_bonus = motion_improvement / motion_std * 0.5
        
        # Total anime reward
        total_reward = (attr_reward + top_k_reward + motion_bonus) * self.anime_scale
        
        components = {
            "attr_reward": attr_reward,
            "top_k_recall": top_k_recall,
            "top_k_precision": top_k_precision,
            "top_k_reward": top_k_reward,
            "motion_bonus": motion_bonus,
        }
        components.update({f"improvement_{k}": v for k, v in improvements.items()})
        
        # Log to tracker
        self.tracker.log_anime_metrics(improvements, top_k_recall, top_k_precision)
        
        return float(total_reward), components
    
    # ==================== Combined Reward ====================
    
    def compute_reward(
        self,
        feats_all: np.ndarray,
        sel_idx: List[int],
        anime_attrs: Optional[np.ndarray] = None,
        motion: Optional[np.ndarray] = None,
        current_epoch: int = 0,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute combined reward for both RecErr and Anime objectives
        
        Returns:
            rewards: Dict with 'rec' and 'anime' rewards
            components: Detailed breakdown for logging
        """
        components = {}
        
        # Get curriculum weights
        weights = self.get_curriculum_weights(current_epoch)
        
        # RecErr reward
        rec_reward, rec_components = self.compute_rec_err_reward(feats_all, sel_idx)
        frechet_reward = self.compute_frechet_reward(feats_all, sel_idx)
        
        total_rec_reward = (rec_reward + frechet_reward) * weights["rec"]
        
        components["rec_reward"] = rec_reward
        components["frechet_reward"] = frechet_reward
        components["rec_weight"] = weights["rec"]
        components.update({f"rec_{k}": v for k, v in rec_components.items()})
        
        # Anime reward
        if anime_attrs is not None:
            anime_reward, anime_components = self.compute_anime_reward(
                anime_attrs, sel_idx, motion
            )
            total_anime_reward = anime_reward * weights["anime"]
            
            components["anime_reward"] = anime_reward
            components["anime_weight"] = weights["anime"]
            components.update({f"anime_{k}": v for k, v in anime_components.items()})
        else:
            total_anime_reward = 0.0
            components["anime_reward"] = 0.0
        
        rewards = {
            "rec": total_rec_reward,
            "anime": total_anime_reward,
            "total": total_rec_reward + total_anime_reward,
        }
        
        return rewards, components
    
    def get_tracker_summary(self) -> Dict[str, float]:
        """Get summary from tracker for logging"""
        return self.tracker.get_summary()


def compute_quality_metrics_for_eval(
    attrs_all: np.ndarray,
    sel_idx: List[int],
    top_k_ratio: float = 0.1,
) -> Dict[str, float]:
    """
    Standalone function to compute quality metrics for evaluation
    Matches training rewards for fair comparison
    """
    metrics = {}
    
    # Per-attribute means
    for name, idx in ATTR_INDEX.items():
        attr_sel = attrs_all[sel_idx, idx] if len(sel_idx) > 0 else np.array([])
        metrics[f"Anime_{name.capitalize()}_Mean"] = float(np.mean(attr_sel)) if len(attr_sel) > 0 else 0.0
    
    # Quality improvement
    if len(sel_idx) > 0:
        avg_quality = np.mean(attrs_all, axis=1)
        mean_all = float(np.mean(avg_quality))
        mean_sel = float(np.mean(avg_quality[sel_idx]))
        metrics["Quality_Improvement"] = mean_sel - mean_all
    else:
        metrics["Quality_Improvement"] = 0.0
    
    # Top-k coverage
    T = len(attrs_all)
    k = max(1, int(T * top_k_ratio))
    avg_quality = np.mean(attrs_all, axis=1)
    top_k_indices = set(np.argsort(avg_quality)[-k:])
    sel_set = set(sel_idx)
    
    true_positives = len(top_k_indices & sel_set)
    metrics["Top10_Recall"] = true_positives / k if k > 0 else 0.0
    metrics["Top10_Precision"] = true_positives / len(sel_set) if len(sel_set) > 0 else 0.0
    
    return metrics
