#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Premium Rewards V10: VLM-Guided Quality Optimization

Extends V9 with:
1. Hybrid reward: CLIP-IQA + VLM Distillation scores
2. VLM Distillation Loss for training the student head
3. Adaptive weighting: High VLM guidance early, Standalone student later
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from src.rl.premium_rewards_v9 import PremiumRewardV9, ConstraintConfig, DPPConfig, TestTimeScalingConfig
from src.models.vlm_distiller import VLMDistiller

class PremiumRewardV10(PremiumRewardV9):
    """
    V10 Reward System: Distilling VLM knowledge into CLIP-based agents.
    """
    
    def __init__(
        self,
        # V9 configs
        constraint_config: Optional[ConstraintConfig] = None,
        dpp_config: Optional[DPPConfig] = None,
        tts_config: Optional[TestTimeScalingConfig] = None,
        # V10 specific scales
        vlm_scale: float = 4.0,           # Scale for VLM-based percentile reward
        distillation_weight: float = 1.0, # Weight for distillation loss
        vlm_decay_epochs: int = 40,       # Epochs to phase out VLM guidance
        num_vlm_dims: int = 8,
        **kwargs
    ):
        super().__init__(
            constraint_config=constraint_config,
            dpp_config=dpp_config,
            tts_config=tts_config,
            **kwargs
        )
        
        self.vlm_scale = vlm_scale
        self.distillation_weight = distillation_weight
        self.vlm_decay_epochs = vlm_decay_epochs
        self.num_vlm_dims = num_vlm_dims
        
        # Student head (distiller) - to be trained during RL
        self.distiller = VLMDistiller(input_dim=512, n_quality_dims=num_vlm_dims).cuda()
        self.distiller_optimizer = torch.optim.Adam(self.distiller.parameters(), lr=1e-4)

    def compute_vlm_percentile_reward(
        self,
        vlm_scores_all: np.ndarray,  # (T, 8) precomputed VLM scores
        sel_idx: List[int]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward based on VLM-predicted quality percentiles.
        Uses the same percentile rank logic as V9 but on VLM scores.
        """
        if len(sel_idx) == 0 or len(vlm_scores_all) == 0:
            return 0.0, {}
            
        T = len(vlm_scores_all)
        # Aggregate 8 dims into 1 for ranking (mean)
        vlm_agg = np.mean(vlm_scores_all, axis=1) # (T,)
        
        # Rank: 0 to 1
        ranks = np.argsort(np.argsort(vlm_agg))
        percentiles = ranks / max(1, T - 1)
        
        sel_percentiles = percentiles[sel_idx]
        mean_p = float(np.mean(sel_percentiles))
        
        # Reward components
        base_reward = (mean_p - 0.5) * 2 # [-1, 1]
        
        # Top-K recall (on VLM scores)
        k10 = max(1, int(T * 0.1))
        top10_idx = set(np.argsort(vlm_agg)[-k10:])
        top10_cov = len(set(sel_idx) & top10_idx) / k10
        
        total_reward = (base_reward + top10_cov * 2.0) * self.vlm_scale
        
        info = {
            "vlm_mean_percentile": mean_p,
            "vlm_top10_coverage": top10_cov,
            "vlm_base_reward": base_reward,
            "vlm_total_reward": total_reward
        }
        
        return total_reward, info

    def train_distiller(
        self, 
        clip_features: torch.Tensor, # (T, 512)
        vlm_targets: torch.Tensor    # (T, 8) 
    ) -> float:
        """One step of distillation training."""
        self.distiller.train()
        self.distiller_optimizer.zero_grad()
        
        preds = self.distiller(clip_features)
        loss = torch.nn.functional.mse_loss(preds, vlm_targets)
        
        loss.backward()
        self.distiller_optimizer.step()
        
        return loss.item()

    def compute_reward(
        self,
        feats_all: np.ndarray,
        sel_idx: List[int],
        anime_attrs: Optional[np.ndarray] = None,
        vlm_scores: Optional[np.ndarray] = None, # NEW: Precomputed VLM scores
        current_epoch: int = 0,
        update_lagrangian: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute V10 reward: V9 Rewards + VLM percentile reward.
        """
        # 1. Get baseline V9 rewards (Constraints + CLIP Anime + CLIP Percentile)
        rewards, components = super().compute_reward(
            feats_all=feats_all,
            sel_idx=sel_idx,
            anime_attrs=anime_attrs,
            current_epoch=current_epoch,
            update_lagrangian=update_lagrangian
        )
        
        # 2. Add VLM-based reward if available
        if vlm_scores is not None:
            vlm_reward, vlm_info = self.compute_vlm_percentile_reward(vlm_scores, sel_idx)
            
            # Curriculum: Weight VLM reward more in early epochs
            vlm_weight = max(0.0, 1.0 - (current_epoch / self.vlm_decay_epochs))
            
            # Add to total
            rewards["vlm"] = vlm_reward * vlm_weight
            rewards["total"] += rewards["vlm"]
            
            components.update({f"vlm_{k}": v in vlm_info.items()})
            components["vlm_weight"] = vlm_weight
            
        return rewards, components
