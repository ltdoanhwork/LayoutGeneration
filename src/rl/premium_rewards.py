"""
Premium Anime-CLIP-IQA Reward System.

This module implements advanced, academic-grade reward mechanisms to maximize 
Anime-CLIP-IQA scores during RL training. It goes beyond simple linear weighting 
by introducing:

1. Percentile-Based Rewards: Rewarding frames that are in the top K percentile of the video.
2. Curriculum Learning: Dynamically adjusting reward weights during training.
3. Contrastive Rewards: Maximizing the gap between selected and rejected frames.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
from src.rl.rewards import ATTR_INDEX

class PremiumAnimeReward:
    def __init__(self, 
                 percentile_threshold: float = 0.75,
                 contrastive_margin: float = 0.2,
                 use_curriculum: bool = True,
                 total_epochs: int = 20):
        """
        Args:
            percentile_threshold: Top fraction of frames to consider "good" (0.0-1.0).
            contrastive_margin: Minimum desired gap between selected and mean score.
            use_curriculum: Whether to ramp up weights over time.
            total_epochs: Total training epochs for curriculum scheduling.
        """
        self.percentile_threshold = percentile_threshold
        self.contrastive_margin = contrastive_margin
        self.use_curriculum = use_curriculum
        self.total_epochs = total_epochs
        
    def get_curriculum_weight(self, current_epoch: int, base_weight: float) -> float:
        """
        Linearly ramp up weight from 0.1 * base to 1.0 * base over first 50% of training.
        Then keep it constant.
        """
        if not self.use_curriculum:
            return base_weight
            
        progress = min(1.0, current_epoch / (0.5 * self.total_epochs))
        # Start at 20% strength, ramp to 100%
        scale = 0.2 + 0.8 * progress
        return base_weight * scale

    def compute_reward(self, 
                       attrs_all: np.ndarray, 
                       sel_idx: List[int],
                       current_epoch: int = 0) -> Dict[str, float]:
        """
        Compute premium rewards.
        
        Args:
            attrs_all: (T, K) Anime-CLIP IQA scores.
            sel_idx: Selected indices.
            current_epoch: Current training epoch.
            
        Returns:
            Dict with 'look', 'sakuga', 'story' rewards.
        """
        if len(sel_idx) == 0 or attrs_all.shape[0] == 0:
            return {"look": 0.0, "sakuga": 0.0, "story": 0.0}
            
        # 1. Extract raw scores
        # Look = (Sharpness + Colorfulness + Brightness) / 3
        look_scores = (attrs_all[:, ATTR_INDEX["sharpness"]] + 
                       attrs_all[:, ATTR_INDEX["colorfulness"]] + 
                       attrs_all[:, ATTR_INDEX["brightness"]]) / 3.0
                       
        sakuga_scores = attrs_all[:, ATTR_INDEX["sakuga"]]
        
        # Story = (Sakuga + Cinematic) / 2
        story_scores = (attrs_all[:, ATTR_INDEX["sakuga"]] + 
                        attrs_all[:, ATTR_INDEX["cinematic"]]) / 2.0
        
        # 2. Compute Percentile Thresholds for this video
        # We want selected frames to be in the top (1-percentile) of the video
        look_thresh = np.quantile(look_scores, self.percentile_threshold)
        sakuga_thresh = np.quantile(sakuga_scores, self.percentile_threshold)
        story_thresh = np.quantile(story_scores, self.percentile_threshold)
        
        # 3. Compute Rewards
        # R = Mean(Selected > Threshold) - Penalty(Selected < Threshold)
        
        sel_look = look_scores[sel_idx]
        sel_sakuga = sakuga_scores[sel_idx]
        sel_story = story_scores[sel_idx]
        
        # "Premium" Reward: 
        # +1 for every frame above threshold
        # -1 for every frame below threshold
        # This forces the agent to pick ONLY the best frames.
        
        def compute_component(selected, threshold):
            good = (selected >= threshold).sum()
            bad = (selected < threshold).sum()
            total = len(selected)
            if total == 0: return 0.0
            # Normalized score in [-1, 1]
            return float((good - bad) / total)

        R_look = compute_component(sel_look, look_thresh)
        R_sakuga = compute_component(sel_sakuga, sakuga_thresh)
        R_story = compute_component(sel_story, story_thresh)
        
        # 4. Contrastive Boost (Optional)
        # Bonus if mean(selected) is significantly higher than mean(all)
        if sel_look.mean() > look_scores.mean() + self.contrastive_margin:
            R_look += 0.5
            
        return {
            "look": R_look,
            "sakuga": R_sakuga,
            "story": R_story
        }
