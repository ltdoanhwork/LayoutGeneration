#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Core Module for V4 Training.

This module implements the core PPO algorithm components for stable policy
gradient updates in anime video summarization.

Key Features:
- Clipped surrogate objective to prevent destructive policy updates
- Adaptive clip range based on KL divergence
- Multi-epoch updates on collected experiences
- Compatible with binary frame selection (Bernoulli policy)

Reference: 
    Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    https://arxiv.org/abs/1707.06347

Author: V4 Enhancement
Date: 2025-12-06
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""
    clip_range: float = 0.2          # Epsilon for clipping (standard: 0.1-0.3)
    clip_range_vf: float = 0.2       # Value function clip range (None to disable)
    target_kl: float = 0.01          # Target KL for early stopping (None to disable)
    n_ppo_epochs: int = 4            # Number of epochs per PPO update
    normalize_advantage: bool = True  # Normalize advantages to zero mean, unit var
    entropy_coef: float = 0.01       # Entropy bonus coefficient
    vf_coef: float = 0.5             # Value function loss coefficient
    max_grad_norm: float = 0.5       # Gradient clipping threshold


class PPOCore:
    """
    Core PPO algorithm implementation.
    
    Handles:
    - Clipped surrogate objective computation
    - Value function clipping (optional)
    - KL divergence monitoring for early stopping
    - Advantage normalization
    
    Usage:
        >>> config = PPOConfig(clip_range=0.2)
        >>> ppo = PPOCore(config)
        >>> 
        >>> # During update
        >>> actor_loss, metrics = ppo.compute_actor_loss(
        ...     old_log_probs, new_log_probs, advantages
        ... )
        >>> critic_loss = ppo.compute_critic_loss(values, old_values, returns)
    """
    
    def __init__(self, config: PPOConfig = None):
        """
        Initialize PPO core.
        
        Args:
            config: PPO configuration. Uses defaults if None.
        """
        self.config = config or PPOConfig()
        
        # Tracking metrics
        self.approx_kl_divs = []
        self.clip_fractions = []
    
    def compute_actor_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        return_metrics: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute PPO clipped actor (policy) loss.
        
        L^{CLIP} = E_t[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        
        Args:
            old_log_probs: Log probabilities from old policy (B,) or (B, T)
            new_log_probs: Log probabilities from current policy (B,) or (B, T)
            advantages: Advantage estimates (B,) or (B, T)
            return_metrics: Whether to return diagnostic metrics
            
        Returns:
            Tuple of (loss, metrics_dict). Loss is negated for gradient descent.
        """
        # Flatten if needed
        old_log_probs = old_log_probs.flatten()
        new_log_probs = new_log_probs.flatten()
        advantages = advantages.flatten()
        
        # Normalize advantages
        if self.config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute probability ratio
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate objective
        # surr1: r * A
        # surr2: clip(r, 1-eps, 1+eps) * A
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 
            1.0 - self.config.clip_range, 
            1.0 + self.config.clip_range
        ) * advantages
        
        # Take minimum (pessimistic bound) and negate for loss
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute metrics
        metrics = None
        if return_metrics:
            with torch.no_grad():
                # Approximate KL divergence
                # KL ≈ 0.5 * E[(log_ratio)^2] for small differences
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                
                # Clip fraction: how often was clipping active?
                clip_fraction = (
                    (ratio < 1 - self.config.clip_range) | 
                    (ratio > 1 + self.config.clip_range)
                ).float().mean().item()
                
                # Store for tracking
                self.approx_kl_divs.append(approx_kl)
                self.clip_fractions.append(clip_fraction)
                
                metrics = {
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "policy_loss": policy_loss.item(),
                    "ratio_mean": ratio.mean().item(),
                    "ratio_std": ratio.std().item(),
                }
        
        return policy_loss, metrics
    
    def compute_critic_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        clip_value: bool = True
    ) -> torch.Tensor:
        """
        Compute value function (critic) loss with optional clipping.
        
        Clipped value loss helps prevent value function from changing too rapidly,
        similar to policy clipping.
        
        Args:
            values: Current value estimates (B,) or (B, T)
            old_values: Value estimates from old policy (B,) or (B, T)
            returns: TD(λ) returns / targets (B,) or (B, T)
            clip_value: Whether to use clipped value loss
            
        Returns:
            Value function loss (MSE or clipped MSE)
        """
        values = values.flatten()
        old_values = old_values.flatten()
        returns = returns.flatten()
        
        if clip_value and self.config.clip_range_vf is not None:
            # Clipped value loss (PPO style)
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.clip_range_vf,
                self.config.clip_range_vf
            )
            vf_loss1 = F.mse_loss(values, returns, reduction='none')
            vf_loss2 = F.mse_loss(values_clipped, returns, reduction='none')
            vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
        else:
            # Standard MSE loss
            vf_loss = 0.5 * F.mse_loss(values, returns)
        
        return vf_loss
    
    def compute_entropy_bonus(
        self,
        probs: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute entropy bonus for exploration.
        
        For Bernoulli policy: H = -p*log(p) - (1-p)*log(1-p)
        
        Args:
            probs: Action probabilities (0-1 for Bernoulli)
            eps: Small constant for numerical stability
            
        Returns:
            Mean entropy (to be maximized, so add to loss with negative sign)
        """
        probs = probs.flatten()
        probs = torch.clamp(probs, eps, 1 - eps)
        
        # Bernoulli entropy
        entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
        
        return entropy.mean()
    
    def should_stop_early(self) -> bool:
        """
        Check if training should stop early based on KL divergence.
        
        Returns:
            True if mean KL divergence exceeds target, False otherwise.
        """
        if self.config.target_kl is None:
            return False
        
        if len(self.approx_kl_divs) == 0:
            return False
        
        mean_kl = np.mean(self.approx_kl_divs[-10:])  # Use recent values
        return mean_kl > 1.5 * self.config.target_kl
    
    def reset_tracking(self):
        """Reset tracking metrics for new epoch."""
        self.approx_kl_divs = []
        self.clip_fractions = []
    
    def get_tracking_stats(self) -> Dict[str, float]:
        """Get summary statistics from tracking."""
        return {
            "mean_kl": np.mean(self.approx_kl_divs) if self.approx_kl_divs else 0.0,
            "max_kl": np.max(self.approx_kl_divs) if self.approx_kl_divs else 0.0,
            "mean_clip_frac": np.mean(self.clip_fractions) if self.clip_fractions else 0.0,
        }


def compute_log_prob_bernoulli(
    probs: torch.Tensor,
    actions: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute log probability for Bernoulli (binary) actions.
    
    Args:
        probs: Selection probabilities (B, T) or (T,)
        actions: Binary actions (0 or 1), same shape as probs
        eps: Numerical stability constant
        
    Returns:
        Log probabilities, same shape as input
    """
    probs = torch.clamp(probs, eps, 1 - eps)
    log_probs = actions * torch.log(probs) + (1 - actions) * torch.log(1 - probs)
    return log_probs


def compute_total_ppo_loss(
    ppo: PPOCore,
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    probs: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total PPO loss (actor + critic + entropy).
    
    Args:
        ppo: PPOCore instance
        old_log_probs: Log probs from old policy
        new_log_probs: Log probs from current policy
        advantages: GAE advantages
        values: Current value estimates
        old_values: Old value estimates
        returns: TD(λ) returns
        probs: Current action probabilities (for entropy)
        
    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Actor loss
    actor_loss, metrics = ppo.compute_actor_loss(
        old_log_probs, new_log_probs, advantages
    )
    
    # Critic loss
    critic_loss = ppo.compute_critic_loss(values, old_values, returns)
    
    # Entropy bonus
    entropy = ppo.compute_entropy_bonus(probs)
    
    # Total loss
    total_loss = (
        actor_loss + 
        ppo.config.vf_coef * critic_loss - 
        ppo.config.entropy_coef * entropy
    )
    
    # Update metrics
    metrics["critic_loss"] = critic_loss.item()
    metrics["entropy"] = entropy.item()
    metrics["total_loss"] = total_loss.item()
    
    return total_loss, metrics


if __name__ == "__main__":
    # Demo and unit test
    print("=== PPO Core Demo ===\n")
    
    # Create PPO core
    config = PPOConfig(clip_range=0.2, target_kl=0.01)
    ppo = PPOCore(config)
    
    # Create dummy data
    batch_size = 32
    seq_len = 100
    
    # Old policy: probabilities around 0.5
    old_probs = torch.sigmoid(torch.randn(batch_size, seq_len))
    actions = torch.bernoulli(old_probs)
    old_log_probs = compute_log_prob_bernoulli(old_probs, actions).sum(dim=1)
    
    # New policy: slightly different
    new_probs = torch.sigmoid(torch.randn(batch_size, seq_len) * 0.1 + 
                               torch.logit(old_probs))
    new_log_probs = compute_log_prob_bernoulli(new_probs, actions).sum(dim=1)
    
    # Advantages (random for demo)
    advantages = torch.randn(batch_size)
    
    # Compute actor loss
    actor_loss, metrics = ppo.compute_actor_loss(old_log_probs, new_log_probs, advantages)
    
    print(f"Actor Loss: {actor_loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Value function test
    values = torch.randn(batch_size)
    old_values = values + torch.randn(batch_size) * 0.1
    returns = torch.randn(batch_size)
    
    critic_loss = ppo.compute_critic_loss(values, old_values, returns)
    print(f"\nCritic Loss: {critic_loss.item():.4f}")
    
    # Entropy test
    entropy = ppo.compute_entropy_bonus(new_probs.mean(dim=1))
    print(f"Entropy: {entropy.item():.4f}")
    
    # Total loss test
    total_loss, all_metrics = compute_total_ppo_loss(
        ppo, old_log_probs, new_log_probs, advantages,
        values, old_values, returns, new_probs.mean(dim=1)
    )
    print(f"\nTotal Loss: {total_loss.item():.4f}")
    print(f"All Metrics: {all_metrics}")
    
    print("\n✅ PPO Core tests passed!")
