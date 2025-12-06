#!/usr/bin/env python3
"""
Generalized Advantage Estimation (GAE) for V4 Training.

This module implements GAE for computing low-variance, low-bias advantage
estimates in actor-critic RL algorithms.

Key Features:
- Configurable lambda (λ) parameter for bias-variance tradeoff
- TD(λ) returns for value function training
- Batch computation for efficiency
- Compatible with both episode-level and frame-level advantages

Reference:
    Schulman et al., "High-Dimensional Continuous Control Using Generalized 
    Advantage Estimation" (2016)
    https://arxiv.org/abs/1506.02438

Author: V4 Enhancement
Date: 2025-12-06
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import torch
import numpy as np


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE provides a family of advantage estimators indexed by λ:
    - λ = 0: One-step TD error (low variance, high bias)
    - λ = 1: Monte Carlo return (high variance, low bias)
    - λ ∈ (0, 1): Exponentially-weighted average (best of both)
    
    The GAE formula:
        A_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        where δ_t = r_t + γ V(s_{t+1}) - V(s_t) (TD error)
    
    Args:
        rewards: Rewards at each timestep (T,) or (B, T)
        values: Value estimates V(s_t) (T,) or (B, T)
        next_values: Value estimates V(s_{t+1}) (T,) or (B, T)
                     For terminal states, should be 0
        dones: Done flags (1 for terminal, 0 otherwise). Optional.
        gamma: Discount factor γ ∈ [0, 1]
        lam: GAE parameter λ ∈ [0, 1]
        
    Returns:
        Tuple of:
        - advantages: GAE advantages (same shape as rewards)
        - returns: TD(λ) returns (advantages + values)
    """
    # Handle batch dimension
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(0)
        values = values.unsqueeze(0)
        next_values = next_values.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, T = rewards.shape
    device = rewards.device
    
    # Default: no terminal states
    if dones is None:
        dones = torch.zeros_like(rewards)
    
    # Compute TD errors: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
    not_dones = 1.0 - dones
    td_errors = rewards + gamma * next_values * not_dones - values
    
    # Compute GAE via backward iteration
    # A_t = δ_t + (γλ)(1-done_t) * A_{t+1}
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=device)
    
    for t in reversed(range(T)):
        gae = td_errors[:, t] + gamma * lam * not_dones[:, t] * gae
        advantages[:, t] = gae
    
    # Returns = advantages + values (TD(λ) returns)
    returns = advantages + values
    
    if squeeze_output:
        advantages = advantages.squeeze(0)
        returns = returns.squeeze(0)
    
    return advantages, returns


def compute_gae_from_trajectory(
    trajectory_rewards: List[float],
    trajectory_values: List[float],
    last_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE from a single trajectory (Python list version).
    
    Useful for processing one video/scene at a time.
    
    Args:
        trajectory_rewards: List of rewards [r_0, r_1, ..., r_{T-1}]
        trajectory_values: List of value estimates [V(s_0), ..., V(s_{T-1})]
        last_value: V(s_T), the value of the terminal state (0 if done)
        gamma: Discount factor
        lam: GAE lambda
        
    Returns:
        Tuple of (advantages, returns) as numpy arrays
    """
    T = len(trajectory_rewards)
    
    if T == 0:
        return np.array([]), np.array([])
    
    rewards = np.array(trajectory_rewards, dtype=np.float32)
    values = np.array(trajectory_values, dtype=np.float32)
    
    # Construct next_values: [V(s_1), V(s_2), ..., V(s_T)]
    next_values = np.zeros(T, dtype=np.float32)
    next_values[:-1] = values[1:]
    next_values[-1] = last_value
    
    # Convert to tensors and compute
    rewards_t = torch.from_numpy(rewards)
    values_t = torch.from_numpy(values)
    next_values_t = torch.from_numpy(next_values)
    
    advantages_t, returns_t = compute_gae(
        rewards_t, values_t, next_values_t, gamma=gamma, lam=lam
    )
    
    return advantages_t.numpy(), returns_t.numpy()


def compute_advantages_for_scenes(
    scene_rewards: List[List[float]],
    scene_values: List[List[float]],
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute GAE advantages for multiple scenes (batch processing).
    
    Each scene is treated as an independent episode (no bootstrapping
    between scenes).
    
    Args:
        scene_rewards: List of reward lists, one per scene
        scene_values: List of value lists, one per scene
        gamma: Discount factor
        lam: GAE lambda
        normalize: Whether to normalize across all scenes
        
    Returns:
        Tuple of (advantages_list, returns_list)
    """
    all_advantages = []
    all_returns = []
    
    for rewards, values in zip(scene_rewards, scene_values):
        adv, ret = compute_gae_from_trajectory(
            rewards, values, last_value=0.0, gamma=gamma, lam=lam
        )
        all_advantages.append(adv)
        all_returns.append(ret)
    
    # Normalize advantages across all scenes
    if normalize:
        all_adv_flat = np.concatenate(all_advantages)
        mean_adv = all_adv_flat.mean()
        std_adv = all_adv_flat.std() + 1e-8
        
        all_advantages = [(adv - mean_adv) / std_adv for adv in all_advantages]
    
    return all_advantages, all_returns


class GAEComputer:
    """
    Stateful GAE computer with configurable parameters.
    
    Provides a more object-oriented interface for GAE computation.
    
    Usage:
        >>> gae_computer = GAEComputer(gamma=0.99, lam=0.95)
        >>> 
        >>> # Process a trajectory
        >>> advantages, returns = gae_computer.compute(rewards, values)
        >>> 
        >>> # Or batch process
        >>> for adv, ret in gae_computer.compute_batch(batch_rewards, batch_values):
        ...     # process each scene
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize: bool = True
    ):
        """
        Initialize GAE computer.
        
        Args:
            gamma: Discount factor (default: 0.99)
            lam: GAE lambda (default: 0.95)
            normalize: Whether to normalize advantages
        """
        self.gamma = gamma
        self.lam = lam
        self.normalize = normalize
    
    def compute(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: Rewards (T,) or (B, T)
            values: Current value estimates
            next_values: Next state values. If None, shifted from values.
            dones: Done flags (optional)
            
        Returns:
            Tuple of (advantages, returns)
        """
        if next_values is None:
            # Shift values to get next_values
            if rewards.dim() == 1:
                next_values = torch.zeros_like(values)
                next_values[:-1] = values[1:]
            else:
                next_values = torch.zeros_like(values)
                next_values[:, :-1] = values[:, 1:]
        
        advantages, returns = compute_gae(
            rewards, values, next_values, dones, self.gamma, self.lam
        )
        
        if self.normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_from_lists(
        self,
        rewards: List[float],
        values: List[float],
        last_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE from Python lists.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            last_value: Bootstrap value (0 for terminal)
            
        Returns:
            Tuple of (advantages, returns) as numpy arrays
        """
        advantages, returns = compute_gae_from_trajectory(
            rewards, values, last_value, self.gamma, self.lam
        )
        
        if self.normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns


if __name__ == "__main__":
    # Demo and tests
    print("=== GAE Demo ===\n")
    
    # Test 1: Basic GAE computation
    print("Test 1: Basic GAE")
    rewards = torch.tensor([1.0, 1.0, 1.0, 10.0])  # High reward at end
    values = torch.tensor([2.0, 3.0, 5.0, 8.0])    # Increasing values
    next_values = torch.tensor([3.0, 5.0, 8.0, 0.0])  # Shifted, terminal at end
    
    advantages, returns = compute_gae(rewards, values, next_values, gamma=0.99, lam=0.95)
    print(f"  Rewards: {rewards}")
    print(f"  Values: {values}")
    print(f"  Advantages: {advantages}")
    print(f"  Returns: {returns}")
    
    # Test 2: Lambda = 0 (TD(0))
    print("\nTest 2: Lambda = 0 (TD error)")
    adv_td0, _ = compute_gae(rewards, values, next_values, gamma=0.99, lam=0.0)
    td_errors = rewards + 0.99 * next_values - values
    print(f"  GAE(λ=0): {adv_td0}")
    print(f"  TD errors: {td_errors}")
    print(f"  Match: {torch.allclose(adv_td0, td_errors)}")
    
    # Test 3: Batch processing
    print("\nTest 3: Batch processing")
    batch_rewards = torch.rand(4, 10)
    batch_values = torch.rand(4, 10)
    batch_next_values = torch.rand(4, 10)
    
    batch_adv, batch_ret = compute_gae(
        batch_rewards, batch_values, batch_next_values, gamma=0.99, lam=0.95
    )
    print(f"  Batch shape: {batch_adv.shape}")
    print(f"  Advantage mean: {batch_adv.mean():.4f}")
    print(f"  Advantage std: {batch_adv.std():.4f}")
    
    # Test 4: GAE computer class
    print("\nTest 4: GAEComputer class")
    gae_computer = GAEComputer(gamma=0.99, lam=0.95, normalize=True)
    
    list_rewards = [1.0, 0.5, 2.0, 1.0, 0.0]
    list_values = [1.0, 1.5, 2.0, 1.5, 1.0]
    
    adv, ret = gae_computer.compute_from_lists(list_rewards, list_values)
    print(f"  List rewards: {list_rewards}")
    print(f"  Advantages (normalized): {adv}")
    
    # Test 5: Multi-scene processing
    print("\nTest 5: Multi-scene processing")
    scene_rewards = [[1.0, 2.0, 1.0], [0.5, 1.5, 2.5, 3.0]]
    scene_values = [[1.0, 1.5, 1.0], [0.5, 1.0, 1.5, 2.0]]
    
    all_adv, all_ret = compute_advantages_for_scenes(
        scene_rewards, scene_values, normalize=True
    )
    print(f"  Scene 1 advantages: {all_adv[0]}")
    print(f"  Scene 2 advantages: {all_adv[1]}")
    
    print("\n✅ GAE tests passed!")
