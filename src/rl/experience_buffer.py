#!/usr/bin/env python3
"""
Experience Buffer for PPO Training (V4).

This module implements an experience buffer for collecting and processing
rollouts during PPO training. It stores transitions and computes advantages.

Key Features:
- Efficient storage of (state, action, reward, value, log_prob) tuples
- Automatic advantage computation using GAE
- Mini-batch sampling for PPO updates
- Support for variable-length scenes

Author: V4 Enhancement
Date: 2025-12-06
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterator
import torch
import numpy as np

from src.rl.gae import compute_gae


@dataclass
class Transition:
    """Single transition in the experience buffer."""
    scene_id: str
    features: np.ndarray           # (T, D) frame features
    actions: np.ndarray            # (T,) binary selection actions
    log_probs: np.ndarray          # (T,) log probabilities
    reward: float                  # Scalar episode reward
    values: np.ndarray             # (T,) value estimates per frame
    probs: np.ndarray              # (T,) action probabilities


@dataclass
class BatchData:
    """Batch of data for PPO update."""
    features: torch.Tensor         # (B, T, D) or list of (T_i, D)
    actions: torch.Tensor          # (B, T) or flattened
    old_log_probs: torch.Tensor    # (B, T) or flattened
    advantages: torch.Tensor       # (B, T) or flattened
    returns: torch.Tensor          # (B, T) or flattened
    old_values: torch.Tensor       # (B, T) or flattened
    old_probs: torch.Tensor        # (B, T) or flattened
    scene_ids: List[str]           # Scene identifiers


class ExperienceBuffer:
    """
    Experience buffer for PPO rollout collection.
    
    Collects transitions during rollout phase, then processes them
    for PPO updates with GAE advantage computation.
    
    Usage:
        >>> buffer = ExperienceBuffer(gamma=0.99, lam=0.95)
        >>> 
        >>> # Collect experiences
        >>> for scene in scenes:
        ...     buffer.add(scene_id, features, actions, log_probs, 
        ...                reward, values, probs)
        >>> 
        >>> # Process and get batches
        >>> buffer.compute_advantages()
        >>> for batch in buffer.get_batches(batch_size=16):
        ...     # PPO update with batch
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_advantages: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize experience buffer.
        
        Args:
            gamma: Discount factor for GAE
            lam: Lambda parameter for GAE (bias-variance tradeoff)
            normalize_advantages: Whether to normalize advantages
            device: Device for tensor operations
        """
        self.gamma = gamma
        self.lam = lam
        self.normalize_advantages = normalize_advantages
        self.device = device
        
        # Storage
        self.transitions: List[Transition] = []
        
        # Processed data (after compute_advantages)
        self.advantages: List[np.ndarray] = []
        self.returns: List[np.ndarray] = []
        self._processed = False
    
    def add(
        self,
        scene_id: str,
        features: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        values: np.ndarray,
        probs: np.ndarray
    ):
        """
        Add a transition (scene) to the buffer.
        
        Args:
            scene_id: Unique identifier for the scene
            features: Frame features (T, D)
            actions: Binary selection actions (T,)
            log_probs: Log probabilities of actions (T,)
            reward: Scalar reward for the episode/scene
            values: Value estimates per frame (T,)
            probs: Action probabilities (T,)
        """
        transition = Transition(
            scene_id=scene_id,
            features=features.astype(np.float32),
            actions=actions.astype(np.float32),
            log_probs=log_probs.astype(np.float32),
            reward=float(reward),
            values=values.astype(np.float32),
            probs=probs.astype(np.float32)
        )
        self.transitions.append(transition)
        self._processed = False
    
    def compute_advantages(self) -> None:
        """
        Compute GAE advantages for all collected transitions.
        
        For video summarization, we treat each scene as an episode
        with the final reward. We distribute the reward across frames
        based on the selection probability and compute GAE.
        """
        self.advantages = []
        self.returns = []
        
        for trans in self.transitions:
            T = len(trans.actions)
            
            # Distribute episode reward to frame-level rewards
            # Option 1: Equal distribution (simple)
            # Option 2: Weighted by selection probability
            # We use a weighted approach for better credit assignment
            
            # Create frame rewards: selected frames get proportional reward
            selection_weight = trans.actions * trans.probs
            weight_sum = selection_weight.sum() + 1e-8
            frame_rewards = trans.reward * (selection_weight / weight_sum)
            
            # Add a small baseline reward for non-selected frames to reduce variance
            frame_rewards += trans.reward * 0.01 * (1 - trans.actions)
            
            # Prepare values for GAE
            # next_values: shifted values, terminal is 0
            next_values = np.zeros_like(trans.values)
            next_values[:-1] = trans.values[1:]
            
            # Compute GAE
            rewards_t = torch.from_numpy(frame_rewards)
            values_t = torch.from_numpy(trans.values)
            next_values_t = torch.from_numpy(next_values)
            
            adv, ret = compute_gae(
                rewards_t, values_t, next_values_t,
                gamma=self.gamma, lam=self.lam
            )
            
            self.advantages.append(adv.numpy())
            self.returns.append(ret.numpy())
        
        # Normalize advantages across all transitions
        if self.normalize_advantages and len(self.advantages) > 0:
            all_adv = np.concatenate(self.advantages)
            mean_adv = all_adv.mean()
            std_adv = all_adv.std() + 1e-8
            
            self.advantages = [
                (adv - mean_adv) / std_adv for adv in self.advantages
            ]
        
        self._processed = True
    
    def get_batches(
        self,
        batch_size: int = 16,
        shuffle: bool = True
    ) -> Iterator[BatchData]:
        """
        Generate mini-batches for PPO updates.
        
        Args:
            batch_size: Number of scenes per batch
            shuffle: Whether to shuffle before batching
            
        Yields:
            BatchData objects containing batched tensors
        """
        if not self._processed:
            self.compute_advantages()
        
        n = len(self.transitions)
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            # Collect batch data
            batch_features = []
            batch_actions = []
            batch_log_probs = []
            batch_advantages = []
            batch_returns = []
            batch_values = []
            batch_probs = []
            batch_scene_ids = []
            
            for idx in batch_indices:
                trans = self.transitions[idx]
                batch_features.append(trans.features)
                batch_actions.append(trans.actions)
                batch_log_probs.append(trans.log_probs)
                batch_advantages.append(self.advantages[idx])
                batch_returns.append(self.returns[idx])
                batch_values.append(trans.values)
                batch_probs.append(trans.probs)
                batch_scene_ids.append(trans.scene_id)
            
            # Flatten for variable-length sequences
            # (Can be modified to use padding if needed)
            yield BatchData(
                features=[torch.from_numpy(f).to(self.device) for f in batch_features],
                actions=torch.from_numpy(np.concatenate(batch_actions)).to(self.device),
                old_log_probs=torch.from_numpy(np.concatenate(batch_log_probs)).to(self.device),
                advantages=torch.from_numpy(np.concatenate(batch_advantages)).to(self.device),
                returns=torch.from_numpy(np.concatenate(batch_returns)).to(self.device),
                old_values=torch.from_numpy(np.concatenate(batch_values)).to(self.device),
                old_probs=torch.from_numpy(np.concatenate(batch_probs)).to(self.device),
                scene_ids=batch_scene_ids
            )
    
    def get_all_data(self) -> BatchData:
        """Get all data as a single batch."""
        if not self._processed:
            self.compute_advantages()
        
        all_features = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_values = []
        all_probs = []
        all_scene_ids = []
        
        for i, trans in enumerate(self.transitions):
            all_features.append(trans.features)
            all_actions.append(trans.actions)
            all_log_probs.append(trans.log_probs)
            all_advantages.append(self.advantages[i])
            all_returns.append(self.returns[i])
            all_values.append(trans.values)
            all_probs.append(trans.probs)
            all_scene_ids.append(trans.scene_id)
        
        return BatchData(
            features=[torch.from_numpy(f).to(self.device) for f in all_features],
            actions=torch.from_numpy(np.concatenate(all_actions)).to(self.device),
            old_log_probs=torch.from_numpy(np.concatenate(all_log_probs)).to(self.device),
            advantages=torch.from_numpy(np.concatenate(all_advantages)).to(self.device),
            returns=torch.from_numpy(np.concatenate(all_returns)).to(self.device),
            old_values=torch.from_numpy(np.concatenate(all_values)).to(self.device),
            old_probs=torch.from_numpy(np.concatenate(all_probs)).to(self.device),
            scene_ids=all_scene_ids
        )
    
    def clear(self):
        """Clear the buffer for next epoch."""
        self.transitions = []
        self.advantages = []
        self.returns = []
        self._processed = False
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if len(self.transitions) == 0:
            return {}
        
        rewards = [t.reward for t in self.transitions]
        n_frames = [len(t.actions) for t in self.transitions]
        n_selected = [t.actions.sum() for t in self.transitions]
        
        stats = {
            "n_transitions": len(self.transitions),
            "total_frames": sum(n_frames),
            "total_selected": sum(n_selected),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_frames_per_scene": np.mean(n_frames),
        }
        
        if self._processed:
            all_adv = np.concatenate(self.advantages)
            stats["mean_advantage"] = all_adv.mean()
            stats["std_advantage"] = all_adv.std()
        
        return stats


if __name__ == "__main__":
    print("=== Experience Buffer Demo ===\n")
    
    # Create buffer
    buffer = ExperienceBuffer(gamma=0.99, lam=0.95, device="cpu")
    
    # Add some dummy transitions
    for i in range(5):
        T = np.random.randint(50, 150)
        D = 512
        
        features = np.random.randn(T, D).astype(np.float32)
        probs = np.random.rand(T).astype(np.float32) * 0.3 + 0.1
        actions = (np.random.rand(T) < probs).astype(np.float32)
        log_probs = actions * np.log(probs + 1e-8) + (1 - actions) * np.log(1 - probs + 1e-8)
        reward = np.random.randn() + 1.0
        values = np.random.randn(T).astype(np.float32) * 0.5 + reward
        
        buffer.add(
            scene_id=f"scene_{i}",
            features=features,
            actions=actions,
            log_probs=log_probs,
            reward=reward,
            values=values,
            probs=probs
        )
    
    print(f"Buffer size: {len(buffer)}")
    
    # Compute advantages
    buffer.compute_advantages()
    
    # Get stats
    stats = buffer.get_stats()
    print(f"Stats: {stats}")
    
    # Get batches
    print("\nIterating over batches:")
    for i, batch in enumerate(buffer.get_batches(batch_size=2)):
        print(f"  Batch {i}: {len(batch.scene_ids)} scenes, "
              f"advantages shape: {batch.advantages.shape}")
    
    print("\n✅ Experience Buffer tests passed!")
