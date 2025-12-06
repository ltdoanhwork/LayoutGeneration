"""
Multi-Video Replay Buffer for DSN RL Training.

This module provides a replay buffer capable of storing and sampling episodes 
from multiple videos, supporting various sampling strategies.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from dataclasses import dataclass
from collections import deque
import random

@dataclass
class Episode:
    """Stores a single episode trajectory."""
    video_id: str
    feats: np.ndarray          # (T, D)
    actions: np.ndarray        # (T,)
    log_probs: np.ndarray      # (T,)
    rewards: float             # scalar episode return
    entropy: float             # mean entropy
    
    # Optional metadata
    motion_feats: Optional[np.ndarray] = None
    anime_attrs: Optional[np.ndarray] = None
    
    # Computed advantages/returns can be added later if needed
    # For REINFORCE, we often just need the total reward R

class MultiVideoReplayBuffer:
    """
    Replay buffer for multi-video training.
    Stores episodes grouped by video_id or as a flat list depending on sampling needs.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: Maximum number of episodes to store
        """
        self.capacity = capacity
        self.buffer: deque[Episode] = deque(maxlen=capacity)
        self.video_ids: List[str] = []
        
        # Index for round-robin sampling
        self.rr_index = 0
        
    def add(self, episode: Episode):
        """Add an episode to the buffer."""
        self.buffer.append(episode)
        if episode.video_id not in self.video_ids:
            self.video_ids.append(episode.video_id)
            
    def sample_batch(self, batch_size: int, strategy: str = "random_uniform") -> List[Episode]:
        """
        Sample a batch of episodes.
        
        Args:
            batch_size: Number of episodes to sample
            strategy: "random_uniform" or "round_robin" (across videos not implemented yet for deque)
            
        Returns:
            List of sampled Episode objects
        """
        if len(self.buffer) < batch_size:
            # If not enough samples, return everything
            return list(self.buffer)
            
        if strategy == "random_uniform":
            return random.sample(self.buffer, batch_size)
        else:
            # Default to random for now
            return random.sample(self.buffer, batch_size)
            
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.video_ids = []
        
    def __len__(self):
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self.buffer:
            return {"size": 0, "mean_reward": 0.0}
            
        rewards = [ep.rewards for ep in self.buffer]
        return {
            "size": len(self.buffer),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "num_videos": len(set(ep.video_id for ep in self.buffer))
        }
