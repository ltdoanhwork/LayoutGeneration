#!/usr/bin/env python3
"""
Value Network for Actor-Critic Architecture (V4).

This module implements the value head (critic) for the Actor-Critic
PPO training setup. The value network estimates V(s) for advantage computation.

Key Features:
- Shared encoder with policy (optional)
- Separate value head for state value estimation
- Frame-level and sequence-level value outputs
- Compatible with DSNAdvanced architecture

Author: V4 Enhancement
Date: 2025-12-06
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """
    Value head for Actor-Critic architecture.
    
    Takes encoded features from the shared encoder and outputs
    value estimates V(s) for each frame or the entire sequence.
    
    Architecture:
        encoded_features (B, T, D) -> MLP -> values (B, T) or (B,)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_type: str = "frame",  # "frame" or "sequence"
        dropout: float = 0.1
    ):
        """
        Initialize value head.
        
        Args:
            input_dim: Dimension of input features from encoder
            hidden_dim: Hidden layer dimension
            output_type: "frame" for per-frame values, "sequence" for single value
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_type = output_type
        
        # Value network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Sequence-level pooling (if needed)
        if output_type == "sequence":
            self.attention_pool = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        encoded_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute value estimates.
        
        Args:
            encoded_features: Encoded features (B, T, D) from shared encoder
            mask: Optional mask for variable-length sequences (B, T)
            
        Returns:
            values: Value estimates
                - If output_type == "frame": (B, T)
                - If output_type == "sequence": (B,)
        """
        B, T, D = encoded_features.shape
        
        if self.output_type == "frame":
            # Per-frame value estimation
            values = self.net(encoded_features).squeeze(-1)  # (B, T)
            
            if mask is not None:
                values = values * mask
            
        else:  # sequence
            # Attention-weighted pooling for sequence value
            attn_weights = self.attention_pool(encoded_features)  # (B, T, 1)
            
            if mask is not None:
                attn_weights = attn_weights * mask.unsqueeze(-1)
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            pooled = (encoded_features * attn_weights).sum(dim=1)  # (B, D)
            values = self.net(pooled).squeeze(-1)  # (B,)
        
        return values


class DualOutputHead(nn.Module):
    """
    Combined policy and value heads for Actor-Critic.
    
    Efficiently processes encoded features to output both
    action probabilities and value estimates.
    
    Architecture:
        encoded_features -> [Policy MLP -> probs, Value MLP -> values]
    """
    
    def __init__(
        self,
        input_dim: int,
        policy_hidden_dim: int = 256,
        value_hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize dual output head.
        
        Args:
            input_dim: Dimension of encoded features
            policy_hidden_dim: Hidden dim for policy head
            value_hidden_dim: Hidden dim for value head
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, policy_hidden_dim),
            nn.LayerNorm(policy_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(policy_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Value head (critic)
        self.value_head = ValueHead(
            input_dim=input_dim,
            hidden_dim=value_hidden_dim,
            output_type="frame",
            dropout=dropout
        )
        
        # Initialize policy head
        self._init_policy_weights()
    
    def _init_policy_weights(self):
        """Initialize policy weights for exploration."""
        for m in self.policy_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    # Initialize bias to produce probabilities around 0.1-0.2
                    # (sparse selection is typical for summarization)
                    if m.out_features == 1:
                        nn.init.constant_(m.bias, -1.5)  # sigmoid(-1.5) ≈ 0.18
                    else:
                        nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        encoded_features: torch.Tensor,
        return_value: bool = True
    ) -> tuple:
        """
        Compute policy probabilities and value estimates.
        
        Args:
            encoded_features: Encoded features (B, T, D)
            return_value: Whether to compute value estimates
            
        Returns:
            Tuple of (probs, values) if return_value else just probs
                - probs: Selection probabilities (B, T)
                - values: Value estimates (B, T)
        """
        # Policy output
        probs = self.policy_head(encoded_features).squeeze(-1)  # (B, T)
        
        if return_value:
            values = self.value_head(encoded_features)  # (B, T)
            return probs, values
        
        return probs


class SeparateValueNetwork(nn.Module):
    """
    Completely separate value network (not sharing encoder).
    
    For cases where you want independent value function learning.
    This can be more stable but requires more compute.
    """
    
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize separate value network.
        
        Args:
            feat_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        self.feat_dim = feat_dim
        
        # Build MLP
        layers = []
        in_dim = feat_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Last layer with small weights
        final_layer = list(self.net.modules())[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.orthogonal_(final_layer.weight, gain=0.01)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimates from raw features.
        
        Args:
            features: Input features (B, T, D)
            
        Returns:
            values: Value estimates (B, T)
        """
        return self.net(features).squeeze(-1)


if __name__ == "__main__":
    print("=== Value Network Demo ===\n")
    
    # Test configuration
    B, T, D = 4, 100, 512
    encoded_features = torch.randn(B, T, D)
    
    # Test 1: ValueHead (frame-level)
    print("Test 1: ValueHead (frame-level)")
    value_head = ValueHead(input_dim=D, hidden_dim=256, output_type="frame")
    frame_values = value_head(encoded_features)
    print(f"  Input shape: {encoded_features.shape}")
    print(f"  Output shape: {frame_values.shape}")
    print(f"  Value range: [{frame_values.min().item():.4f}, {frame_values.max().item():.4f}]")
    
    # Test 2: ValueHead (sequence-level)
    print("\nTest 2: ValueHead (sequence-level)")
    seq_value_head = ValueHead(input_dim=D, hidden_dim=256, output_type="sequence")
    seq_values = seq_value_head(encoded_features)
    print(f"  Output shape: {seq_values.shape}")
    
    # Test 3: DualOutputHead
    print("\nTest 3: DualOutputHead (Actor-Critic)")
    dual_head = DualOutputHead(input_dim=D)
    probs, values = dual_head(encoded_features, return_value=True)
    print(f"  Probs shape: {probs.shape}, range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"  Values shape: {values.shape}")
    
    # Test 4: SeparateValueNetwork
    print("\nTest 4: SeparateValueNetwork")
    sep_value_net = SeparateValueNetwork(feat_dim=D, hidden_dim=256)
    sep_values = sep_value_net(encoded_features)
    print(f"  Output shape: {sep_values.shape}")
    
    # Test 5: Gradient flow
    print("\nTest 5: Gradient flow")
    loss = values.mean() + probs.mean()
    loss.backward()
    
    grad_policy = dual_head.policy_head[0].weight.grad
    grad_value = dual_head.value_head.net[0].weight.grad
    print(f"  Policy head grad norm: {grad_policy.norm().item():.6f}")
    print(f"  Value head grad norm: {grad_value.norm().item():.6f}")
    
    print("\n✅ Value Network tests passed!")
