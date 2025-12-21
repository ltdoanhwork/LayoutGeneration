"""
Multi-Task DSN Model for V5

This module extends DSNAdvanced with Multi-Task RL architecture:
- Task 1 (RecErr): Optimize reconstruction error / representativeness
- Task 2 (Anime): Optimize anime quality metrics

Each task has its own policy head and value head, sharing the backbone encoder.
During inference, policies are merged with learned weights.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dsn_advanced import DSNAdvanced, DSNConfig, DSNPolicyAdvanced
from src.models.value_network import ValueHead


@dataclass
class MultiTaskConfig:
    """Configuration for Multi-Task DSN."""
    # Base DSN config
    base_config: DSNConfig = None
    
    # Multi-task settings
    num_tasks: int = 2
    task_names: Tuple[str, ...] = ("rec", "anime")
    
    # Policy merge strategy: "learned", "average", "max"
    merge_strategy: str = "learned"
    
    # Initial merge weights (for "learned" strategy)
    init_merge_weight: float = 0.5  # α for rec task


class TaskHead(nn.Module):
    """
    A single task head containing:
    - Policy head (outputs logits for action selection)
    - Value head (outputs value estimates for PPO)
    """
    
    def __init__(self, lstm_out_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value head
        self.value_head = ValueHead(
            input_dim=lstm_out_dim,
            hidden_dim=hidden_dim,
            output_type="frame",
            dropout=dropout
        )
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, T, D) hidden features
        Returns:
            logits: (B, T) policy logits
            values: (B, T) value estimates
        """
        logits = self.policy_head(h).squeeze(-1)  # (B, T)
        values = self.value_head(h)  # (B, T)
        return logits, values


class DSNMultiTask(nn.Module):
    """
    Multi-Task DSN with separate heads for different objectives.
    
    Architecture:
    ```
    Input Features (CLIP + Anime attrs)
              |
        [Shared Encoder]
              |
        [Shared Policy Backbone]
        (Attention + LSTM)
              |
         [Hidden State]
           /        \
      [Head 1]    [Head 2]
      RecErr      Anime
        |           |
      π₁(a|s)    π₂(a|s)
      V₁(s)      V₂(s)
    ```
    
    During training: Compute loss for each head separately
    During inference: Merge policies with learned/fixed weights
    """
    
    def __init__(self, config: DSNConfig, mt_config: Optional[MultiTaskConfig] = None):
        super().__init__()
        
        self.config = config
        self.mt_config = mt_config or MultiTaskConfig(base_config=config)
        
        # Shared encoder
        from src.models.dsn_advanced import EncoderFCAdvanced
        self.encoder = EncoderFCAdvanced(
            in_dim=config.feat_dim,
            hidden_dim=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            use_cache=config.use_cache,
            cache_size=config.cache_size
        )
        
        # Shared policy backbone (everything except final head)
        self.policy_backbone = DSNPolicyAdvanced(config)
        
        # Get LSTM output dimension
        if config.use_lstm:
            self.lstm_out_dim = config.lstm_hidden * (2 if config.bidirectional else 1)
        else:
            self.lstm_out_dim = config.hidden_dim
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            "rec": TaskHead(self.lstm_out_dim, config.value_hidden_dim, config.dropout),
            "anime": TaskHead(self.lstm_out_dim, config.value_hidden_dim, config.dropout)
        })
        
        # Learnable merge weight (α for rec, 1-α for anime)
        if self.mt_config.merge_strategy == "learned":
            self.merge_weight = nn.Parameter(
                torch.tensor([self.mt_config.init_merge_weight])
            )
        else:
            self.register_buffer("merge_weight", torch.tensor([0.5]))
    
    def get_shared_hidden(
        self, 
        x: torch.Tensor, 
        scene_id: Optional[str] = None,
        motion_feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get shared hidden state after encoder and policy backbone.
        
        Args:
            x: (B, T, D) input features
            scene_id: Optional scene ID for caching
            motion_feats: Optional motion features
        Returns:
            h: (B, T, D_lstm) hidden state after LSTM
        """
        # Encoder
        h = self.encoder(x, scene_id)
        
        # Policy backbone processing (replicate the forward path)
        policy = self.policy_backbone
        
        # Motion fusion
        if policy.use_motion and motion_feats is not None and policy.motion_fusion is not None:
            h = policy.motion_fusion(motion_feats, h)
        
        # Positional encoding
        if policy.pos_encoder is not None:
            h = policy.pos_encoder(h)
        
        # Multi-scale temporal modeling
        h = policy.multi_scale(h)
        
        # Self-attention layers
        for attn_layer in policy.attn_layers:
            h = attn_layer(h)
        
        # LSTM
        if policy.use_lstm:
            h, _ = policy.lstm(h)
        
        # Dropout
        h = policy.dropout(h)
        
        return h
    
    def forward_task(
        self,
        x: torch.Tensor,
        task: str,
        scene_id: Optional[str] = None,
        motion_feats: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a specific task.
        
        Args:
            x: (B, T, D) input features
            task: "rec" or "anime"
            scene_id: Optional scene ID
            motion_feats: Optional motion features
        Returns:
            probs: (B, T) selection probabilities
            logits: (B, T) raw logits
            values: (B, T) value estimates
        """
        h = self.get_shared_hidden(x, scene_id, motion_feats)
        logits, values = self.task_heads[task](h)
        probs = torch.sigmoid(logits)
        return probs, logits, values
    
    def forward_all_tasks(
        self,
        x: torch.Tensor,
        scene_id: Optional[str] = None,
        motion_feats: Optional[torch.Tensor] = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for all tasks (efficient - shared hidden computed once).
        
        Returns:
            Dict mapping task name -> (probs, logits, values)
        """
        h = self.get_shared_hidden(x, scene_id, motion_feats)
        
        results = {}
        for task_name, head in self.task_heads.items():
            logits, values = head(h)
            probs = torch.sigmoid(logits)
            results[task_name] = (probs, logits, values)
        
        return results
    
    def forward(
        self,
        x: torch.Tensor,
        scene_id: Optional[str] = None,
        motion_feats: Optional[torch.Tensor] = None,
        return_value: bool = False,
        return_all_tasks: bool = False
    ) -> torch.Tensor:
        """
        Standard forward pass - returns merged policy for inference.
        
        Args:
            x: (B, T, D) input features
            scene_id: Optional scene ID
            motion_feats: Optional motion features
            return_value: If True, return (probs, values) tuple
            return_all_tasks: If True, return full task outputs dict
        Returns:
            probs: (B, T) merged selection probabilities
            values: (B, T) merged value estimates (if return_value=True)
        """
        if return_all_tasks:
            return self.forward_all_tasks(x, scene_id, motion_feats)
        
        # Get task outputs
        h = self.get_shared_hidden(x, scene_id, motion_feats)
        
        logits_rec, values_rec = self.task_heads["rec"](h)
        logits_anime, values_anime = self.task_heads["anime"](h)
        
        # Merge policies
        alpha = torch.sigmoid(self.merge_weight)  # Keep in [0, 1]
        merged_logits = alpha * logits_rec + (1 - alpha) * logits_anime
        merged_probs = torch.sigmoid(merged_logits)
        
        if return_value:
            # Also merge values
            merged_values = alpha * values_rec + (1 - alpha) * values_anime
            return merged_probs, merged_values
        
        return merged_probs
    
    def get_cache_stats(self) -> Optional[Dict[str, int]]:
        """Get encoder cache statistics."""
        return self.encoder.get_cache_stats()
    
    def clear_cache(self):
        """Clear encoder cache."""
        if self.encoder.cache is not None:
            self.encoder.cache.clear()


def create_dsn_multitask(
    feat_dim: int = 512,
    hidden_dim: int = 256,
    lstm_hidden: int = 128,
    **kwargs
) -> DSNMultiTask:
    """Factory function to create DSNMultiTask model."""
    config = DSNConfig(
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        lstm_hidden=lstm_hidden,
        use_actor_critic=True,  # Always enabled for multi-task
        **kwargs
    )
    return DSNMultiTask(config)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multi-Task DSN Model")
    print("=" * 60)
    
    # Create model
    config = DSNConfig(
        feat_dim=518,  # 512 CLIP + 6 anime attrs
        hidden_dim=256,
        lstm_hidden=128,
        num_attn_heads=4,
        num_attn_layers=2,
        use_cache=True,
        value_hidden_dim=128,
        use_actor_critic=True
    )
    
    model = DSNMultiTask(config)
    print(f"\n✓ Model created")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    B, T, D = 2, 50, 518
    x = torch.randn(B, T, D)
    
    print(f"\n✓ Testing merged forward (inference mode)")
    probs = model(x)
    print(f"  Output shape: {probs.shape}")
    assert probs.shape == (B, T)
    
    print(f"\n✓ Testing with return_value")
    probs, values = model(x, return_value=True)
    print(f"  Probs: {probs.shape}, Values: {values.shape}")
    
    print(f"\n✓ Testing all tasks forward")
    all_outputs = model(x, return_all_tasks=True)
    for task, (p, l, v) in all_outputs.items():
        print(f"  {task}: probs={p.shape}, logits={l.shape}, values={v.shape}")
    
    print(f"\n✓ Testing gradient flow")
    loss = probs.mean()
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {has_grad}/{len(list(model.parameters()))}")
    
    print(f"\n{'=' * 60}")
    print("All tests passed! ✅")
