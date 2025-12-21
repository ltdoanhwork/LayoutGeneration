#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSN V8: Multi-Task DSN with State-Dependent Gating and PCGrad

Key improvements:
1. State-dependent gating: learns alpha_t per frame instead of fixed 0.5
2. PCGrad gradient surgery: projects conflicting gradients for stable training
3. Separate value heads for constrained objectives
4. Support for DPP-based selection

Architecture:
    Input Features (CLIP + Anime attrs)
              |
        [Shared Encoder]
              |
        [Policy Backbone (BiLSTM)]
              |
    +---------+---------+
    |                   |
[Rec Head]        [Anime Head]
    |                   |
(pi_rec, V_rec)  (pi_anime, V_anime)
    |                   |
    +-------+-------+
            |
     [Gating Network]
            |
        alpha_t
            |
     [Merged Policy]
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.dsn_advanced import DSNConfig
from src.models.value_network import ValueHead


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class V8Config:
    """V8-specific configuration"""
    # Base DSN config
    base_config: Optional[DSNConfig] = None
    
    # Gating network
    gating_hidden_dim: int = 64
    gating_num_layers: int = 2
    gating_dropout: float = 0.1
    gating_init_bias: float = 0.0  # Initial bias (0 = start at 0.5)
    
    # PCGrad settings
    use_pcgrad: bool = True
    pcgrad_reduction: str = "mean"  # 'mean' or 'sum'
    
    # Task settings
    num_tasks: int = 2
    task_names: Tuple[str, ...] = ("rec", "anime")


# ============================================================================
# Gating Network
# ============================================================================

class GatingNetwork(nn.Module):
    """
    State-dependent gating network.
    
    Takes hidden states and outputs alpha_t ∈ [0, 1] per frame.
    
    alpha_t close to 1 → favor rec head (content preservation)
    alpha_t close to 0 → favor anime head (aesthetic quality)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        init_bias: float = 0.0,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Final layer outputs logit (sigmoid applied later)
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize last layer bias for starting alpha
        with torch.no_grad():
            self.net[-1].bias.fill_(init_bias)
            self.net[-1].weight.mul_(0.01)  # Small weights for stable start
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, D) hidden states
        
        Returns:
            alpha: (B, T) gating weights in [0, 1]
        """
        logits = self.net(h)  # (B, T, 1)
        alpha = torch.sigmoid(logits.squeeze(-1))  # (B, T)
        return alpha


# ============================================================================
# Task Head
# ============================================================================

class TaskHeadV8(nn.Module):
    """
    Task-specific head with policy and value outputs.
    
    Enhanced with layer normalization and residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Policy head
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Value head
        self.value_head = ValueHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, T, D) hidden features
        
        Returns:
            logits: (B, T) policy logits
            values: (B, T) value estimates
        """
        logits = self.policy_net(h).squeeze(-1)  # (B, T)
        values = self.value_head(h)  # (B, T)
        
        return logits, values


# ============================================================================
# PCGrad Implementation
# ============================================================================

def pcgrad_project(grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    PCGrad: Project conflicting gradients.
    
    For each gradient, project out components that conflict with other gradients.
    
    From "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
    
    Args:
        grads: List of gradient tensors (flattened)
    
    Returns:
        projected_grads: List of projected gradients
    """
    num_tasks = len(grads)
    projected = [g.clone() for g in grads]
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                continue
            
            # Check if gradients conflict (negative dot product)
            dot = torch.dot(projected[i].flatten(), grads[j].flatten())
            
            if dot < 0:
                # Project out conflicting component
                norm_sq = torch.dot(grads[j].flatten(), grads[j].flatten())
                if norm_sq > 1e-8:
                    projected[i] = projected[i] - (dot / norm_sq) * grads[j]
    
    return projected


class PCGradOptimizer:
    """
    Wrapper for PCGrad optimization.
    
    Usage:
        pcgrad_opt = PCGradOptimizer(optimizer, num_tasks=2)
        
        # Compute per-task losses
        loss_rec = compute_loss_rec(...)
        loss_anime = compute_loss_anime(...)
        
        # Backward with PCGrad
        pcgrad_opt.backward([loss_rec, loss_anime], model)
        pcgrad_opt.step()
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, num_tasks: int = 2):
        self.optimizer = optimizer
        self.num_tasks = num_tasks
        self._grads = None
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        self._grads = None
    
    def backward(self, losses: List[torch.Tensor], model: nn.Module):
        """
        Compute PCGrad-corrected gradients.
        
        Args:
            losses: List of per-task losses
            model: Model to compute gradients for
        """
        assert len(losses) == self.num_tasks
        
        # Compute per-task gradients
        task_grads = []
        
        for i, loss in enumerate(losses):
            self.optimizer.zero_grad()
            loss.backward(retain_graph=(i < len(losses) - 1))
            
            # Collect gradients
            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.clone().flatten())
                else:
                    grads.append(torch.zeros_like(param).flatten())
            
            task_grads.append(torch.cat(grads))
        
        # Project conflicting gradients
        projected = pcgrad_project(task_grads)
        
        # Average projected gradients
        avg_grad = torch.stack(projected).mean(dim=0)
        
        # Apply to model
        self.optimizer.zero_grad()
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            param.grad = avg_grad[offset:offset + numel].view(param.shape)
            offset += numel
        
        self._grads = avg_grad
    
    def step(self):
        self.optimizer.step()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)


# ============================================================================
# Main Model
# ============================================================================

class DSNMultiTaskV8(nn.Module):
    """
    V8 Multi-Task DSN with state-dependent gating and PCGrad support.
    
    Key features:
    1. State-dependent alpha_t for dynamic head weighting
    2. Separate task heads for rec and anime objectives
    3. PCGrad-compatible architecture
    4. DPP selection support via quality scoring
    
    Note: Uses standalone encoder components instead of DSNPolicyAdvanced
    to have full control over hidden state extraction.
    """
    
    def __init__(
        self,
        config: DSNConfig,
        v8_config: Optional[V8Config] = None,
    ):
        super().__init__()
        
        self.config = config
        self.v8_config = v8_config or V8Config()
        
        # Standalone encoder components
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.feat_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Positional encoding
        self.pos_enc = nn.Sequential(
            nn.Dropout(config.dropout),
        )
        # Add sinusoidal positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_sinusoidal_encoding(config.hidden_dim, max_len=2000)
        )
        
        # Self-attention layers
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attn_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
            )
            for _ in range(config.num_attn_layers)
        ])
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0
        )
        
        # LSTM output dimension
        lstm_out_dim = config.lstm_hidden * (2 if config.bidirectional else 1)
        self.lstm_out_dim = lstm_out_dim
        
        # Task heads
        self.rec_head = TaskHeadV8(
            input_dim=lstm_out_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        
        self.anime_head = TaskHeadV8(
            input_dim=lstm_out_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        
        # Gating network
        self.gating = GatingNetwork(
            input_dim=lstm_out_dim,
            hidden_dim=self.v8_config.gating_hidden_dim,
            num_layers=self.v8_config.gating_num_layers,
            dropout=self.v8_config.gating_dropout,
            init_bias=self.v8_config.gating_init_bias,
        )
        
        # Store last gating weights for logging
        self._last_alpha = None
    
    def _create_sinusoidal_encoding(self, d_model: int, max_len: int = 2000) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def get_shared_hidden(
        self,
        x: torch.Tensor,
        motion_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get shared hidden state from encoder.
        
        Args:
            x: (B, T, D) input features
            motion_feats: Optional motion features (not used in V8, kept for API compat)
        
        Returns:
            h: (B, T, D_lstm) hidden state
        """
        B, T, D = x.shape
        
        # Input projection
        h = self.input_proj(x)  # (B, T, hidden)
        
        # Add positional encoding
        h = h + self.pos_encoding[:, :T, :].to(h.device)
        
        # Self-attention layers
        for layer in self.attn_layers:
            h = layer(h)
        
        # LSTM
        h, _ = self.lstm(h)  # (B, T, D_lstm)
        
        return h
    
    def forward_task(
        self,
        x: torch.Tensor,
        task: str,
        motion_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a specific task.
        
        Returns:
            probs: (B, T) selection probabilities  
            logits: (B, T) raw logits
            values: (B, T) value estimates
        """
        h = self.get_shared_hidden(x, motion_feats)
        
        if task == "rec":
            logits, values = self.rec_head(h)
        elif task == "anime":
            logits, values = self.anime_head(h)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        probs = torch.softmax(logits, dim=-1)
        
        return probs, logits, values
    
    def forward_all_tasks(
        self,
        x: torch.Tensor,
        motion_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for all tasks (efficient - shared hidden computed once).
        
        Returns:
            Dict mapping task name -> (probs, logits, values)
        """
        h = self.get_shared_hidden(x, motion_feats)
        
        # Rec head
        rec_logits, rec_values = self.rec_head(h)
        rec_probs = torch.softmax(rec_logits, dim=-1)
        
        # Anime head
        anime_logits, anime_values = self.anime_head(h)
        anime_probs = torch.softmax(anime_logits, dim=-1)
        
        return {
            "rec": (rec_probs, rec_logits, rec_values),
            "anime": (anime_probs, anime_logits, anime_values),
        }
    
    def forward(
        self,
        x: torch.Tensor,
        motion_feats: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        return_all_tasks: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with state-dependent gating.
        
        Args:
            x: (B, T, D) input features
            motion_feats: Optional motion features
            return_gating: Whether to return gating weights
            return_all_tasks: Whether to return all task outputs
        
        Returns:
            If return_all_tasks:
                Dict mapping task name -> (probs, values)
            Else:
                probs: (B, T) merged selection probabilities
                values: (B, T) merged value estimates
                [alpha]: (B, T) gating weights (if return_gating)
        """
        h = self.get_shared_hidden(x, motion_feats)
        
        # Get task outputs
        rec_logits, rec_values = self.rec_head(h)
        anime_logits, anime_values = self.anime_head(h)
        
        rec_probs = torch.softmax(rec_logits, dim=-1)
        anime_probs = torch.softmax(anime_logits, dim=-1)
        
        if return_all_tasks:
            return {
                "rec": (rec_probs, rec_values),
                "anime": (anime_probs, anime_values),
            }
        
        # State-dependent gating
        alpha = self.gating(h)  # (B, T)
        self._last_alpha = alpha.detach()
        
        # Merge policies: pi = alpha * pi_rec + (1 - alpha) * pi_anime
        merged_probs = alpha * rec_probs + (1 - alpha) * anime_probs
        merged_values = alpha * rec_values + (1 - alpha) * anime_values
        
        if return_gating:
            return merged_probs, merged_values, alpha
        
        return merged_probs, merged_values
    
    def get_gating_stats(self) -> Dict[str, float]:
        """Get statistics about gating weights for logging"""
        if self._last_alpha is None:
            return {}
        
        alpha = self._last_alpha.cpu().numpy().flatten()
        
        return {
            "gating_mean": float(np.mean(alpha)),
            "gating_std": float(np.std(alpha)),
            "gating_min": float(np.min(alpha)),
            "gating_max": float(np.max(alpha)),
            "gating_rec_dominant": float(np.mean(alpha > 0.5)),  # Fraction favoring rec
        }
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get encoder cache statistics (not used in V8)"""
        return {}
    
    def clear_cache(self):
        """Clear encoder cache (not used in V8)"""
        pass


# ============================================================================
# Factory Functions
# ============================================================================

def create_dsn_v8(
    feat_dim: int = 512,
    hidden_dim: int = 256,
    lstm_hidden: int = 128,
    use_pcgrad: bool = True,
    gating_hidden: int = 64,
    **kwargs
) -> DSNMultiTaskV8:
    """Factory function to create V8 model"""
    
    base_config = DSNConfig(
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        lstm_hidden=lstm_hidden,
        **{k: v for k, v in kwargs.items() if hasattr(DSNConfig, k)}
    )
    
    v8_config = V8Config(
        base_config=base_config,
        gating_hidden_dim=gating_hidden,
        use_pcgrad=use_pcgrad,
    )
    
    return DSNMultiTaskV8(base_config, v8_config)


def create_pcgrad_optimizer(
    model: DSNMultiTaskV8,
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
) -> PCGradOptimizer:
    """Create PCGrad optimizer for V8 model"""
    
    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    return PCGradOptimizer(base_optimizer, num_tasks=2)
