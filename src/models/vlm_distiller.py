#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Distiller Module (Student)

This module implements the student model that distills knowledge from large VLMs
into a lightweight MLP head on top of frozen CLIP features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class VLMDistiller(nn.Module):
    """
    Student model for anime quality prediction.
    Outputs 8 dimensions of quality scores based on VLM teacher distillation.
    """
    
    def __init__(
        self,
        input_dim: int = 512,      # CLIP Vit-B/32 embedding size
        n_quality_dims: int = 8,   # Number of quality dimensions
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_quality_dims = n_quality_dims
        
        # MLP Head
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(curr_dim, n_quality_dims)
        
    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_features: (B, 512) CLIP image embeddings
        Returns:
            quality_scores: (B, 8) scores in [0, 1]
        """
        # Normalize features if they aren't already
        clip_features = F.normalize(clip_features, dim=-1)
        
        feat = self.mlp(clip_features)
        scores = self.head(feat)
        
        return torch.sigmoid(scores)

class DistillationLoss(nn.Module):
    """Loss for training student on teacher pseudo-labels."""
    
    def __init__(self, use_log_scale: bool = False):
        super().__init__()
        self.use_log_scale = use_log_scale
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE loss between student predictions and VLM teacher labels."""
        if self.use_log_scale:
            # Penalize errors more heavily in high-quality regions
            return F.mse_loss(torch.log(pred + 1e-6), torch.log(target + 1e-6))
        return F.mse_loss(pred, target)

if __name__ == "__main__":
    model = VLMDistiller()
    test_input = torch.randn(4, 512)
    output = model(test_input)
    print("Predicted Scores Shape:", output.shape)
    print("Values (first row):", output[0].detach().numpy())
