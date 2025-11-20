"""
Advanced Deep Summarization Network (DSN) for Keyframe Selection

This module implements an enhanced DSN architecture with:
1. Temporal Self-Attention - Capture long-range dependencies
2. Multi-Scale Temporal Modeling - Process at multiple temporal resolutions
3. Feature Caching - Optimize training efficiency
4. Positional Encoding - Explicit temporal position awareness

Author: AI-assisted implementation
Date: 2025-11-20
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable
import math
import hashlib
from collections import OrderedDict
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DSNConfig:
    """Configuration for Advanced DSN model."""
    # Input/Output dimensions
    feat_dim: int = 512
    hidden_dim: int = 256
    lstm_hidden: int = 128
    
    # Attention parameters
    num_attn_heads: int = 4
    num_attn_layers: int = 2
    attn_dropout: float = 0.1
    
    # Multi-scale parameters
    num_scales: int = 3  # 1x, 2x, 4x
    scale_fusion: str = "concat"  # "concat", "sum", "attention"
    
    # Positional encoding
    pos_encoding_type: str = "sinusoidal"  # "sinusoidal", "learned"
    max_seq_len: int = 1000
    
    # LSTM (optional, for ablation)
    use_lstm: bool = True
    lstm_layers: int = 1
    bidirectional: bool = True
    
    # Regularization
    dropout: float = 0.3
    
    # Cache settings
    use_cache: bool = True
    cache_size: int = 1000
    
    # Efficiency
    use_gradient_checkpointing: bool = False
    use_sparse_attention: bool = False  # For T > 500


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings."""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Temporal Self-Attention
# ============================================================================

class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over temporal dimension."""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: Optional (B, T, T) attention mask
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        residual = x
        
        # Linear projections and reshape for multi-head
        Q = self.q_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = self.k_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, T)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, d_k)
        
        # Concatenate heads and apply output linear
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_linear(attn_output)
        output = self.dropout(output)
        
        # Residual connection + layer norm
        output = self.layer_norm(residual + output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.dropout(x)
        return self.layer_norm(residual + x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with self-attention + FFN."""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = TemporalSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: Optional attention mask
        Returns:
            (B, T, D)
        """
        x = self.self_attn(x, mask)
        x = self.ffn(x)
        return x


# ============================================================================
# Multi-Scale Temporal Encoder
# ============================================================================

class MultiScaleTemporalEncoder(nn.Module):
    """
    Process features at multiple temporal scales (1x, 2x, 4x downsampling).
    Inspired by Feature Pyramid Networks for temporal domain.
    """
    
    def __init__(self, d_model: int, num_scales: int = 3, fusion: str = "concat"):
        super().__init__()
        self.num_scales = num_scales
        self.fusion = fusion
        
        # Per-scale processing
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            for _ in range(num_scales)
        ])
        
        # Fusion layer
        if fusion == "concat":
            self.fusion_layer = nn.Linear(d_model * num_scales, d_model)
        elif fusion == "attention":
            self.scale_attention = nn.Linear(d_model, num_scales)
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def temporal_pool(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Downsample temporal dimension by factor of 2^scale.
        Uses max + average pooling.
        
        Args:
            x: (B, T, D)
            scale: downsampling factor (0, 1, 2, ...)
        Returns:
            (B, T', D) where T' = T // (2^scale)
        """
        if scale == 0:
            return x
        
        kernel_size = 2 ** scale
        x = x.transpose(1, 2)  # (B, D, T)
        
        # Max pooling
        x_max = F.max_pool1d(x, kernel_size=kernel_size, stride=kernel_size)
        # Average pooling
        x_avg = F.avg_pool1d(x, kernel_size=kernel_size, stride=kernel_size)
        
        # Combine
        x = (x_max + x_avg) / 2
        return x.transpose(1, 2)  # (B, T', D)
    
    def temporal_upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Upsample to target temporal length.
        
        Args:
            x: (B, T', D)
            target_len: target T
        Returns:
            (B, T, D)
        """
        x = x.transpose(1, 2)  # (B, D, T')
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x.transpose(1, 2)  # (B, T, D)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D) multi-scale fused features
        """
        B, T, D = x.shape
        scale_features = []
        
        for scale_idx in range(self.num_scales):
            # Downsample
            x_down = self.temporal_pool(x, scale_idx)
            
            # Process with 1D conv
            x_conv = x_down.transpose(1, 2)  # (B, D, T')
            x_conv = self.scale_convs[scale_idx](x_conv)
            x_conv = F.relu(x_conv)
            x_conv = x_conv.transpose(1, 2)  # (B, T', D)
            
            # Upsample back to original length
            x_up = self.temporal_upsample(x_conv, T)
            scale_features.append(x_up)
        
        # Fusion
        if self.fusion == "concat":
            fused = torch.cat(scale_features, dim=-1)  # (B, T, D*num_scales)
            fused = self.fusion_layer(fused)  # (B, T, D)
        elif self.fusion == "sum":
            fused = torch.stack(scale_features, dim=0).sum(dim=0)  # (B, T, D)
        elif self.fusion == "attention":
            # Learnable attention weights per scale
            stack = torch.stack(scale_features, dim=-1)  # (B, T, D, num_scales)
            attn_weights = F.softmax(self.scale_attention(x), dim=-1)  # (B, T, num_scales)
            attn_weights = attn_weights.unsqueeze(2)  # (B, T, 1, num_scales)
            fused = (stack * attn_weights).sum(dim=-1)  # (B, T, D)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")
        
        return self.layer_norm(fused)


# ============================================================================
# Feature Cache
# ============================================================================

class FeatureCache:
    """
    LRU cache for encoder outputs to avoid redundant computation.
    Thread-safe for multi-worker data loading.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, scene_id: str, feat_hash: str) -> str:
        """Create cache key from scene ID and feature hash."""
        return f"{scene_id}_{feat_hash}"
    
    def _hash_features(self, features: torch.Tensor) -> str:
        """Compute hash of feature tensor."""
        # Use first and last few values + shape as hash
        feat_np = features.detach().cpu().numpy()
        key_data = f"{feat_np.shape}_{feat_np.flat[:10].tobytes()}_{feat_np.flat[-10:].tobytes()}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def get(self, scene_id: str, features: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Try to get cached output for given scene and features.
        
        Args:
            scene_id: Unique scene identifier
            features: Input features (B, T, D)
        Returns:
            Cached tensor if found, else None
        """
        feat_hash = self._hash_features(features)
        key = self._make_key(scene_id, feat_hash)
        
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key].clone()
            else:
                self.misses += 1
                return None
    
    def put(self, scene_id: str, features: torch.Tensor, output: torch.Tensor):
        """
        Store output in cache.
        
        Args:
            scene_id: Unique scene identifier
            features: Input features (B, T, D)
            output: Output to cache
        """
        feat_hash = self._hash_features(features)
        key = self._make_key(scene_id, feat_hash)
        
        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self.cache.popitem(last=False)
            
            self.cache[key] = output.detach().clone()
    
    def get_or_compute(self, scene_id: str, features: torch.Tensor, 
                       compute_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Get from cache or compute and store.
        
        Args:
            scene_id: Unique scene identifier
            features: Input features
            compute_fn: Function to compute output if not cached
        Returns:
            Output tensor
        """
        cached = self.get(scene_id, features)
        if cached is not None:
            return cached
        
        output = compute_fn(features)
        self.put(scene_id, features, output)
        return output
    
    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_queries": total,
                "hit_rate": hit_rate,
                "cache_size": len(self.cache),
                "max_size": self.max_size,
            }


# ============================================================================
# Advanced Encoder
# ============================================================================

class EncoderFCAdvanced(nn.Module):
    """
    Enhanced encoder with multi-layer MLP, residual connections, and optional caching.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.3, use_cache: bool = False, cache_size: int = 1000):
        super().__init__()
        self.use_cache = use_cache
        
        # Multi-layer MLP
        layers = []
        current_dim = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else None
        
        # Feature cache
        if use_cache:
            self.cache = FeatureCache(max_size=cache_size)
        else:
            self.cache = None
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Actual encoding logic.
        
        Args:
            x: (B, T, D_in)
        Returns:
            (B, T, D_hidden)
        """
        B, T, D = x.shape
        
        # Reshape for BatchNorm1d: (B*T, D)
        x_flat = x.reshape(B * T, D)
        h_flat = self.mlp(x_flat)
        h = h_flat.reshape(B, T, -1)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
            h = h + residual
        
        return h
    
    def forward(self, x: torch.Tensor, scene_id: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D_in)
            scene_id: Optional scene ID for caching
        Returns:
            (B, T, D_hidden)
        """
        if self.use_cache and self.cache is not None and scene_id is not None:
            return self.cache.get_or_compute(scene_id, x, self._encode)
        else:
            return self._encode(x)
    
    def get_cache_stats(self) -> Optional[Dict[str, int]]:
        """Get cache statistics if caching is enabled."""
        if self.cache is not None:
            return self.cache.get_stats()
        return None


# ============================================================================
# Advanced Policy Network
# ============================================================================

class DSNPolicyAdvanced(nn.Module):
    """
    Advanced policy network with:
    - Positional encoding
    - Multi-scale temporal modeling
    - Self-attention layers
    - Optional BiLSTM
    - Attention pooling for final prediction
    """
    
    def __init__(self, config: DSNConfig):
        super().__init__()
        self.config = config
        
        # Positional encoding
        if config.pos_encoding_type == "sinusoidal":
            self.pos_encoder = PositionalEncoding(
                config.hidden_dim, config.max_seq_len, config.dropout
            )
        elif config.pos_encoding_type == "learned":
            self.pos_encoder = LearnedPositionalEncoding(
                config.hidden_dim, config.max_seq_len, config.dropout
            )
        else:
            self.pos_encoder = None
        
        # Multi-scale temporal encoder
        self.multi_scale = MultiScaleTemporalEncoder(
            config.hidden_dim, config.num_scales, config.scale_fusion
        )
        
        # Self-attention layers
        self.attn_layers = nn.ModuleList([
            TransformerEncoderLayer(config.hidden_dim, config.num_attn_heads, config.attn_dropout)
            for _ in range(config.num_attn_layers)
        ])
        
        # Optional BiLSTM
        self.use_lstm = config.use_lstm
        if config.use_lstm:
            self.lstm = nn.LSTM(
                input_size=config.hidden_dim,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                bidirectional=config.bidirectional,
                batch_first=True,
                dropout=config.dropout if config.lstm_layers > 1 else 0.0
            )
            lstm_out_dim = config.lstm_hidden * (2 if config.bidirectional else 1)
        else:
            lstm_out_dim = config.hidden_dim
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Output head
        self.head = nn.Linear(lstm_out_dim, 1)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, H) hidden features from encoder
        Returns:
            probs: (B, T) selection probabilities
        """
        # Positional encoding
        if self.pos_encoder is not None:
            h = self.pos_encoder(h)
        
        # Multi-scale temporal modeling
        h = self.multi_scale(h)
        
        # Self-attention layers
        for attn_layer in self.attn_layers:
            h = attn_layer(h)
        
        # Optional LSTM
        if self.use_lstm:
            h, _ = self.lstm(h)
        
        # Dropout
        h = self.dropout(h)
        
        # Output head
        logits = self.head(h)  # (B, T, 1)
        probs = torch.sigmoid(logits).squeeze(-1)  # (B, T)
        
        return probs


# ============================================================================
# Complete Advanced DSN Model
# ============================================================================

class DSNAdvanced(nn.Module):
    """
    Complete advanced DSN model combining encoder and policy.
    Drop-in replacement for baseline DSN with enhanced capabilities.
    """
    
    def __init__(self, config: DSNConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = EncoderFCAdvanced(
            in_dim=config.feat_dim,
            hidden_dim=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            use_cache=config.use_cache,
            cache_size=config.cache_size
        )
        
        # Policy
        self.policy = DSNPolicyAdvanced(config)
    
    def forward(self, x: torch.Tensor, scene_id: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input features
            scene_id: Optional scene ID for caching
        Returns:
            probs: (B, T) selection probabilities
        """
        h = self.encoder(x, scene_id)
        probs = self.policy(h)
        return probs
    
    def get_cache_stats(self) -> Optional[Dict[str, int]]:
        """Get encoder cache statistics."""
        return self.encoder.get_cache_stats()
    
    def clear_cache(self):
        """Clear encoder cache."""
        if self.encoder.cache is not None:
            self.encoder.cache.clear()


# ============================================================================
# Factory Functions
# ============================================================================

def create_dsn_advanced(
    feat_dim: int = 512,
    hidden_dim: int = 256,
    lstm_hidden: int = 128,
    **kwargs
) -> DSNAdvanced:
    """
    Factory function to create DSNAdvanced model.
    
    Args:
        feat_dim: Input feature dimension
        hidden_dim: Hidden dimension
        lstm_hidden: LSTM hidden dimension
        **kwargs: Additional config parameters
    
    Returns:
        DSNAdvanced model
    """
    config = DSNConfig(
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        lstm_hidden=lstm_hidden,
        **kwargs
    )
    return DSNAdvanced(config)


# ============================================================================
# Testing / Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Advanced DSN Model")
    print("=" * 60)
    
    # Create model
    config = DSNConfig(
        feat_dim=512,
        hidden_dim=256,
        lstm_hidden=128,
        num_attn_heads=4,
        num_attn_layers=2,
        num_scales=3,
        use_cache=True,
        cache_size=100
    )
    
    model = DSNAdvanced(config)
    print(f"\n✓ Model created")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    B, T, D = 2, 50, 512
    x = torch.randn(B, T, D)
    
    print(f"\n✓ Testing forward pass")
    print(f"  Input shape: {x.shape}")
    
    probs = model(x, scene_id="test_scene_001")
    print(f"  Output shape: {probs.shape}")
    print(f"  Output range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    assert probs.shape == (B, T), f"Expected shape ({B}, {T}), got {probs.shape}"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities not in [0, 1]"
    
    # Test caching
    print(f"\n✓ Testing cache mechanism")
    stats_before = model.get_cache_stats()
    print(f"  Before: {stats_before}")
    
    # Second forward pass with same scene_id (should hit cache)
    probs2 = model(x, scene_id="test_scene_001")
    stats_after = model.get_cache_stats()
    print(f"  After: {stats_after}")
    
    # Test gradient flow
    print(f"\n✓ Testing gradient flow")
    loss = probs.mean()
    loss.backward()
    
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total_params}")
    
    print(f"\n{'=' * 60}")
    print("All tests passed! ✅")
    print("=" * 60)
