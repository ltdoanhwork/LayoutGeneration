"""
Motion Fusion Module for DSN

Implements cross-attention mechanism to fuse motion features with appearance features.
Motion features act as queries, attending to CLIP appearance features as keys/values.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionCrossAttention(nn.Module):
    """
    Cross-attention module: motion features attend to CLIP features.
    
    Architecture:
        Query: motion_feats (B, T, D_m)
        Key/Value: clip_feats (B, T, D_clip)
        Output: fused_feats (B, T, D_out)
    
    This allows motion information to selectively attend to relevant visual features,
    creating a rich fused representation for keyframe selection.
    """
    
    def __init__(
        self,
        motion_dim: int,
        clip_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        """
        Args:
            motion_dim: Dimension of motion features
            clip_dim: Dimension of CLIP features
            output_dim: Dimension of output fused features
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_gating: Whether to use gating mechanism for adaptive fusion
        """
        super().__init__()
        
        self.motion_dim = motion_dim
        self.clip_dim = clip_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_gating = use_gating
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Projection layers
        self.query_proj = nn.Linear(motion_dim, output_dim)
        self.key_proj = nn.Linear(clip_dim, output_dim)
        self.value_proj = nn.Linear(clip_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        # Normalization
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Optional gating mechanism for adaptive fusion
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(motion_dim + clip_dim, output_dim),
                nn.Sigmoid()
            )
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout)
        )
        
        # Project motion to output_dim for residual connection
        self.motion_residual_proj = nn.Linear(motion_dim, output_dim)
    
    def forward(
        self,
        motion_feats: torch.Tensor,
        clip_feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            motion_feats: (B, T, D_m) motion features
            clip_feats: (B, T, D_clip) CLIP appearance features
            mask: Optional (B, T) mask for padding
        
        Returns:
            fused_feats: (B, T, D_out) fused features
        """
        B, T, _ = motion_feats.shape
        
        # Project to query, key, value
        Q = self.query_proj(motion_feats)  # (B, T, D_out)
        K = self.key_proj(clip_feats)      # (B, T, D_out)
        V = self.value_proj(clip_feats)    # (B, T, D_out)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D_h)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D_h)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D_h)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, D_h)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.output_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Residual connection with projected motion features
        motion_proj = self.motion_residual_proj(motion_feats)
        fused = self.layer_norm1(motion_proj + attn_output)
        
        # Optional gating
        if self.use_gating:
            gate_input = torch.cat([motion_feats, clip_feats], dim=-1)
            gate_weights = self.gate(gate_input)  # (B, T, D_out)
            fused = gate_weights * fused + (1 - gate_weights) * motion_proj
        
        # Feedforward network with residual
        ffn_output = self.ffn(fused)
        fused = self.layer_norm2(fused + ffn_output)
        
        return fused


class SimpleMotionFusion(nn.Module):
    """
    Simpler alternative: concatenation + projection.
    Use this for ablation studies or if cross-attention is too complex.
    """
    
    def __init__(
        self,
        motion_dim: int,
        clip_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(motion_dim + clip_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        motion_feats: torch.Tensor,
        clip_feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            motion_feats: (B, T, D_m)
            clip_feats: (B, T, D_clip)
            mask: Unused, for API compatibility
        
        Returns:
            fused_feats: (B, T, D_out)
        """
        # Concatenate and project
        concat = torch.cat([motion_feats, clip_feats], dim=-1)
        fused = self.fusion(concat)
        return fused


def test_motion_fusion():
    """Test the motion fusion modules."""
    print("Testing Motion Fusion Modules")
    print("=" * 60)
    
    # Test parameters
    B, T = 2, 10
    motion_dim = 128
    clip_dim = 256
    output_dim = 256
    
    # Create dummy data
    motion_feats = torch.randn(B, T, motion_dim)
    clip_feats = torch.randn(B, T, clip_dim)
    
    # Test MotionCrossAttention
    print("\n[Test 1] MotionCrossAttention")
    cross_attn = MotionCrossAttention(
        motion_dim=motion_dim,
        clip_dim=clip_dim,
        output_dim=output_dim,
        num_heads=4,
        use_gating=True
    )
    
    fused = cross_attn(motion_feats, clip_feats)
    print(f"  Input motion: {motion_feats.shape}")
    print(f"  Input CLIP: {clip_feats.shape}")
    print(f"  Output fused: {fused.shape}")
    print(f"  Parameters: {sum(p.numel() for p in cross_attn.parameters()):,}")
    
    # Test with mask
    print("\n[Test 2] MotionCrossAttention with mask")
    mask = torch.ones(B, T)
    mask[:, T//2:] = 0  # Mask second half
    fused_masked = cross_attn(motion_feats, clip_feats, mask)
    print(f"  Output with mask: {fused_masked.shape}")
    
    # Test SimpleMotionFusion
    print("\n[Test 3] SimpleMotionFusion")
    simple_fusion = SimpleMotionFusion(
        motion_dim=motion_dim,
        clip_dim=clip_dim,
        output_dim=output_dim
    )
    
    fused_simple = simple_fusion(motion_feats, clip_feats)
    print(f"  Output fused: {fused_simple.shape}")
    print(f"  Parameters: {sum(p.numel() for p in simple_fusion.parameters()):,}")
    
    # Test gradient flow
    print("\n[Test 4] Gradient flow")
    loss = fused.mean()
    loss.backward()
    print(f"  Backward pass successful")
    
    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    test_motion_fusion()
