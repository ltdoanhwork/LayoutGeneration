#!/usr/bin/env python3
"""
Quick test to verify Advanced DSN model runs on GPU correctly.
"""

import torch
from src.models.dsn_advanced import DSNAdvanced, DSNConfig

def test_gpu_usage():
    print("=" * 60)
    print("Testing GPU Usage for Advanced DSN")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    device = torch.device("cuda:0")
    print(f"\n✓ Using device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
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
    
    print(f"\n✓ Creating model...")
    model = DSNAdvanced(config).to(device)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Check if model is on GPU
    model_device = next(model.parameters()).device
    print(f"\n✓ Model device: {model_device}")
    if model_device.type != "cuda":
        print("❌ Model is NOT on GPU!")
        return False
    
    # Create input tensor on GPU
    B, T, D = 2, 50, 512
    x = torch.randn(B, T, D, device=device)
    print(f"\n✓ Input tensor created on GPU")
    print(f"  Shape: {x.shape}")
    print(f"  Device: {x.device}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Forward pass
    print(f"\n✓ Running forward pass...")
    with torch.no_grad():
        probs = model(x, scene_id="test_scene")
    
    print(f"  Output shape: {probs.shape}")
    print(f"  Output device: {probs.device}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Check if output is on GPU
    if probs.device.type != "cuda":
        print("❌ Output is NOT on GPU!")
        return False
    
    # Test backward pass
    print(f"\n✓ Testing backward pass...")
    loss = probs.mean()
    loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total_params}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Verify gradients are on GPU
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.device.type != "cuda":
                print(f"❌ Gradient for {name} is NOT on GPU!")
                return False
    
    print(f"\n✓ All gradients are on GPU")
    
    # Final memory check
    print(f"\n✓ Final GPU memory usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! Model runs correctly on GPU.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_gpu_usage()
    exit(0 if success else 1)
