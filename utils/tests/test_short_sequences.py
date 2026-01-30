#!/usr/bin/env python3
"""Quick test for short sequence handling in advanced DSN"""

import torch
from src.models.dsn_advanced import DSNAdvanced, DSNConfig

def test_short_sequences():
    print("Testing short sequence handling...")
    
    config = DSNConfig(
        feat_dim=512,
        hidden_dim=256,
        lstm_hidden=128,
        num_scales=3,  # Will try 1x, 2x, 4x
        use_cache=False
    )
    
    model = DSNAdvanced(config)
    model.eval()
    
    # Test various short sequence lengths
    test_cases = [
        (1, 2),   # T=2 (< 4, will fail on 4x scale without fix)
        (1, 3),   # T=3
        (1, 5),   # T=5
        (1, 10),  # T=10
        (1, 50),  # T=50 (normal case)
    ]
    
    for B, T in test_cases:
        print(f"\n  Testing B={B}, T={T}...")
        x = torch.randn(B, T, 512)
        
        try:
            with torch.no_grad():
                probs = model(x)
            print(f"    ✓ Success! Output shape: {probs.shape}")
            assert probs.shape == (B, T), f"Shape mismatch: {probs.shape} != ({B}, {T})"
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            return False
    
    print("\n✅ All short sequence tests passed!")
    return True

if __name__ == "__main__":
    success = test_short_sequences()
    exit(0 if success else 1)
