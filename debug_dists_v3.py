import torch
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(".")

from src.distance_selector.registry import create_metric

def test_dists():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create metric
    metric = create_metric("dists", device=device)
    
    # Create dummy images (random)
    # H, W = 256, 256
    # img1 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    # img2 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    # Better: Load real images if possible, or use distinct noise
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img1[:, :128, :] = 255 # White left
    
    img2 = np.zeros((256, 256, 3), dtype=np.uint8)
    img2[:, 128:, :] = 255 # White right
    
    print("Testing pair_distance...")
    t1 = metric.preprocess_bgr(img1)
    t2 = metric.preprocess_bgr(img2)
    
    dist = metric.pair_distance(t1, t2)
    print(f"Distance (pair): {dist}")
    
    # Test batching logic from script
    print("\nTesting batch logic...")
    # Simulate batch of 2
    curr = torch.cat([t1, t2], dim=0) # (2, 3, 256, 256)
    target = t1 # (1, 3, 256, 256)
    
    # Expand
    B_curr = 2
    K = 1
    
    # (B, 1, C, H, W) -> (B, K, C, H, W)
    curr_exp = curr.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(-1, *curr.shape[1:])
    target_exp = target.unsqueeze(0).expand(B_curr, -1, -1, -1, -1).reshape(-1, *target.shape[1:])
    
    print(f"Shapes: {curr_exp.shape}, {target_exp.shape}")
    
    model = metric._dists
    d = model(curr_exp, target_exp)
    print(f"Raw output: {d}")
    print(f"Raw output shape: {d.shape}")
    
    if d.dim() > 1:
        d = d.view(d.size(0))
        
    print(f"Processed: {d}")

if __name__ == "__main__":
    test_dists()
