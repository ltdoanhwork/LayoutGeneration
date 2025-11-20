from __future__ import annotations
from typing import List
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None

def compute_flow_magnitude_robust(frames: List[np.ndarray]) -> np.ndarray:
    """Compute (T,) forward flow magnitude with fallback chain: TV-L1 -> DIS -> Farneback."""
    T = len(frames)
    if T < 2 or cv2 is None:
        return np.zeros((T,), np.float32)

    def gray_u8(img):
        """Convert to grayscale uint8 for DIS/TV-L1."""
        if img.ndim==3 and img.shape[2]==3:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = img[...,0] if img.ndim==3 else img
        if g.dtype != np.uint8:
            if g.max() <= 1.0:
                g = (g * 255).astype(np.uint8)
            else:
                g = g.astype(np.uint8)
        return g
    
    def gray_f32(img):
        """Convert to grayscale float32 for Farneback."""
        if img.ndim==3 and img.shape[2]==3:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = img[...,0] if img.ndim==3 else img
        g = g.astype(np.float32)
        if g.max() > 1.0:
            g /= 255.0
        return g

    method = None
    try:
        if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
            method = ("tvl1", cv2.optflow.DualTVL1OpticalFlow_create())
    except Exception: pass
    if method is None:
        try:
            if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DualTVL1"):
                method = ("tvl1_old", cv2.optflow.createOptFlow_DualTVL1())
        except Exception: pass
    if method is None:
        try:
            if hasattr(cv2, "DISOpticalFlow_create"):
                method = ("dis", cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST))
        except Exception: pass
    if method is None:
        method = ("farneback", None)

    mags = []
    for t in range(T-1):
        name, obj = method
        if name in ("tvl1","tvl1_old","dis"):
            # DIS and TV-L1 require uint8 grayscale
            a, b = gray_u8(frames[t]), gray_u8(frames[t+1])
            flow = obj.calc(a, b, None)
        else:
            # Farneback works with float32
            a, b = gray_f32(frames[t]), gray_f32(frames[t+1])
            flow = cv2.calcOpticalFlowFarneback(a,b,None,0.5,3,15,3,5,1.2,0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
        mags.append(float(mag))
    mags.append(mags[-1] if mags else 0.0)
    return np.array(mags, np.float32)

def main():
    """Test optical flow computation with various synthetic scenarios."""
    print("=" * 60)
    print("Testing Optical Flow Computation")
    print("=" * 60)
    
    if cv2 is None:
        print("ERROR: OpenCV not available. Cannot run tests.")
        return
    
    # Test parameters
    H, W = 128, 128
    T = 5  # Number of frames
    
    # Test 1: Static frames (no motion)
    print("\n[Test 1] Static frames (no motion expected)")
    static_frames = [np.ones((H, W, 3), dtype=np.uint8) * 128 for _ in range(T)]
    flow_static = compute_flow_magnitude_robust(static_frames)
    print(f"  Shape: {flow_static.shape}")
    print(f"  Flow magnitudes: {flow_static}")
    print(f"  Mean flow: {flow_static.mean():.6f} (should be ~0)")
    
    # Test 2: Horizontal motion (moving square)
    print("\n[Test 2] Horizontal motion (moving square)")
    moving_frames = []
    for t in range(T):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        x_pos = 20 + t * 10  # Move 10 pixels right each frame
        cv2.rectangle(frame, (x_pos, 40), (x_pos + 30, 70), (255, 255, 255), -1)
        moving_frames.append(frame)
    flow_horizontal = compute_flow_magnitude_robust(moving_frames)
    print(f"  Shape: {flow_horizontal.shape}")
    print(f"  Flow magnitudes: {flow_horizontal}")
    print(f"  Mean flow: {flow_horizontal.mean():.6f} (should be > 0)")
    
    # Test 3: Vertical motion (moving circle)
    print("\n[Test 3] Vertical motion (moving circle)")
    vertical_frames = []
    for t in range(T):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        y_pos = 20 + t * 15  # Move 15 pixels down each frame
        cv2.circle(frame, (W // 2, y_pos), 20, (200, 200, 200), -1)
        vertical_frames.append(frame)
    flow_vertical = compute_flow_magnitude_robust(vertical_frames)
    print(f"  Shape: {flow_vertical.shape}")
    print(f"  Flow magnitudes: {flow_vertical}")
    print(f"  Mean flow: {flow_vertical.mean():.6f} (should be > 0)")
    
    # Test 4: Random noise (chaotic motion)
    print("\n[Test 4] Random noise (chaotic motion)")
    noise_frames = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(T)]
    flow_noise = compute_flow_magnitude_robust(noise_frames)
    print(f"  Shape: {flow_noise.shape}")
    print(f"  Flow magnitudes: {flow_noise}")
    print(f"  Mean flow: {flow_noise.mean():.6f}")
    
    # Test 5: Edge case - single frame
    print("\n[Test 5] Edge case - single frame")
    single_frame = [np.ones((H, W, 3), dtype=np.uint8) * 128]
    flow_single = compute_flow_magnitude_robust(single_frame)
    print(f"  Shape: {flow_single.shape}")
    print(f"  Flow magnitudes: {flow_single}")
    print(f"  Expected: all zeros for T < 2")
    
    # Test 6: Edge case - empty list
    print("\n[Test 6] Edge case - empty list")
    flow_empty = compute_flow_magnitude_robust([])
    print(f"  Shape: {flow_empty.shape}")
    print(f"  Flow magnitudes: {flow_empty}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Static motion:      {flow_static.mean():.6f}")
    print(f"  Horizontal motion:  {flow_horizontal.mean():.6f}")
    print(f"  Vertical motion:    {flow_vertical.mean():.6f}")
    print(f"  Random noise:       {flow_noise.mean():.6f}")
    print("=" * 60)
    
    # Optional: Visualize one test case
    try:
        import matplotlib.pyplot as plt
        print("\n[Visualization] Saving flow magnitude plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Static
        axes[0, 0].plot(flow_static, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Static Frames', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Frame Index')
        axes[0, 0].set_ylabel('Flow Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Horizontal
        axes[0, 1].plot(flow_horizontal, 'o-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_title('Horizontal Motion', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Frame Index')
        axes[0, 1].set_ylabel('Flow Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Vertical
        axes[1, 0].plot(flow_vertical, 'o-', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_title('Vertical Motion', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Frame Index')
        axes[1, 0].set_ylabel('Flow Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Noise
        axes[1, 1].plot(flow_noise, 'o-', linewidth=2, markersize=8, color='red')
        axes[1, 1].set_title('Random Noise', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Frame Index')
        axes[1, 1].set_ylabel('Flow Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = '/tmp/optical_flow_test.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
        plt.close()
        
    except ImportError:
        print("\n[Note] matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"\n[Warning] Visualization failed: {e}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
