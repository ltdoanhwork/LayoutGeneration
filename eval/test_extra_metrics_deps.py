#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to check if extra_metrics dependencies are available
"""
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✅ cv2 (opencv) available")
    except ImportError as e:
        print(f"❌ cv2 not available: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy available")
    except ImportError as e:
        print(f"❌ numpy not available: {e}")
        return False
    
    try:
        from src.distance_selector.registry import create_metric
        print("✅ src.distance_selector.registry available")
        
        # Try to create LPIPS metric
        try:
            metric = create_metric("lpips", net="alex", device="cpu")
            print("✅ LPIPS metric can be created")
        except Exception as e:
            print(f"⚠️  LPIPS metric creation failed: {e}")
            print("   This might be because 'lpips' package is not installed")
            print("   Install with: pip install lpips")
            
    except ImportError as e:
        print(f"❌ src.distance_selector.registry not available: {e}")
        return False
    
    try:
        from src.metrics.ms_swd import ms_swd_color
        print("✅ src.metrics.ms_swd available")
    except ImportError as e:
        print(f"❌ src.metrics.ms_swd not available: {e}")
        print(f"   You may need to implement this module")
        return False
    
    return True


def test_lpips_package():
    """Test if lpips package is installed"""
    print("\nTesting lpips package...")
    try:
        import lpips
        print("✅ lpips package is installed")
        
        # Try to create a model
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            loss_fn = lpips.LPIPS(net='alex').to(device)
            print(f"✅ LPIPS model created successfully on {device}")
            return True
        except Exception as e:
            print(f"⚠️  LPIPS model creation failed: {e}")
            return False
            
    except ImportError:
        print("❌ lpips package not installed")
        print("   Install with: pip install lpips")
        return False


def main():
    print("="*80)
    print("EXTRA METRICS DEPENDENCY CHECK")
    print("="*80)
    
    imports_ok = test_imports()
    lpips_ok = test_lpips_package()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if imports_ok and lpips_ok:
        print("✅ All dependencies are available")
        print("   You can run extra_metrics.py")
        return 0
    else:
        print("❌ Some dependencies are missing")
        print("\nTo fix:")
        if not lpips_ok:
            print("  1. Install lpips: pip install lpips")
        print("  2. Make sure src.metrics.ms_swd is implemented")
        return 1


if __name__ == "__main__":
    sys.exit(main())
