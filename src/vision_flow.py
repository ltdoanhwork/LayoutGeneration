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

    def gray32(img):
        if img.ndim==3 and img.shape[2]==3:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = img[...,0] if img.ndim==3 else img
        g = g.astype(np.float32); 
        if g.max() > 1.0: g/=255.0
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
        a, b = gray32(frames[t]), gray32(frames[t+1])
        name, obj = method
        if name in ("tvl1","tvl1_old","dis"):
            flow = obj.calc(a, b, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(a,b,None,0.5,3,15,3,5,1.2,0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
        mags.append(float(mag))
    mags.append(mags[-1] if mags else 0.0)
    return np.array(mags, np.float32)
