import os
from .u2net import U2NET # Import model từ U2NET repo
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

# Load model
net = U2NET(3, 1)
net.load_state_dict(torch.load('/home/serverai/ltdoanh/LayoutGeneration/repos/CAST/saved_models/u2net.pth', map_location='cpu'))
net.eval()

def preprocess_image(image):
    """Preprocess image cho U2NET: resize, normalize, to tensor."""
    # Resize về 320x320 (standard cho U2NET)
    h, w = image.shape[:2]
    resized = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
    
    # Normalize: BGR to RGB, subtract mean, divide std
    tmp_img = resized.astype(np.float32) / 255.0
    tmp_img = tmp_img.transpose((2, 0, 1))  # HWC to CHW
    tmp_img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)  # Mean
    tmp_img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)  # Std
    
    # To tensor
    inputs = torch.from_numpy(tmp_img).unsqueeze(0)  # Add batch dim
    # Nếu GPU: inputs = inputs.cuda()
    
    return inputs, h, w

def predict_mask(net, inputs):
    """Predict mask từ model."""
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(inputs)
        # U2NET output nhiều scales, dùng d1 (highest resolution)
        pred = d1[:, 0, :, :]  # Shape: [1, 320, 320]
        pred = F.interpolate(pred.unsqueeze(1), size=(320, 320), mode='bilinear', align_corners=False)
        pred = pred.squeeze().cpu().numpy()  # To numpy
    return pred

def refine_mask(mask, orig_h, orig_w):
    """Refine mask: resize, fill holes, smooth edges, keep largest component.
    
    Output convention: WHITE (255) = foreground/shape, BLACK (0) = background
    This matches voronoi_layout.py expectation: mask_binary > 127 = shape
    
    Args:
        mask: Raw U2-Net prediction (float32, 0-1)
        orig_h: Original image height
        orig_w: Original image width
    
    Returns:
        mask_final: Binary mask (uint8) with:
            - 255 = foreground/shape (area to fill with images)
            - 0 = background
            - All internal holes filled
    """
    # Resize về original size
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # Threshold để binary
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
    
    # ===== STEP 1: Fill ALL Holes using Flood Fill =====
    # The strategy: flood fill from corners (background), then invert
    # This ensures ALL internal holes are filled, not just small ones
    
    # Create a larger canvas for flood fill (need 2px border for floodFill)
    h, w = mask_binary.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Copy mask to center
    work_mask = mask_binary.copy()
    
    # Flood fill from corners (these are guaranteed background)
    # Use cv2.floodFill to mark all connected background as 128
    cv2.floodFill(work_mask, flood_mask, (0, 0), 128)  # Top-left
    cv2.floodFill(work_mask, flood_mask, (w-1, 0), 128)  # Top-right
    cv2.floodFill(work_mask, flood_mask, (0, h-1), 128)  # Bottom-left
    cv2.floodFill(work_mask, flood_mask, (w-1, h-1), 128)  # Bottom-right
    
    # Now: 128 = external background, 0 = holes (internal), 255 = foreground
    # Fill holes: anything that's 0 (internal) becomes foreground (255)
    mask_filled = work_mask.copy()
    mask_filled[work_mask == 0] = 255  # Fill holes
    mask_filled[work_mask == 128] = 0  # External background stays 0
    
    # ===== STEP 2: Keep Only Largest Connected Component =====
    # This removes stray noise/artifacts
    mask_temp = (mask_filled > 127).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_temp, connectivity=8)
    
    if num_labels > 1:
        # Find largest component (excluding background=0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_largest = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask_largest = mask_filled
    
    # ===== STEP 3: Morphological Cleanup =====
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)
    
    # Remove small protrusions (OPEN)
    mask_clean = cv2.morphologyEx(mask_largest, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Fill small remaining gaps (CLOSE) - aggressive to ensure no holes
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # ===== STEP 4: Smooth Edges =====
    # Gaussian blur followed by re-threshold for smooth boundaries
    mask_smooth = cv2.GaussianBlur(mask_clean.astype(np.float32), (7, 7), 0)
    mask_smooth = (mask_smooth > 127).astype(np.uint8) * 255
    
    # ===== STEP 5: Final Hole Check (Safety) =====
    # Use contour hierarchy to detect and fill any remaining holes
    contours, hierarchy = cv2.findContours(mask_smooth, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # If hierarchy exists and there are internal contours (holes), fill them
    if hierarchy is not None and len(contours) > 0:
        mask_final = mask_smooth.copy()
        for idx, (contour, hier) in enumerate(zip(contours, hierarchy[0])):
            # hier = [next, prev, child, parent]
            # If parent >= 0, this is an internal contour (hole)
            if hier[3] >= 0:  # Has parent = it's a hole
                cv2.drawContours(mask_final, [contour], 0, 255, -1)  # Fill hole
    else:
        mask_final = mask_smooth
    
    # Ensure proper uint8 output
    mask_final = np.clip(mask_final, 0, 255).astype(np.uint8)
    
    print(f"  [Mask] Output: WHITE (255) = foreground, BLACK (0) = background")
    print(f"  [Mask] Foreground pixels: {np.sum(mask_final > 127):,} / {mask_final.size:,}")
    
    return mask_final

def extract_object(image, mask):
    """Extract object: composite foreground với background transparent."""
    # Convert mask to 3-channel
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Create RGBA image (add alpha channel)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask  # Alpha channel từ mask
    
    return rgba

def get_mask(image):
    """
    Main function: Nhận vào ảnh, trả về mask.
    
    Args:
        image: Input image (numpy array, BGR format from cv2.imread)
               hoặc str path to image file
    
    Returns:
        mask: Binary mask (uint8, 0-255)
              - 255 = foreground/object
              - 0 = background
    
    Example:
        import cv2
        from CAST.utils.get_mask import get_mask
        
        # Option 1: Load image first
        img = cv2.imread('path/to/image.jpg')
        mask = get_mask(img)
        
        # Option 2: Pass path directly
        mask = get_mask('path/to/image.jpg')
        
        # Save mask
        cv2.imwrite('mask.png', mask)
    """
    # If input is path string, load image
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found: {image}")
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Failed to load image: {image}")
    
    # Validate input
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy array or path string, got {type(image)}")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H, W, 3), got {image.shape}")
    
    print(f"[get_mask] Processing image: {image.shape}")
    
    # Step 1: Preprocess
    inputs, orig_h, orig_w = preprocess_image(image)
    
    # Step 2: Predict
    print(f"[get_mask] Running U2-Net prediction...")
    raw_mask = predict_mask(net, inputs)
    
    # Step 3: Refine
    print(f"[get_mask] Refining mask...")
    mask = refine_mask(raw_mask, orig_h, orig_w)
    
    print(f"[get_mask] Done! Mask shape: {mask.shape}")
    
    return mask