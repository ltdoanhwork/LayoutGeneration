"""
Saliency detection utilities for collage assembly.
Includes U2-Net based saliency and fast heuristic methods.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from os.path import isfile

# Import U2-Net - handle both local and package import
try:
    from utils.u2net import U2NET
except ModuleNotFoundError:
    try:
        from repos.Colla.utils.u2net import U2NET
    except ModuleNotFoundError:
        U2NET = None


# === U2-NET SALIENCY MODEL (SINGLETON) ===
class U2NetSaliency:
    """Global singleton for U2-Net saliency detection."""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            if U2NET is None:
                raise ImportError("U2NET model not available")
            print("[U2NetSaliency] Loading U2-Net model...")
            self._model = U2NET(3, 1)
            # Load weight from existing path
            weight_path = '/home/serverai/ltdoanh/LayoutGeneration/create_mask_for_user/U-2-Net/saved_models/u2net/u2net.pth'
            if isfile(weight_path):
                self._model.load_state_dict(torch.load(weight_path, map_location='cpu'))
                print(f"[U2NetSaliency] Model loaded from {weight_path}")
            else:
                print(f"[U2NetSaliency] Warning: Weight file not found at {weight_path}, using untrained model")
            self._model.eval()
        return self._model

    @torch.no_grad()
    def compute_saliency(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Compute saliency map using U2-Net.
        Args:
            image_bgr: uint8 BGR image, HxWx3
        Returns:
            saliency: float32 [0,1], HxW
        """
        model = self.get_model()
        h, w = image_bgr.shape[:2]
        
        # Prepare input tensor
        input_tensor = torch.from_numpy(image_bgr.transpose(2, 0, 1)).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0)
        
        # Forward pass
        d1, *_ = model(input_tensor)
        saliency = d1.squeeze().cpu().numpy()
        
        # Normalize to [0,1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Resize to original dimensions
        saliency = cv2.resize(saliency, (w, h))
        return saliency


# Global saliency model instance
_saliency_model = None


def get_saliency_model():
    """Get or create global saliency model instance."""
    global _saliency_model
    if _saliency_model is None:
        _saliency_model = U2NetSaliency()
    return _saliency_model


def rgba_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGBA image to BGR without mutating the input."""
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image[:, :, :3]


def compute_fast_saliency(image: np.ndarray) -> np.ndarray:
    """Fast heuristic saliency using gradients + center bias."""
    bgr = rgba_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Edge emphasis
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)

    # Color variance
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    mean_l = cv2.GaussianBlur(l_channel, (31, 31), 0)
    variance = cv2.GaussianBlur((l_channel - mean_l) ** 2, (31, 31), 0)

    # Center prior
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    center_bias = 1.0 - (dist / dist.max())

    saliency = (
        0.4 * (gradient / (gradient.max() + 1e-6)) +
        0.3 * (variance / (variance.max() + 1e-6)) +
        0.3 * center_bias
    )

    saliency = cv2.GaussianBlur(saliency, (25, 25), 0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    return saliency.astype(np.float32)


def compute_u2net_saliency(image):
    """Compute saliency using U2-Net model."""
    model = get_saliency_model()
    bgr = rgba_to_bgr(image)
    return model.compute_saliency(bgr)


def compute_u2net_saliency_downsampled(image: np.ndarray, max_size: int = 320) -> np.ndarray:
    """Run U2-Net on a downsampled image for speed, then upscale back.
    
    Args:
        image: Input image
        max_size: Maximum dimension for U2-Net input (smaller = faster, default 320)
    """
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = image

    saliency_small = compute_u2net_saliency(resized)
    saliency = cv2.resize(saliency_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return saliency


def compute_fast_saliency_for_center(image: np.ndarray) -> np.ndarray:
    """
    FAST saliency estimation using color + gradient + center bias.
    Much faster than U2-Net (~0.01s vs ~40s), good enough for finding salient center.
    """
    h, w = image.shape[:2]
    
    # Downscale for speed
    scale = min(1.0, 256 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image
    
    # Convert to LAB for better color saliency
    if small.shape[2] == 4:
        small_bgr = cv2.cvtColor(small, cv2.COLOR_RGBA2BGR)
    else:
        small_bgr = small[:, :, :3]
    
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Color uniqueness: distance from mean color
    mean_lab = lab.mean(axis=(0, 1))
    color_dist = np.sqrt(((lab - mean_lab) ** 2).sum(axis=2))
    color_dist = (color_dist - color_dist.min()) / (color_dist.max() - color_dist.min() + 1e-6)
    
    # Edge/gradient magnitude
    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx**2 + gy**2)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    
    # Strong center bias (most subjects are near center)
    sh, sw = small.shape[:2]
    yy, xx = np.ogrid[:sh, :sw]
    cy, cx = sh / 2.0, sw / 2.0
    sigma = min(sh, sw) * 0.4
    center_bias = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    
    # Combine: color uniqueness + edges + strong center bias
    saliency = 0.35 * color_dist + 0.25 * gradient + 0.40 * center_bias
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (15, 15), 0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    
    # Upscale back
    if scale < 1.0:
        saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return saliency.astype(np.float32)


def apply_center_bias(saliency: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Blend saliency with a Gaussian center bias to keep subject near center."""
    h, w = saliency.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    sigma_y = 0.35 * h
    sigma_x = 0.35 * w
    gaussian = np.exp(-(((yy - cy) ** 2) / (2 * sigma_y ** 2) + ((xx - cx) ** 2) / (2 * sigma_x ** 2)))
    gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min() + 1e-6)
    blended = (1 - strength) * saliency + strength * gaussian
    return (blended - blended.min()) / (blended.max() - blended.min() + 1e-6)


def expand_saliency_region(saliency: np.ndarray, threshold: float = 0.12, expansion_factor: float = 1.5) -> np.ndarray:
    """Dilate saliency mask to cover MUCH wider foreground with smooth edges.
    
    Args:
        saliency: Input saliency map [0, 1]
        threshold: Lower threshold = more area considered salient
        expansion_factor: How much to expand (1.0 = normal, 2.0 = very large)
    """
    # Lower threshold to include more area as salient
    mask = (saliency >= threshold).astype(np.uint8)
    
    # Much larger kernel for bigger expansion
    kernel_size = int(35 * expansion_factor)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Must be odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # More iterations = bigger expansion
    iterations = int(3 * expansion_factor)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    
    # Larger blur for smoother falloff
    blur_size = int(51 * expansion_factor)
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    expanded = cv2.GaussianBlur(dilated.astype(np.float32), (blur_size, blur_size), 0)
    expanded = np.clip(expanded, 0.0, 1.0)
    
    # Blend expanded mask with original, but give more weight to expanded
    combined = np.maximum(saliency, expanded * 0.8 + saliency * 0.2)
    return (combined - combined.min()) / (combined.max() - combined.min() + 1e-6)


def compute_saliency_hybrid(image: np.ndarray,
                            prefer_u2net: bool = True,
                            fast_only: bool = False,
                            center_bias_strength: float = 0.45,
                            threshold: float = 0.10,
                            expansion_factor: float = 1.8) -> np.ndarray:
    """Hybrid saliency: U2-Net (downsampled) with LARGER expanded regions + center bias.
    
    Args:
        image: Input image
        prefer_u2net: Use U2-Net if available
        fast_only: Use fast heuristic only
        center_bias_strength: Higher = more center bias (keeps content near center)
        threshold: Lower = more area considered salient
        expansion_factor: Higher = larger salient region (less distortion)
    """
    saliency = None
    if prefer_u2net and not fast_only:
        try:
            saliency = compute_u2net_saliency_downsampled(image)
        except Exception as e:
            print(f"[SaliencyHybrid] U2-Net failed ({e}), using fast heuristic")
    if saliency is None:
        saliency = compute_fast_saliency(image)

    # Apply stronger center bias to keep subjects centered
    saliency = apply_center_bias(saliency, strength=center_bias_strength)
    
    # Expand salient region MORE to reduce distortion
    saliency = expand_saliency_region(saliency, threshold=threshold, expansion_factor=expansion_factor)
    return saliency.astype(np.float32)


def get_salient_bbox_from_saliency(saliency_map, threshold=0.25):
    """
    Find bounding box of salient region from U2-Net saliency map.
    
    Args:
        saliency_map: Float32 saliency map [0,1] from U2-Net
        threshold: Threshold to binarize saliency
    
    Returns:
        [x1, x2, y1, y2] bounding box of salient region, or None if not found
    """
    # Binarize saliency
    mask = (saliency_map > threshold).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Add small padding (5% on each side)
    h_img, w_img = saliency_map.shape[:2]
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    
    x1 = max(0, x - pad_x)
    x2 = min(w_img, x + w + pad_x)
    y1 = max(0, y - pad_y)
    y2 = min(h_img, y + h + pad_y)
    
    return [x1, x2, y1, y2]


def get_salient_center_from_saliency(saliency_map, threshold=0.25):
    """
    Find the weighted center of salient region from U2-Net saliency map.
    
    Args:
        saliency_map: Float32 saliency map [0,1] from U2-Net
        threshold: Threshold for considering salient pixels
    
    Returns:
        (cx, cy) center coordinates, or None if not found
    """
    h, w = saliency_map.shape[:2]
    
    # Create coordinate grids
    yy, xx = np.mgrid[:h, :w]
    
    # Weight by saliency values above threshold
    weights = np.where(saliency_map > threshold, saliency_map, 0)
    total_weight = weights.sum()
    
    if total_weight < 1e-6:
        return None
    
    # Weighted centroid
    cx = (xx * weights).sum() / total_weight
    cy = (yy * weights).sum() / total_weight
    
    return (cx, cy)


def smart_crop_to_center_salient(image, salient_center, target_w, target_h, shape_centroid_ratio=(0.5, 0.5)):
    """
    Smart crop image so that salient center aligns with target shape's centroid.
    Crops more from the opposite side of where the salient region is.
    
    Args:
        image: Source image (H, W, C)
        salient_center: (cx, cy) center of salient region in source image
        target_w, target_h: Target dimensions
        shape_centroid_ratio: (rx, ry) where the centroid is in target shape (0.5, 0.5 = center)
    
    Returns:
        cropped_image: Image cropped and resized to target dimensions
        crop_box: (x1, y1, x2, y2) the crop region used
    """
    h_src, w_src = image.shape[:2]
    src_cx, src_cy = salient_center
    
    # Target aspect ratio
    target_aspect = target_w / target_h
    src_aspect = w_src / h_src
    
    # Determine crop dimensions to match target aspect ratio
    if src_aspect > target_aspect:
        # Source is wider - crop width
        crop_h = h_src
        crop_w = int(h_src * target_aspect)
    else:
        # Source is taller - crop height
        crop_w = w_src
        crop_h = int(w_src / target_aspect)
    
    # Calculate crop position to align salient center with shape centroid
    target_rx, target_ry = shape_centroid_ratio
    
    # In crop coordinates, salient should be at target_rx, target_ry
    ideal_x1 = src_cx - target_rx * crop_w
    ideal_y1 = src_cy - target_ry * crop_h
    
    # Clamp to valid range
    x1 = int(max(0, min(w_src - crop_w, ideal_x1)))
    y1 = int(max(0, min(h_src - crop_h, ideal_y1)))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    # Crop
    cropped = image[y1:y2, x1:x2]
    
    # Resize to target
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return resized, (x1, y1, x2, y2)
