"""
Image utility functions for collage assembly.
Basic image loading, writing, preprocessing, and overlay operations.
"""

import cv2
import numpy as np
from os.path import isfile


def load_color_image(path):
    """Load image with fallback to different extensions if file not found."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    # Fallback: try different extensions
    if image is None:
        base_path = path.rsplit('.', 1)[0]  # Remove extension
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for ext in possible_extensions:
            fallback_path = base_path + ext
            if isfile(fallback_path):
                print(f"[FALLBACK] Original path not found, trying: {fallback_path}")
                image = cv2.imread(fallback_path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    break
    
    # If still None, raise error with helpful message
    if image is None:
        raise FileNotFoundError(f"Cannot find image file: {path} (or any common extension)")
    
    # Convert to RGBA if needed
    if image.ndim == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.shape[2] == 3:  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif image.shape[2] == 4:  # BGRA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    
    return image


def write_color_image(array, path):
    """Write RGBA array to file as BGRA."""
    bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, bgr)


def preprocess_image(img):
    """Scale image if too large (The preprocessing step)."""
    max_side = max(img.shape[0], img.shape[1])
    if max_side > 1500:
        scale_factor = 1500 / max_side
        img = cv2.resize(img, (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor)))
    return img


def retarget(image, width, height):
    """Simple resize image to target dimensions."""
    return cv2.resize(image, (width, height))


def image_overlay(target, source, origin):
    """
    Overlay an image over the target image at origin (in target image coordinate).
    
    Args:
        target: Target image to overlay onto
        source: Source image to overlay
        origin: (starting row, starting column)
    
    Returns:
        Target image with source overlaid
    """
    target = target.copy()
    source_crop = source.copy()
    
    # Row handling (Case 1-4)
    if origin[0] < 0 and origin[0] + source.shape[0] - 1 <= target.shape[0]:
        start_row = 0
        end_row = origin[0] + source.shape[0]
        source_crop = source_crop[-origin[0]:, :].copy()
    elif origin[0] >= 0 and origin[0] + source.shape[0] - 1 <= target.shape[0]:
        start_row = origin[0]
        end_row = origin[0] + source.shape[0]
        source_crop = source_crop.copy()
    elif origin[0] >= 0 and origin[0] + source.shape[0] - 1 > target.shape[0]:
        start_row = origin[0]
        end_row = target.shape[0] - 1
        source_crop = source_crop[0:target.shape[0] - origin[0] - 1, :].copy()
    else:
        start_row = 0
        end_row = target.shape[0] - 1
        source_crop = source_crop[-origin[0]:target.shape[0] - origin[0] - 1, :].copy()
    
    # Column handling (Case 1-4)
    if origin[1] < 0 and origin[1] + source.shape[1] - 1 <= target.shape[1]:
        start_col = 0
        end_col = origin[1] + source.shape[1]
        source_crop = source_crop[:, -origin[1]:].copy()
    elif origin[1] >= 0 and origin[1] + source.shape[1] - 1 <= target.shape[1]:
        start_col = origin[1]
        end_col = origin[1] + source.shape[1]
        source_crop = source_crop.copy()
    elif origin[1] >= 0 and origin[1] + source.shape[1] - 1 > target.shape[1]:
        start_col = origin[1]
        end_col = target.shape[1] - 1
        source_crop = source_crop[:, 0:target.shape[1] - origin[1] - 1].copy()
    else:
        start_col = 0
        end_col = target.shape[1] - 1
        source_crop = source_crop[:, -origin[1]:target.shape[1] - origin[1] - 1].copy()
    
    target[start_row:end_row, start_col:end_col] = source_crop
    return target


def overlay_mask(mask1, mask2):
    """
    Overlay masks avoiding overlaps.
    
    Args:
        mask1: First mask (uint8)
        mask2: Second mask (uint8)
    
    Returns:
        mask2 with overlapping regions removed
    """
    overlaps = cv2.bitwise_and(mask1, mask2)
    return mask2 - overlaps


def rgba_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGBA image to BGR without mutating the input."""
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image[:, :, :3]


def adjust_inner_rec(outer, inner):
    """
    Enlarge the main object rectangle to add some margin.
    
    Args:
        outer: Outer rectangle (x1, x2, y1, y2)
        inner: Inner rectangle (x1, x2, y1, y2)
    
    Returns:
        Tuple of (adjusted_rectangle, touch_boundary_flag)
    """
    outer_width = outer[1] - outer[0]
    outer_height = outer[3] - outer[2]
    
    inner_width = inner[1] - inner[0]
    inner_height = inner[3] - inner[2]
    margin_width = int(inner_width / 18)
    margin_height = int(inner_height / 18)
    
    new_x1 = max(inner[0] - margin_width, int(outer_width / 120))
    new_x2 = min(inner[1] + margin_width, outer[1] - int(outer_width / 120))
    new_y1 = max(inner[2] - margin_height, int(outer_height / 120))
    new_y2 = min(inner[3] + margin_height, outer[3] - int(outer_height / 120))
    
    touch_boundary = False
    if new_x1 == 0 or new_x2 == outer[1] or new_y1 == 0 or new_y2 == outer[3]:
        touch_boundary = True
    
    return (new_x1, new_x2, new_y1, new_y2), touch_boundary
