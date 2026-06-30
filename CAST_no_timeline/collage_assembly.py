from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, dirname, abspath
import sys
import json
import seam_carving
import glob
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import minimize
import torch
import torch.nn as nn
import math

# Import U2-Net from utils

from utils.u2net import U2NET

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
    
    # Normalize to BGRA (OpenCV-native channel order)
    if image.ndim == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    return image

'''
Scale image if too large (The preprocessing step)
'''
def preprocess_image(img):
    max_side = max(img.shape[0], img.shape[1])
    if max_side > 1500:
        scale_factor = 1500 / max_side
        img = cv2.resize(img, (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor)))
    return img


def write_color_image(array, path):
    cv2.imwrite(path, array)
    
def rowcol2xy(row, col, ymax):
    return int(col), int(row)

def xy2rowcol(x, y , ymax):
    return int(round(y, 0)), int(round(x, 0))

def retarget(image, width, height):
    return cv2.resize(image, (width, height))
'''
move origin to minX, minY of the polygon bounding box
'''
def polygon2local_coordinate(polygon):
    bounding_box = polygon.bounds
    return np.array([(int(coord[0] - bounding_box[0]), int(coord[1] - bounding_box[1]))
                     for coord in list(polygon.exterior.coords)])

'''
overaly an image over the target image at origin (in target image coordinate)
origin: (starting row, starting column)
'''
def image_overlay(target, source, origin):
    target = target.copy()
    src_h, src_w = source.shape[:2]
    dst_h, dst_w = target.shape[:2]
    origin_row, origin_col = origin

    dst_r0 = max(origin_row, 0)
    dst_c0 = max(origin_col, 0)
    dst_r1 = min(origin_row + src_h, dst_h)
    dst_c1 = min(origin_col + src_w, dst_w)

    if dst_r0 >= dst_r1 or dst_c0 >= dst_c1:
        return target

    src_r0 = dst_r0 - origin_row
    src_c0 = dst_c0 - origin_col
    src_r1 = src_r0 + (dst_r1 - dst_r0)
    src_c1 = src_c0 + (dst_c1 - dst_c0)

    target[dst_r0:dst_r1, dst_c0:dst_c1] = source[src_r0:src_r1, src_c0:src_c1]
    return target

'''
enlarge the main object rectangle to add some margin
if touch to boundary, return True
input format: (x1, y1, x2, y2)
'''
def adjust_inner_rec(outer, inner):
    outer_width = outer[2] - outer[0]
    outer_height = outer[3] - outer[1]
    
    inner_width = inner[2] - inner[0]
    inner_height = inner[3] - inner[1]
    margin_width = int(inner_width/18)
    margin_height = int(inner_height/18)
    
    new_x1 = max(inner[0]-margin_width, int(outer_width/120))
    new_x2 = min(inner[2]+margin_width, outer[2]-int(outer_width/120))
    new_y1 = max(inner[1]-margin_height, int(outer_height/120))
    new_y2 = min(inner[3]+margin_height, outer[3]-int(outer_height/120))
    
#     new_x1 = max(inner[0]-margin_width, 0)
#     new_x2 = min(inner[1]+margin_width, outer[1])
#     new_y1 = max(inner[2]-margin_height, 0)
#     new_y2 = min(inner[3]+margin_height, outer[3])
    
    touch_boundary = False
    if new_x1==0 or new_x2==outer[2] or new_y1 == 0 or new_y2==outer[3]:
        touch_boundary = True
    
    return (new_x1, new_y1, new_x2, new_y2), touch_boundary


def normalize_foreground_bbox(fg_bbox, img_w, img_h):
    """Normalize bbox to [x1, y1, x2, y2] and clamp to image bounds."""
    if fg_bbox is None or len(fg_bbox) != 4:
        return [0, 0, img_w, img_h]

    a, b, c, d = [float(v) for v in fg_bbox]

    # Candidate 1: already [x1, y1, x2, y2]
    cand1 = [a, b, c, d]
    # Candidate 2: legacy [x1, x2, y1, y2] -> [x1, y1, x2, y2]
    cand2 = [a, c, b, d]

    def _score(cand):
        x1, y1, x2, y2 = cand
        if x2 <= x1 or y2 <= y1:
            return -1e9
        in_bounds = (
            (0 <= x1 <= img_w) + (0 <= x2 <= img_w) +
            (0 <= y1 <= img_h) + (0 <= y2 <= img_h)
        )
        area = (x2 - x1) * (y2 - y1)
        return in_bounds * 1000.0 + area

    x1, y1, x2, y2 = cand1 if _score(cand1) >= _score(cand2) else cand2

    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), img_w))
    y2 = max(y1 + 1, min(int(round(y2)), img_h))
    return [x1, y1, x2, y2]

'''
Get the triangulation given outer and innter rectangles (counter-clockwise order start from (0,0))
    [
        bottem left, bottom right, top right, top left
    ]
'''
def triangulation(outer_rec, inner_rec, height):
    triangles = [[outer_rec[3], inner_rec[3], outer_rec[2]],
     [inner_rec[3], inner_rec[2], outer_rec[2]],
     [inner_rec[2], outer_rec[1], outer_rec[2]],
     [inner_rec[2], inner_rec[1], outer_rec[1]],
     [inner_rec[0], outer_rec[1], inner_rec[1]],
     [outer_rec[0], outer_rec[1], inner_rec[0]],
     [outer_rec[0], inner_rec[0], inner_rec[3]],
     [outer_rec[3], outer_rec[0], inner_rec[3]],
     [inner_rec[3], inner_rec[2], inner_rec[1]],
     [inner_rec[0], inner_rec[1], inner_rec[3]]
    ]
    return [[(vertex[0], height-vertex[1]) for vertex in t] for t in triangles]

'''
masks are uint8 array of shape (height, width)
'''
def overlay_mask(mask1, mask2):
    overlaps = cv2.bitwise_and(mask1, mask2)
    return mask2 - overlaps # remove overlaps


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
_saliency_model = U2NetSaliency()


def _rgba_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert BGRA image to BGR without mutating the input."""
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3]


def compute_fast_saliency(image: np.ndarray) -> np.ndarray:
    """Fast heuristic saliency using gradients + center bias."""
    bgr = _rgba_to_bgr(image)
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


def compute_u2net_saliency_downsampled(image: np.ndarray, max_size: int = 640) -> np.ndarray:
    """Run U2-Net on a downsampled image for speed, then upscale back."""
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


def expand_saliency_region(saliency: np.ndarray, threshold: float = 0.18) -> np.ndarray:
    """Dilate saliency mask to cover wider foreground with smooth edges."""
    mask = (saliency >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    expanded = cv2.GaussianBlur(dilated.astype(np.float32), (31, 31), 0)
    expanded = np.clip(expanded, 0.0, 1.0)
    # Blend expanded mask with original for softer falloff
    combined = np.maximum(saliency, expanded)
    return (combined - combined.min()) / (combined.max() - combined.min() + 1e-6)


def compute_saliency_hybrid(image: np.ndarray,
                            prefer_u2net: bool = True,
                            fast_only: bool = False,
                            center_bias_strength: float = 0.35,
                            threshold: float = 0.18) -> np.ndarray:
    """Hybrid saliency: U2-Net (downsampled) with expanded regions + center bias."""
    saliency = None
    if prefer_u2net and not fast_only:
        try:
            saliency = compute_u2net_saliency_downsampled(image)
        except Exception as e:
            print(f"[SaliencyHybrid] U2-Net failed ({e}), using fast heuristic")
    if saliency is None:
        saliency = compute_fast_saliency(image)

    saliency = apply_center_bias(saliency, strength=center_bias_strength)
    saliency = expand_saliency_region(saliency, threshold=threshold)
    return saliency.astype(np.float32)

def create_dense_mesh(h, w, grid_size=16):
    """V2: Clean 2D grid — no duplicate boundary points.
    Always includes edges (y=0, y=h-1, x=0, x=w-1).
    Returns (src_pts, grid_rows, grid_cols) for 2D-aware smoothness.
    """
    ys = list(range(0, h, grid_size))
    if ys[-1] != h - 1:
        ys.append(h - 1)
    xs = list(range(0, w, grid_size))
    if xs[-1] != w - 1:
        xs.append(w - 1)

    grid_rows = len(ys)
    grid_cols = len(xs)

    pts = []
    for y in ys:
        for x in xs:
            pts.append([x, y])

    return np.array(pts, dtype=np.float32), grid_rows, grid_cols


def _draw_control_grid(image, pts, grid_rows, grid_cols, color=(0, 255, 0),
                       point_radius=2, line_thickness=1):
    if image.shape[2] == 4:
        viz_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        viz_img = image[:, :, :3].copy()

    if grid_rows <= 0 or grid_cols <= 0:
        return viz_img

    n_pts = len(pts)
    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = r * grid_cols + c
            if idx >= n_pts:
                continue
            x, y = int(pts[idx][0]), int(pts[idx][1])
            if c + 1 < grid_cols:
                idx_r = idx + 1
                if idx_r < n_pts:
                    xr, yr = int(pts[idx_r][0]), int(pts[idx_r][1])
                    cv2.line(viz_img, (x, y), (xr, yr), color, line_thickness)
            if r + 1 < grid_rows:
                idx_d = idx + grid_cols
                if idx_d < n_pts:
                    xd, yd = int(pts[idx_d][0]), int(pts[idx_d][1])
                    cv2.line(viz_img, (x, y), (xd, yd), color, line_thickness)

    for i in range(min(n_pts, grid_rows * grid_cols)):
        x, y = int(pts[i][0]), int(pts[i][1])
        cv2.circle(viz_img, (x, y), point_radius, color, -1)

    return viz_img


def _save_warp_grid_images(original_image, warped_image, src_pts, dst_pts, grid_rows,
                           grid_cols, grid_debug_dir, image_id):
    if not grid_debug_dir:
        return

    os.makedirs(grid_debug_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(str(image_id)))[0]

    pre_img = _draw_control_grid(original_image, src_pts, grid_rows, grid_cols)
    post_img = _draw_control_grid(warped_image, dst_pts, grid_rows, grid_cols)

    pre_path = join(grid_debug_dir, f"grid_pre_{base_name}.png")
    post_path = join(grid_debug_dir, f"grid_post_{base_name}.png")
    cv2.imwrite(pre_path, pre_img)
    cv2.imwrite(post_path, post_img)


def compute_u2net_saliency(image):
    """Compute saliency map using U2-Net."""
    try:
        # Convert BGRA to BGR for U2-Net
        if image.shape[2] == 4:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            image_bgr = image[:, :, :3]
        
        # Use global U2-Net model
        saliency = _saliency_model.compute_saliency(image_bgr)
        return saliency
    
    except Exception as e:
        print(f"[U2NetSaliency] Failed to compute saliency ({e}), using fallback")
        # Fallback: use center-weighted saliency
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        saliency = 1.0 - (dist / max_dist)
        return saliency.astype(np.float32)




def optimize_mesh_warp(src_pts,
                       dst_pts_init,
                       saliency_map,
                       inner_src_box,
                       inner_dest_box,
                       image_shape,
                       target_shape,
                       grid_rows=0,
                       grid_cols=0,
                       salient_weight: float = 50.0,
                       background_weight: float = 5.0,
                       smooth_lambda: float = 15.0):
    """
    V2: Optimize mesh control points with 2D grid-aware smoothness.
    
    Fixes over V1:
    - 2D spatial smoothness: penalizes deviation from initial grid spacing
      between actual grid neighbors (right, down), not sequential array indices.
    - Balanced weights: salient:bg = 10:1 (was 875:1) to prevent fold-over.
    - Minimum target size guard: skip optimization for tiny cells.
    - Clamp output to valid bounds.
    """
    h_src, w_src = image_shape
    h_dst, w_dst = target_shape
    N = len(src_pts)
    
    # Guard: skip optimization for tiny targets
    if h_dst < 10 or w_dst < 10 or N < 4:
        return dst_pts_init.copy()
    
    # Saliency weights per mesh point
    xs = np.clip(src_pts[:, 0].astype(int), 0, w_src - 1)
    ys = np.clip(src_pts[:, 1].astype(int), 0, h_src - 1)
    weights = np.maximum(0.1, saliency_map[ys, xs])
    
    # Salient region mask
    in_salient = (
        (src_pts[:, 0] >= inner_src_box[0]) & (src_pts[:, 0] <= inner_src_box[2]) &
        (src_pts[:, 1] >= inner_src_box[1]) & (src_pts[:, 1] <= inner_src_box[3])
    )
    
    # Target positions by proportional mapping src→dst inner box
    src_box_w = max(inner_src_box[2] - inner_src_box[0], 1e-6)
    src_box_h = max(inner_src_box[3] - inner_src_box[1], 1e-6)
    dst_box_w = inner_dest_box[2] - inner_dest_box[0]
    dst_box_h = inner_dest_box[3] - inner_dest_box[1]
    
    ratio_x = (src_pts[:, 0] - inner_src_box[0]) / src_box_w
    ratio_y = (src_pts[:, 1] - inner_src_box[1]) / src_box_h
    target_pts = np.stack([
        inner_dest_box[0] + ratio_x * dst_box_w,
        inner_dest_box[1] + ratio_y * dst_box_h
    ], axis=1)
    
    # Weight multipliers
    sal_w = weights * salient_weight
    bg_w = np.maximum(background_weight, weights * background_weight)
    
    # Build 2D neighbor pairs (right + down) for grid-aware smoothness
    pairs_i = []
    pairs_j = []
    if grid_rows > 0 and grid_cols > 0:
        for r in range(grid_rows):
            for c in range(grid_cols):
                idx = r * grid_cols + c
                if idx >= N:
                    continue
                if c + 1 < grid_cols:
                    idx_r = r * grid_cols + (c + 1)
                    if idx_r < N:
                        pairs_i.append(idx)
                        pairs_j.append(idx_r)
                if r + 1 < grid_rows:
                    idx_d = (r + 1) * grid_cols + c
                    if idx_d < N:
                        pairs_i.append(idx)
                        pairs_j.append(idx_d)
    else:
        # Fallback: sequential pairs (V1 style)
        for i in range(N - 1):
            pairs_i.append(i)
            pairs_j.append(i + 1)
    
    pi = np.array(pairs_i, dtype=int)
    pj = np.array(pairs_j, dtype=int)
    n_pairs = len(pi)
    
    # Reference: initial grid spacing between neighbors
    dst_init = dst_pts_init.copy()
    init_spacing = dst_init[pj] - dst_init[pi] if n_pairs > 0 else np.empty((0, 2))
    
    def loss_and_grad(x):
        dst = x.reshape(N, 2)
        g = np.zeros_like(dst)
        total = 0.0
        
        # Salient loss: pull salient points toward target
        diff_sal = dst - target_pts
        dist_sal = np.sqrt(np.sum(diff_sal ** 2, axis=1) + 1e-12)
        total += np.sum(sal_w[in_salient] * dist_sal[in_salient])
        g_sal = sal_w[:, None] * diff_sal / (dist_sal[:, None] + 1e-8)
        g[in_salient] += g_sal[in_salient]
        
        # Background loss: keep background near init
        diff_bg = dst - dst_init
        dist_bg = np.sqrt(np.sum(diff_bg ** 2, axis=1) + 1e-12)
        bg_mask = ~in_salient
        total += np.sum(bg_w[bg_mask] * dist_bg[bg_mask])
        g_bg = bg_w[:, None] * diff_bg / (dist_bg[:, None] + 1e-8)
        g[bg_mask] += g_bg[bg_mask]
        
        # Boundary penalty
        oob = ((dst[:, 0] < 0) | (dst[:, 0] >= w_dst) |
               (dst[:, 1] < 0) | (dst[:, 1] >= h_dst))
        total += np.sum(oob) * 1e6
        g[dst[:, 0] < 0, 0] -= 1e6
        g[dst[:, 0] >= w_dst, 0] += 1e6
        g[dst[:, 1] < 0, 1] -= 1e6
        g[dst[:, 1] >= h_dst, 1] += 1e6
        
        # 2D Grid smoothness: penalize change in neighbor spacing vs initial
        if n_pairs > 0:
            curr_spacing = dst[pj] - dst[pi]
            diff_s = curr_spacing - init_spacing
            total += smooth_lambda * np.sum(diff_s ** 2)
            g_s = 2.0 * smooth_lambda * diff_s
            np.add.at(g, pi, -g_s)
            np.add.at(g, pj, g_s)
        
        return total, g.flatten()
    
    # Cache to avoid double computation in L-BFGS-B (calls loss then grad separately)
    _cache = {}
    def _cached(x):
        key = x.tobytes()
        if key not in _cache:
            _cache.clear()
            _cache[key] = loss_and_grad(x)
        return _cache[key]
    
    result = minimize(
        lambda x: _cached(x)[0],
        dst_pts_init.flatten(),
        jac=lambda x: _cached(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 50, 'ftol': 1e-4}
    )
    
    optimized_pts = result.x.reshape(-1, 2)
    
    # Clamp to valid bounds
    optimized_pts[:, 0] = np.clip(optimized_pts[:, 0], 0, w_dst - 1)
    optimized_pts[:, 1] = np.clip(optimized_pts[:, 1], 0, h_dst - 1)
    
    return optimized_pts


def apply_mesh_warp_remap(image, src_pts, dst_pts, target_shape):
    """V2: griddata with nearest-neighbor fallback for NaN + INTER_CUBIC.
    
    Fixes over V1:
    - No fill_value=0 — uses nearest-neighbor for pixels outside convex hull.
    - INTER_CUBIC for sharper output (was INTER_LINEAR).
    - BORDER_REFLECT avoids black edges.
    - Guard for tiny targets.
    """
    from scipy.interpolate import griddata

    h_dst, w_dst = target_shape

    # Guard: tiny target → just resize
    if h_dst < 2 or w_dst < 2:
        return cv2.resize(image, (max(1, w_dst), max(1, h_dst)))

    grid_x, grid_y = np.meshgrid(np.arange(w_dst), np.arange(h_dst))
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Linear interpolation (no fill_value — NaN for outside convex hull)
    map_x_lin = griddata(dst_pts, src_pts[:, 0], grid_points, method='linear')
    map_y_lin = griddata(dst_pts, src_pts[:, 1], grid_points, method='linear')

    # Fill NaN with nearest-neighbor (instead of fill_value=0 which mapped to pixel 0,0)
    nan_mask = np.isnan(map_x_lin) | np.isnan(map_y_lin)
    if np.any(nan_mask):
        map_x_nn = griddata(dst_pts, src_pts[:, 0], grid_points, method='nearest')
        map_y_nn = griddata(dst_pts, src_pts[:, 1], grid_points, method='nearest')
        map_x_lin[nan_mask] = map_x_nn[nan_mask]
        map_y_lin[nan_mask] = map_y_nn[nan_mask]

    map_x = map_x_lin.reshape(h_dst, w_dst).astype(np.float32)
    map_y = map_y_lin.reshape(h_dst, w_dst).astype(np.float32)

    map_x = np.clip(map_x, 0, image.shape[1] - 1)
    map_y = np.clip(map_y, 0, image.shape[0] - 1)

    warped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REFLECT)
    return warped


def apply_mesh_warp_tps(image, src_pts, dst_pts, target_shape):
    """Smooth thin-plate-spline-style warp with cubic interpolation."""
    try:
        from scipy.interpolate import RBFInterpolator

        h_dst, w_dst = target_shape
        grid_y, grid_x = np.meshgrid(np.arange(h_dst), np.arange(w_dst), indexing='ij')
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        rbf_x = RBFInterpolator(dst_pts, src_pts[:, 0], kernel='thin_plate_spline', smoothing=0.5)
        rbf_y = RBFInterpolator(dst_pts, src_pts[:, 1], kernel='thin_plate_spline', smoothing=0.5)

        map_x = rbf_x(grid_points).reshape(h_dst, w_dst).astype(np.float32)
        map_y = rbf_y(grid_points).reshape(h_dst, w_dst).astype(np.float32)

        map_x = np.clip(map_x, 0, image.shape[1] - 1)
        map_y = np.clip(map_y, 0, image.shape[0] - 1)

        warped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        return warped
    except Exception as e:
        print(f"[TPS] Falling back to linear remap ({e})")
        return apply_mesh_warp_remap(image, src_pts, dst_pts, target_shape)


def load_mask_as_saliency(mask_path, target_h, target_w):
    """Load a pre-computed binary mask and convert to a saliency map [0,1]."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    # Resize to match source image
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    saliency = mask.astype(np.float32) / 255.0
    # Smooth edges for better warp
    saliency = cv2.GaussianBlur(saliency, (15, 15), 0)
    saliency = np.clip(saliency, 0.0, 1.0)
    return saliency


def batch_compute_saliency_gpu(images, max_size=640):
    """
    Batch saliency computation using U2-Net on GPU.
    Args:
        images: list of RGBA/BGR numpy arrays
        max_size: Max dimension for processing
    Returns:
        list of saliency maps (float32, [0,1], same H,W as input)
    """
    if not images:
        return []

    model = _saliency_model.get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare all tensors
    orig_sizes = []
    tensors = []
    for img in images:
        bgr = _rgba_to_bgr(img) if img.shape[2] == 4 else img[:, :, :3]
        h, w = bgr.shape[:2]
        orig_sizes.append((h, w))
        # Resize for speed
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(bgr.transpose(2, 0, 1)).float() / 255.0
        tensors.append(t)

    # Pad to same size for batching
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded = []
    for t in tensors:
        ph = max_h - t.shape[1]
        pw = max_w - t.shape[2]
        padded.append(torch.nn.functional.pad(t, (0, pw, 0, ph), value=0))

    # Batch inference
    batch_size = 8  # Process in mini-batches to avoid OOM
    saliency_maps = []
    with torch.no_grad():
        for i in range(0, len(padded), batch_size):
            batch = torch.stack(padded[i:i+batch_size]).to(device)
            d1, *_ = model(batch)
            for j in range(d1.shape[0]):
                sal = d1[j].squeeze().cpu().numpy()
                sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
                # Crop back to unpadded size
                idx = i + j
                th, tw = tensors[idx].shape[1], tensors[idx].shape[2]
                sal = sal[:th, :tw]
                # Resize back to original
                oh, ow = orig_sizes[idx]
                sal = cv2.resize(sal, (ow, oh), interpolation=cv2.INTER_CUBIC)
                saliency_maps.append(sal.astype(np.float32))

    model.cpu()  # Free GPU memory
    return saliency_maps


def _precrop_to_aspect(image, saliency_map, fg_box, target_w, target_h):
    """
    Pre-crop source image to approximately match target aspect ratio,
    keeping the salient region (fg_box) centered.
    
    Returns: (cropped_image, cropped_saliency, new_fg_box)
    """
    h_src, w_src = image.shape[:2]
    target_ar = target_w / (target_h + 1e-6)
    src_ar = w_src / (h_src + 1e-6)
    
    # If aspect ratios are similar enough, skip crop
    ar_ratio = max(target_ar, src_ar) / (min(target_ar, src_ar) + 1e-6)
    if ar_ratio < 1.15:
        return image, saliency_map, fg_box
    
    # fg_box = [x1, y1, x2, y2]
    fg_cx = (fg_box[0] + fg_box[2]) / 2
    fg_cy = (fg_box[1] + fg_box[3]) / 2
    fg_w = fg_box[2] - fg_box[0]
    fg_h = fg_box[3] - fg_box[1]
    
    if target_ar > src_ar:
        # Target is wider → crop height (keep full width)
        new_h = int(w_src / target_ar)
        new_h = max(new_h, int(fg_h * 1.1))  # Must contain fg
        new_h = min(new_h, h_src)
        new_w = w_src
    else:
        # Target is taller → crop width (keep full height)
        new_w = int(h_src * target_ar)
        new_w = max(new_w, int(fg_w * 1.1))  # Must contain fg
        new_w = min(new_w, w_src)
        new_h = h_src
    
    # Center crop on fg center
    x1 = int(max(0, min(fg_cx - new_w / 2, w_src - new_w)))
    y1 = int(max(0, min(fg_cy - new_h / 2, h_src - new_h)))
    x2 = x1 + new_w
    y2 = y1 + new_h
    
    # Ensure fg is inside the crop
    if fg_box[0] < x1: x1 = max(0, int(fg_box[0] - 5)); x2 = x1 + new_w
    if fg_box[2] > x2: x2 = min(w_src, int(fg_box[2] + 5)); x1 = x2 - new_w
    if fg_box[1] < y1: y1 = max(0, int(fg_box[1] - 5)); y2 = y1 + new_h
    if fg_box[3] > y2: y2 = min(h_src, int(fg_box[3] + 5)); y1 = y2 - new_h
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_src, x2)
    y2 = min(h_src, y2)
    
    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_sal = saliency_map[y1:y2, x1:x2].copy()
    new_fg = [fg_box[0] - x1, fg_box[1] - y1, fg_box[2] - x1, fg_box[3] - y1]
    
    return cropped_img, cropped_sal, new_fg


def retarget_warp(image, 
             outer_rectangle_source,
             inner_rectangle_source,
             outer_rectangle_dest,
             inner_rectangle_dest,
             debug_dir=None,
             image_id="",
             saliency_map=None
            ):
    """
    Content-aware image warping.
    Uses pre-computed saliency map if provided, otherwise computes with U2-Net.
    """
    # Extract dimensions
    dest_xs = [pt[0] for pt in outer_rectangle_dest]
    dest_ys = [pt[1] for pt in outer_rectangle_dest]
    target_w = int(max(dest_xs) - min(dest_xs))
    target_h = int(max(dest_ys) - min(dest_ys))
    
    h_src, w_src = image.shape[:2]
    
    print(f"[Warp] Source: {w_src}x{h_src}, Target: {target_w}x{target_h}", end="")
    
    # 1. Get saliency map (pre-computed or compute now)
    if saliency_map is not None:
        if saliency_map.shape != (h_src, w_src):
            saliency_map = cv2.resize(saliency_map, (w_src, h_src))
        print(" [mask]", end="")
    else:
        saliency_map = compute_saliency_hybrid(
            image, prefer_u2net=True, fast_only=False,
            center_bias_strength=0.4, threshold=0.15
        )
        print(" [u2net]", end="")
    
    # === PRE-CROP: Match target aspect ratio BEFORE warping ===
    # TPS warp cannot handle extreme aspect ratio changes without heavy distortion.
    # Pre-crop the source image centered on salient region so warp only does mild adjustments.
    target_ar = target_w / max(target_h, 1)
    src_ar = w_src / max(h_src, 1)
    ar_ratio = target_ar / max(src_ar, 1e-6)
    
    if ar_ratio < 0.85 or ar_ratio > 1.18:  # >~15% AR mismatch — pre-crop before TPS
        # Get salient center from inner rectangle
        inner_cx = (inner_rectangle_source[0][0] + inner_rectangle_source[2][0]) / 2
        inner_cy = (inner_rectangle_source[0][1] + inner_rectangle_source[2][1]) / 2
        
        if target_ar > src_ar:
            # Target is wider → crop height
            new_h = int(w_src / target_ar)
            new_w = w_src
        else:
            # Target is taller → crop width
            new_w = int(h_src * target_ar)
            new_h = h_src
        
        # Ensure crop is at least as large as inner rectangle
        inner_w = abs(inner_rectangle_source[2][0] - inner_rectangle_source[0][0])
        inner_h = abs(inner_rectangle_source[2][1] - inner_rectangle_source[0][1])
        new_w = max(new_w, int(inner_w * 1.2))
        new_h = max(new_h, int(inner_h * 1.2))
        new_w = min(new_w, w_src)
        new_h = min(new_h, h_src)
        
        # Center crop on salient region
        cx1 = int(inner_cx - new_w / 2)
        cy1 = int(inner_cy - new_h / 2)
        
        # Clamp to image bounds
        cx1 = max(0, min(cx1, w_src - new_w))
        cy1 = max(0, min(cy1, h_src - new_h))
        cx2 = cx1 + new_w
        cy2 = cy1 + new_h
        
        # Apply pre-crop
        image = image[cy1:cy2, cx1:cx2].copy()
        saliency_map = saliency_map[cy1:cy2, cx1:cx2].copy()
        
        # Adjust inner rectangles for the crop offset
        inner_rectangle_source = [
            (inner_rectangle_source[0][0] - cx1, inner_rectangle_source[0][1] - cy1),
            (inner_rectangle_source[1][0] - cx1, inner_rectangle_source[1][1] - cy1),
            (inner_rectangle_source[2][0] - cx1, inner_rectangle_source[2][1] - cy1),
            (inner_rectangle_source[3][0] - cx1, inner_rectangle_source[3][1] - cy1),
        ]
        
        # Update dimensions
        h_src, w_src = image.shape[:2]
        print(f" [pre-crop→{w_src}x{h_src}]", end="")
    
    print()  # newline
    
    # 2. Create dense mesh (V2: returns grid dimensions for 2D smoothness)
    grid_size = max(24, min(w_src, h_src) // 16)
    src_pts, grid_rows, grid_cols = create_dense_mesh(h_src, w_src, grid_size)
    # 3. Initialize destination points (smooth proportional mapping)
    # Pre-crop already matches aspect ratio, so simple proportional init is smooth
    # and avoids discontinuities that cause TPS artifacts.
    dst_pts_init = []
    for pt in src_pts:
        ratio_x = pt[0] / max(w_src, 1)
        ratio_y = pt[1] / max(h_src, 1)
        dst_x = ratio_x * target_w
        dst_y = ratio_y * target_h
        dst_pts_init.append([dst_x, dst_y])
    dst_pts_init = np.array(dst_pts_init, dtype=np.float32)
    
    # 4. Extract salient boxes
    inner_src_box = [
        inner_rectangle_source[0][0],  # x1
        inner_rectangle_source[0][1],  # y1
        inner_rectangle_source[2][0],  # x2
        inner_rectangle_source[2][1]   # y2
    ]
    inner_dest_box = [
        inner_rectangle_dest[0][0],
        inner_rectangle_dest[0][1],
        inner_rectangle_dest[2][0],
        inner_rectangle_dest[2][1]
    ]
    
    # 5. Optimize mesh to preserve salient regions (V2: 2D grid-aware smoothness)
    dst_pts_optimized = optimize_mesh_warp(
        src_pts, dst_pts_init, saliency_map,
        inner_src_box, inner_dest_box,
        (h_src, w_src), (target_h, target_w),
        grid_rows=grid_rows, grid_cols=grid_cols,
        salient_weight=50.0,
        background_weight=5.0,
        smooth_lambda=15.0,
    )
    
    # 6. Apply mesh warp via griddata linear remap (fast; TPS O(N³) is too slow for 29 cells)
    warped = apply_mesh_warp_remap(image, src_pts, dst_pts_optimized, (target_h, target_w))

    _save_warp_grid_images(
        image,
        warped,
        src_pts,
        dst_pts_optimized,
        grid_rows,
        grid_cols,
        debug_dir,
        image_id,
    )
    
    # 7. Also warp the saliency mask for accurate Ma/Ms evaluation
    # Convert saliency_map to uint8 if float
    if saliency_map.dtype in (np.float32, np.float64):
        sal_uint8 = (np.clip(saliency_map, 0, 1) * 255).astype(np.uint8)
    else:
        sal_uint8 = saliency_map.astype(np.uint8)
    
    # cv2.remap works fine with 2D grayscale — no need to expand dims
    warped_saliency = apply_mesh_warp_remap(sal_uint8, src_pts, dst_pts_optimized, (target_h, target_w))
    
    # Ensure 2D output
    if len(warped_saliency.shape) == 3:
        warped_saliency = warped_saliency[:, :, 0]
    
    return warped, warped_saliency

def retarget_seam_carving(image, target_width, target_height):
    scale_factor = max(target_width/image.shape[1], target_height/image.shape[0])
    scaled = cv2.resize(image, (int(image.shape[1]*scale_factor), int(image.shape[0]*scale_factor)))
    dst = seam_carving.resize(
        scaled[:,:,0:3], (target_width, target_height),
        energy_mode='backward',   # Choose from {backward, forward}
        order='height-first',  # Choose from {width-first, height-first}
        keep_mask=None
    )
    alpha_channel = np.zeros((dst.shape[0], dst.shape[1], 1), dtype=np.uint8)+255
    new_dst = np.concatenate([dst, alpha_channel], axis=2)
    return new_dst
    
    

'''
part: partition dict of the format
    {'index': 12,
    'coords': [[796.0365929472149, 609.0],.....],
    'foreground': [x1,x2,y1,y2]}
 
image: image dict of the format
    {'filename': '02.jpg', 'foreground': [315, 700, 1, 1043], 'assigned_part': 12}
'''
from shapely.affinity import scale, translate as shapely_translate, scale as shapely_scale


def smart_cover_crop(image, foreground_box, target_width, target_height, bbox_overflow_threshold=1.2):
    """
    Smart cover crop V2: Prioritize keeping BBox visible.
    
    Strategy:
    1. Scale full frame to cover target (like before)
    2. Use sliding window to optimize crop position so bbox is maximally contained
    3. If scaled bbox is too large for target (> threshold) → fallback to blur padding
    
    Args:
        image: Source image (RGBA or RGB), shape (H, W, C)
        foreground_box: [x1, x2, y1, y2] bounding box of main object (NOTE: x1,x2,y1,y2 format!)
        target_width: Target cell width
        target_height: Target cell height
        bbox_overflow_threshold: If scaled bbox > target * threshold, use blur fallback (default 1.2)
    
    Returns:
        Cropped and scaled RGBA image of size (target_height, target_width, 4), debug_info
    """
    h_src, w_src = image.shape[:2]
    
    # Handle foreground_box format: [x1, x2, y1, y2] -> [x1, y1, x2, y2] for internal use
    if foreground_box is None or len(foreground_box) < 4:
        # No bbox provided - use center 80%
        foreground_box = [int(w_src * 0.1), int(w_src * 0.9), int(h_src * 0.1), int(h_src * 0.9)]
    
    x1, x2, y1, y2 = foreground_box
    
    # Clamp foreground box to image bounds
    x1 = max(0, min(w_src - 1, int(x1)))
    x2 = max(x1 + 1, min(w_src, int(x2)))
    y1 = max(0, min(h_src - 1, int(y1)))
    y2 = max(y1 + 1, min(h_src, int(y2)))
    
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    # === STEP 1: Calculate cover scale ===
    scale_cover = max(target_width / w_src, target_height / h_src)
    
    scaled_w = int(w_src * scale_cover)
    scaled_h = int(h_src * scale_cover)
    
    # Scaled bbox dimensions
    s_bbox_w = bbox_w * scale_cover
    s_bbox_h = bbox_h * scale_cover
    
    # === STEP 2: Check if bbox is too large for target ===
    # If bbox overflows target by more than threshold, use blur padding
    if s_bbox_w > target_width * bbox_overflow_threshold or s_bbox_h > target_height * bbox_overflow_threshold:
        return _crop_with_blur_padding(image, foreground_box, target_width, target_height)
    
    # === STEP 3: Resize image ===
    if scaled_w > 0 and scaled_h > 0:
        scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        scaled = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        scaled_w, scaled_h = target_width, target_height
        scale_cover = 1.0
    
    # === STEP 4: Sliding window optimization to contain bbox ===
    # Scaled bbox coordinates
    s_x1, s_y1 = x1 * scale_cover, y1 * scale_cover
    s_x2, s_y2 = x2 * scale_cover, y2 * scale_cover
    
    # Start with center crop
    crop_x = (scaled_w - target_width) / 2
    crop_y = (scaled_h - target_height) / 2
    
    # Horizontal adjustment: shift to contain bbox
    # If left edge of bbox is cut off (s_x1 < crop_x), shift left
    if s_x1 < crop_x:
        crop_x = max(0, s_x1 - 5)  # Small padding
    # If right edge of bbox is cut off, shift right
    if s_x2 > crop_x + target_width:
        crop_x = min(scaled_w - target_width, s_x2 - target_width + 5)
    
    # Vertical adjustment: PRIORITIZE HEAD (top of bbox)
    # If top of bbox is cut off, shift up
    if s_y1 < crop_y:
        crop_y = max(0, s_y1 - 5)
    # If bottom is cut off, we can accept some cut (prioritize top)
    # Only shift down if top is already visible
    if s_y2 > crop_y + target_height and s_y1 >= crop_y:
        crop_y = min(scaled_h - target_height, s_y2 - target_height + 5)
    
    # Clamp crop coordinates
    crop_x = max(0, min(scaled_w - target_width, crop_x))
    crop_y = max(0, min(scaled_h - target_height, crop_y))
    crop_x, crop_y = int(crop_x), int(crop_y)
    
    # === STEP 5: Crop target region ===
    final = scaled[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
    
    # Ensure exact size
    if final.shape[0] != target_height or final.shape[1] != target_width:
        final = cv2.resize(final, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Ensure BGRA
    if len(final.shape) == 2:
        final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGRA)
    elif final.shape[2] == 3:
        alpha = np.ones((final.shape[0], final.shape[1], 1), dtype=np.uint8) * 255
        final = np.concatenate([final, alpha], axis=2)
    
    # Calculate how much of bbox is visible in crop
    visible_x1 = max(crop_x, s_x1)
    visible_y1 = max(crop_y, s_y1)
    visible_x2 = min(crop_x + target_width, s_x2)
    visible_y2 = min(crop_y + target_height, s_y2)
    
    visible_area = max(0, visible_x2 - visible_x1) * max(0, visible_y2 - visible_y1)
    bbox_area = s_bbox_w * s_bbox_h
    bbox_coverage = visible_area / (bbox_area + 1e-6)
    
    # Debug info
    debug_info = {
        'crop_box': (int(crop_x / scale_cover), int(crop_y / scale_cover), 
                     int((crop_x + target_width) / scale_cover), int((crop_y + target_height) / scale_cover)),
        'scale': scale_cover,
        'final_size': (final.shape[1], final.shape[0]),
        'bbox_coverage': bbox_coverage,
        'method': 'sliding_window'
    }
    
    return final, debug_info


def _crop_with_blur_padding(image, foreground_box, target_width, target_height):
    """
    Fallback: Create blurred background, place fitted image on top.
    Used when bbox is too large to fit in target via normal crop.
    
    Args:
        image: Source image
        foreground_box: [x1, x2, y1, y2] bbox
        target_width, target_height: Target dimensions
    
    Returns:
        Final image, debug_info
    """
    h, w = image.shape[:2]
    
    # 1. Scale image to FIT (smaller than or equal to target)
    scale_fit = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale_fit), int(h * scale_fit)
    if new_w <= 0: new_w = 1
    if new_h <= 0: new_h = 1
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Create blurred background (scale to cover, crop center, blur)
    scale_blur = max(target_width / w, target_height / h)
    blur_w, blur_h = int(w * scale_blur), int(h * scale_blur)
    if blur_w <= 0: blur_w = 1
    if blur_h <= 0: blur_h = 1
    bg_img = cv2.resize(image, (blur_w, blur_h), interpolation=cv2.INTER_LINEAR)
    
    # Crop center of blurred background
    bx = max(0, (blur_w - target_width) // 2)
    by = max(0, (blur_h - target_height) // 2)
    bg_crop = bg_img[by:by + target_height, bx:bx + target_width]
    
    # Ensure correct size
    if bg_crop.shape[0] != target_height or bg_crop.shape[1] != target_width:
        bg_crop = cv2.resize(bg_crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Apply heavy blur
    bg_crop = cv2.GaussianBlur(bg_crop, (51, 51), 0)
    
    # Darken background slightly
    bg_crop = (bg_crop.astype(np.float32) * 0.6).clip(0, 255).astype(np.uint8)
    
    # 3. Paste fitted image on center of blurred background
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Create output canvas
    final = bg_crop.copy()
    final[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    
    # Ensure BGRA
    if len(final.shape) == 2:
        final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGRA)
    elif final.shape[2] == 3:
        alpha = np.ones((final.shape[0], final.shape[1], 1), dtype=np.uint8) * 255
        final = np.concatenate([final, alpha], axis=2)
    
    debug_info = {
        'crop_box': (0, 0, w, h),
        'scale': scale_fit,
        'final_size': (final.shape[1], final.shape[0]),
        'bbox_coverage': 1.0,  # Full bbox is visible
        'method': 'blur_padding'
    }
    
    return final, debug_info


def generate_image_patch(part_dict, image_dict, image_directory, whole_canvas_height, magnification=1.0, 
                        debug_dir=None, skip_warp=False):
    """
    Generate image patch to fit partition shape.
    
    Args:
        part_dict: Partition dict with coords, foreground
        image_dict: Image dict with filename, foreground, foreground_exists
        image_directory: Directory containing images
        whole_canvas_height: Canvas height for coordinate transformation
        magnification: Scale factor for output
        debug_dir: Optional debug directory for visualization
        skip_warp: Kept for API compatibility, ignored in warp-only mode
    
    Returns:
        patch: Image patch (RGBA)
        patch_origin: (row, col) for placement on canvas
    """
    #part = scale(part, xfact=magnification, yfact=magnification, origin=(0,0))
    
    polygon = Polygon(part_dict['coords'])
    polygon_scaled = scale(polygon, xfact=magnification, yfact=magnification, origin=(0,0))
    
    bounding_box = polygon_scaled.bounds
    polygon_space_origin = bounding_box[0], bounding_box[1]
    width = int(bounding_box[2] - bounding_box[0])+1
    height = int(bounding_box[3] - bounding_box[1])+1
        
    # Load images
    image = load_color_image(join(image_directory, image_dict["filename"]))
    image_fg = normalize_foreground_bbox(image_dict.get('foreground', None), image.shape[1], image.shape[0])
    # enlarge the inner rectangle to add margins around main object
    enlarged_inner, touch_boundary = adjust_inner_rec([0, 0, image.shape[1], image.shape[0]], image_fg)
    
    touch_boundary = False
    
    # Create unique image ID for debug files
    image_id = image_dict.get("filename", "unknown").split('.')[0]

    if image_dict['foreground_exists'] and not touch_boundary:
        outer_rectangle_source = [(0,0), (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, image.shape[0])]
        inner_rectangle_source = [(enlarged_inner[0], enlarged_inner[1]),
                           (enlarged_inner[2], enlarged_inner[1]),
                           (enlarged_inner[2], enlarged_inner[3]),
                           (enlarged_inner[0], enlarged_inner[3])]

        new_x1 = part_dict['foreground'][0]*magnification - polygon_space_origin[0]
        new_x2 = part_dict['foreground'][2]*magnification - polygon_space_origin[0]
        new_y1 = part_dict['foreground'][1]*magnification - polygon_space_origin[1]
        new_y2 = part_dict['foreground'][3]*magnification - polygon_space_origin[1]

        outer_rectangle_dest = [(0,0), (width, 0), (width, height), (0, height)]
        inner_rectangle_dest = [(new_x1, new_y1),
                           (new_x2, new_y1),
                           (new_x2, new_y2),
                           (new_x1, new_y2)]

        # Crop the image to the size of part proportionally
        x1_diff_source = enlarged_inner[0] # distance to left outer to left inner
        x2_diff_source = image.shape[1] - enlarged_inner[2]
        y1_diff_source = enlarged_inner[1]
        y2_diff_source = image.shape[0] - enlarged_inner[3]

        inner_width_source = enlarged_inner[2] - enlarged_inner[0]
        inner_height_source = enlarged_inner[3] - enlarged_inner[1]

        x1_diff_dest = new_x1
        x2_diff_dest = width - new_x2
        y1_diff_dest = new_y1
        y2_diff_dest = height - new_y2

        inner_width_dest = new_y2 - new_y1
        inner_height_dest = new_x2 - new_x1

        new_x1_outer_source = 0
        new_x2_outer_source = image.shape[1]
        new_y1_outer_source = 0
        new_y2_outer_source = image.shape[0]

        if x1_diff_dest / inner_width_dest < x1_diff_source/inner_width_source:
            new_x1_outer_source = enlarged_inner[0] - x1_diff_dest / inner_width_dest * inner_width_source

        if x2_diff_dest / inner_width_dest < x2_diff_source/inner_width_source:
            new_x2_outer_source = enlarged_inner[2] + x2_diff_dest / inner_width_dest * inner_width_source

        if y1_diff_dest / inner_height_dest < y1_diff_source/inner_height_source:
            new_y1_outer_source = enlarged_inner[1] - y1_diff_dest / inner_height_dest * inner_height_source

        if y2_diff_dest / inner_height_dest < y2_diff_source/inner_height_source:
            new_y2_outer_source = enlarged_inner[3] + y2_diff_dest / inner_height_dest * inner_height_source

        new_outer_source = [int(new_x1_outer_source), int(new_x2_outer_source), int(new_y1_outer_source), int(new_y2_outer_source)]
        new_outer_rectangle_source = [(new_outer_source[0], new_outer_source[2]),
                           (new_outer_source[1], new_outer_source[2]),
                           (new_outer_source[1], new_outer_source[3]),
                           (new_outer_source[0], new_outer_source[3])]
        retargeted = retarget_warp(
            image, new_outer_rectangle_source, inner_rectangle_source, outer_rectangle_dest,
            inner_rectangle_dest, debug_dir=debug_dir, image_id=image_id
        )
    else:
        # Warp-only fallback: use full image as both outer and inner constraints.
        outer_rectangle_source = [(0,0), (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, image.shape[0])]
        inner_rectangle_source = outer_rectangle_source
        outer_rectangle_dest = [(0,0), (width, 0), (width, height), (0, height)]
        inner_rectangle_dest = outer_rectangle_dest
        retargeted = retarget_warp(
            image, outer_rectangle_source, inner_rectangle_source, outer_rectangle_dest,
            inner_rectangle_dest, debug_dir=debug_dir, image_id=image_id
        )

    # Create polygon mask - need to flip Y coords to match image coordinate system
    polygon_mask = np.zeros((height, width), np.uint8)
    
    # Get local coords and flip Y to match image coords (Y=0 at top)
    local_coords = polygon2local_coordinate(polygon_scaled)
    # Flip Y: new_y = height - old_y
    local_coords_flipped = local_coords.copy()
    local_coords_flipped[:, 1] = height - local_coords[:, 1]
    
    cv2.fillPoly(polygon_mask, [local_coords_flipped], (255))
    
    #if magnification > 1:
    blur = cv2.GaussianBlur(polygon_mask,(7,7),0)
    thresh, smoothed = cv2.threshold(blur, 100, 255,cv2.THRESH_BINARY)
    polygon_mask = smoothed

    # Calculate patch origin - convert from XY to row/col
    patch_origin = xy2rowcol(bounding_box[0], bounding_box[3], whole_canvas_height)
    
    # Apply mask - pixels outside polygon become transparent
    retargeted[polygon_mask == 0] = 0
    return retargeted, patch_origin


def _polygon_visual_center(poly):
    """Get the visual center of a polygon - the best interior point for placing content.
    Uses centroid if inside polygon, otherwise representative_point for concave shapes."""
    centroid = poly.centroid
    if poly.contains(centroid):
        return centroid.x, centroid.y
    # For concave polygons, centroid can be outside - use representative_point
    rep = poly.representative_point()
    return rep.x, rep.y


def _add_voronoi_white_space(collage_img, layout, thickness=4):
    """Add opaque white separators between adjacent Voronoi cells."""
    if collage_img is None or collage_img.size == 0:
        return collage_img

    parts = layout.get("parts", [])
    if not parts:
        return collage_img.copy()

    thickness = max(1, int(thickness))
    img = collage_img.copy()
    h_img, w_img = img.shape[:2]

    w_json = float(layout.get("width", w_img))
    h_json = float(layout.get("height", h_img))
    sx = w_img / w_json if w_json > 0 else 1.0
    sy = h_img / h_json if h_json > 0 else 1.0

    # Draw boundaries on color channels only, preserving alpha if available.
    if img.ndim == 2:
        draw_layer = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif img.shape[2] == 4:
        draw_layer = img[:, :, :3].copy()
        alpha = img[:, :, 3].copy()
    else:
        draw_layer = img.copy()
        alpha = None

    labels = np.full((h_img, w_img), -1, dtype=np.int32)
    for part_idx, part in enumerate(parts):
        # Use raw power polygon coords (now stored in 'coords') to match collage rendering
        coords = part.get("coords", [])
        if len(coords) < 3:
            continue
        pts = np.array(coords, dtype=np.float32)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        pts = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        part_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(part_mask, [pts], 255)
        labels[part_mask > 0] = part_idx

    if alpha is not None:
        visible = alpha > 0
    else:
        visible = np.any(draw_layer > 0, axis=2)

    line_bool = np.zeros((h_img, w_img), dtype=bool)

    # Mark only one side of each adjacency pair to avoid double-thick separators.
    diff_h = labels[:, 1:] != labels[:, :-1]
    valid_h = (labels[:, 1:] >= 0) & (labels[:, :-1] >= 0)
    vis_h = visible[:, 1:] & visible[:, :-1]
    line_bool[:, 1:] |= diff_h & valid_h & vis_h

    diff_v = labels[1:, :] != labels[:-1, :]
    valid_v = (labels[1:, :] >= 0) & (labels[:-1, :] >= 0)
    vis_v = visible[1:, :] & visible[:-1, :]
    line_bool[1:, :] |= diff_v & valid_v & vis_v

    line_mask = (line_bool.astype(np.uint8) * 255)
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        line_mask = cv2.dilate(line_mask, kernel, iterations=1)

    draw_layer[line_mask > 0] = 255

    if alpha is not None:
        alpha[line_mask > 0] = 255
        return np.dstack([draw_layer, alpha])
    return draw_layer


def _render_voronoi_collage(layout, input_image_collection_folder, output_dir, scaling_factor, skip_warp=False, warp_mask_dir=None):
    """
    Render collage for Voronoi layout in warp-only mode.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    
    W = layout['width'] * scaling_factor
    H = layout['height'] * scaling_factor
    canvas = np.zeros((H, W, 4), dtype=np.uint8)  # BGRA
    
    bbox_retention = []  # For bbox retention score
    images_pixel = []

    # Warp-only mode: always pre-compute saliency maps.
    precomputed_saliency = {}
    import time
    t0 = time.time()
    if warp_mask_dir and os.path.isdir(warp_mask_dir):
        # Load pre-computed masks as saliency
        print(f"[BatchWarp] Loading pre-computed masks from: {warp_mask_dir}")
        for img_dict in layout['images']:
            fname = img_dict['filename']
            base = os.path.splitext(fname)[0]
            # Try common mask naming patterns
            mask_path = None
            for ext in ['.png', '.jpg']:
                for candidate in [join(warp_mask_dir, base + ext),
                                  join(warp_mask_dir, base + '_mask' + ext)]:
                    if os.path.isfile(candidate):
                        mask_path = candidate
                        break
                if mask_path:
                    break
            if mask_path:
                img_path = join(input_image_collection_folder, fname)
                src_img = cv2.imread(img_path)
                if src_img is not None:
                    sal = load_mask_as_saliency(mask_path, src_img.shape[0], src_img.shape[1])
                    if sal is not None:
                        precomputed_saliency[fname] = sal
        print(f"[BatchWarp] Loaded {len(precomputed_saliency)}/{len(layout['images'])} masks ({time.time()-t0:.1f}s)")
    else:
        # Batch GPU saliency computation
        print(f"[BatchWarp] Computing saliency for {len(layout['images'])} images via batch GPU...")
        batch_images = []
        batch_fnames = []
        for img_dict in layout['images']:
            img_path = join(input_image_collection_folder, img_dict['filename'])
            img = cv2.imread(img_path)
            if img is not None:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                batch_images.append(img)
                batch_fnames.append(img_dict['filename'])
        saliency_maps = batch_compute_saliency_gpu(batch_images)
        for fname, sal in zip(batch_fnames, saliency_maps):
            # Apply center bias + expansion like the hybrid version
            sal = apply_center_bias(sal, strength=0.4)
            sal = expand_saliency_region(sal, threshold=0.15)
            precomputed_saliency[fname] = sal
        print(f"[BatchWarp] Computed {len(precomputed_saliency)} saliency maps ({time.time()-t0:.1f}s)")

    for img_dict in layout['images']:
        img_path = join(input_image_collection_folder, img_dict['filename'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Cannot load: {img_dict['filename']}")
            images_pixel.append(None)
        else:
            # Convert BGR to BGRA
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            images_pixel.append(img)
    
    def _optimize_cover_crop_bbox(poly, img_shape, fg_smart, target_w, target_h, search_steps=9):
        """
        Search crop position (cover mode) to maximize bbox overlap with polygon mask.
        Uses Voronoi polygon (not its bounding box) as the reference target:
        - Zoom boost based on polygon fill ratio so character fills visible area
        - Search starts aligned to polygon centroid
        Returns: (crop_x, crop_y, crop_w, crop_h, best_score)
        """
        h, w = img_shape[:2]
        x1, x2, y1, y2 = fg_smart
        fg_cx = (x1 + x2) / 2.0
        fg_cy = (y1 + y2) / 2.0

        # Precompute polygon bounds for mapping
        px_min, py_min, px_max, py_max = poly.bounds
        cell_w = max(px_max - px_min, 1e-6)
        cell_h = max(py_max - py_min, 1e-6)

        # Scale based on polygon fill ratio (use polygon area, not bounding box)
        # Sparse/narrow polygons get more zoom so character fills the visible polygon area
        poly_fill_ratio = max(0.12, min(1.0, poly.area / max(cell_w * cell_h, 1.0)))
        # Narrow cells (fill_ratio < 0.5) get up to 3.5x boost; regular cells up to 2.0x
        max_boost = 3.5 if poly_fill_ratio < 0.5 else 2.0
        zoom_boost = min(max_boost, 1.0 / (poly_fill_ratio ** 0.5))
        
        # Cap zoom_boost so ISNet bbox still fits in the crop window
        # Without this cap, large bboxes get clipped by the smaller crop window
        cover_base = max(target_w / w, target_h / h)
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        max_zoom_w = target_w / (cover_base * bbox_w)
        max_zoom_h = target_h / (cover_base * bbox_h)
        max_zoom_for_bbox = min(max_zoom_w, max_zoom_h)
        if max_zoom_for_bbox < 1.0:
            zoom_boost = 1.0  # bbox doesn't fit even at 1x — don't zoom in
        else:
            zoom_boost = min(zoom_boost, max_zoom_for_bbox * 0.85)  # 85% margin
        
        scale = cover_base * max(1.0, zoom_boost)

        crop_w = target_w / scale
        crop_h = target_h / scale

        # Clamp crop size
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        # Start position: align ISNet bbox center with polygon centroid
        # Polygon centroid in crop coordinates (polygon local coords → crop scale)
        img_poly_cx = (poly.centroid.x - px_min) / cell_w * crop_w
        img_poly_cy = (poly.centroid.y - py_min) / cell_h * crop_h
        base_x = fg_cx - img_poly_cx
        base_y = fg_cy - img_poly_cy

        # Candidate offsets in a grid
        dx_span = min(w - crop_w, max(0.0, crop_w * 0.4))
        dy_span = min(h - crop_h, max(0.0, crop_h * 0.4))

        xs = np.linspace(base_x - dx_span, base_x + dx_span, search_steps)
        ys = np.linspace(base_y - dy_span, base_y + dy_span, search_steps)

        # Clamp candidates
        xs = [max(0, min(x, w - crop_w)) for x in xs]
        ys = [max(0, min(y, h - crop_h)) for y in ys]

        # bbox polygon in image coords
        bbox_poly = ShapelyPolygon([
            (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        ])
        bbox_area = max((x2 - x1) * (y2 - y1), 1.0)

        best = (base_x, base_y, crop_w, crop_h, -1.0, 0.0)
        scale_to_crop_x = crop_w / cell_w
        scale_to_crop_y = crop_h / cell_h
        # Precompute polygon scaled to crop space (translate only changes per iteration)
        poly_scaled = shapely_translate(poly, -px_min, -py_min)
        poly_scaled = shapely_scale(poly_scaled, xfact=scale_to_crop_x, yfact=scale_to_crop_y, origin=(0, 0))

        for cx in xs:
            for cy in ys:
                poly_in_img = shapely_translate(poly_scaled, cx, cy)

                try:
                    inter_area = poly_in_img.intersection(bbox_poly).area
                except Exception:
                    inter_area = 0.0

                coverage = inter_area / bbox_area

                # Slight bias to keep crop near polygon centroid alignment
                dist = ((cx - base_x) ** 2 + (cy - base_y) ** 2) ** 0.5
                score = coverage - 0.001 * dist

                if score > best[4]:
                    best = (cx, cy, crop_w, crop_h, score, coverage)

        return best

    def _bbox_fit_crop(img_shape, fg_smart, target_w, target_h):
        """Crop so that bbox fits within target after scaling (bbox-priority, FIT mode)."""
        h, w = img_shape[:2]
        x1, x2, y1, y2 = fg_smart
        fg_w = max(1.0, x2 - x1)
        fg_h = max(1.0, y2 - y1)
        fg_cx = (x1 + x2) / 2.0
        fg_cy = (y1 + y2) / 2.0

        # FIT mode: min() ensures full bbox is visible (max() was COVER which clips bbox)
        scale = min(target_w / fg_w, target_h / fg_h)
        # Ensure image still covers target (no empty padding)
        scale_cover = max(target_w / w, target_h / h)
        scale = max(scale, scale_cover)
        crop_w = target_w / scale
        crop_h = target_h / scale

        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        crop_x = fg_cx - crop_w / 2
        crop_y = fg_cy - crop_h / 2

        crop_x = max(0, min(crop_x, w - crop_w))
        crop_y = max(0, min(crop_y, h - crop_h))

        return crop_x, crop_y, crop_w, crop_h

    def _rotate_image_and_bbox(image, fg_smart, angle):
        """
        Rotate image by SMALL angle (degrees) and return rotated image + transformed bbox.
        For small angles (±5°, ±10°), we rotate and then crop to original size to avoid padding.
        """
        if angle == 0:
            return image, fg_smart
        
        h, w = image.shape[:2]
        x1, x2, y1, y2 = fg_smart
        cx, cy = w / 2, h / 2
        
        # For small angles: rotate around center, then crop to original size
        # This avoids creating fake background/padding
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        # Rotate image - use border replicate to minimize artifacts
        rotated = cv2.warpAffine(image, M, (w, h), 
                                  flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE)
        
        # Transform bbox corners
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ]).T
        
        rotated_corners = M @ corners
        
        # Get new bbox (axis-aligned)
        new_x1 = max(0, rotated_corners[0].min())
        new_x2 = min(w, rotated_corners[0].max())
        new_y1 = max(0, rotated_corners[1].min())
        new_y2 = min(h, rotated_corners[1].max())
        
        new_fg = [new_x1, new_x2, new_y1, new_y2]
        
        return rotated, new_fg
    
    def _compute_smart_rotation_angles(fg_smart, target_w, target_h):
        """
        Compute smart rotation angles based on bbox and target aspect ratio mismatch.
        ONLY uses SMALL rotations (±5°, ±10°) to avoid distorting people/characters.
        NO 90° rotations - people would be lying down!
        """
        # Only small angle rotations to fine-tune alignment
        # Large rotations (90°) would make people lie down - NOT acceptable!
        base_angles = [0, -5, 5, -10, 10, -7, 7]
        
        return base_angles

    # =====================================================
    # PARALLEL WARP: Collect all warp jobs, run in parallel
    # =====================================================
    warp_grid_dir = os.path.join(output_dir, "warped_saliency_masks", "warp_grids")
    os.makedirs(warp_grid_dir, exist_ok=True)

    def _single_warp_job(args):
        """Worker function for parallel warp. Returns (idx, (warped_img, warped_sal)) or (idx, None)."""
        (job_idx, img_source, outer_src, inner_src, outer_dst, inner_dst,
         sal_map, target_w_job, target_h_job, fname, dbg_dir) = args
        try:
            warped, warped_sal = retarget_warp(
                img_source, outer_src, inner_src, outer_dst, inner_dst,
                debug_dir=dbg_dir,
                image_id=fname,
                saliency_map=sal_map
            )
            return (job_idx, (warped, warped_sal))
        except Exception as e:
            print(f"  [WARN] Warp failed for {fname}: {e}")
            return (job_idx, None)

    # Phase 1: Collect all jobs
    warp_jobs = []  # list of (job_idx, args_tuple)
    job_metadata = []  # per-job: (idx, img_dict, scaled_coords, minx, miny, maxx, maxy, target_w, target_h, fg_smart, img_source)

    for idx, img_dict in enumerate(layout['images']):
        part_idx = img_dict['assigned_part']
        part = layout['parts'][part_idx]
        coords = part['coords']
        
        # Scale coords
        scaled_coords = [(x * scaling_factor, y * scaling_factor) for x, y in coords]
        
        # Create polygon
        poly = ShapelyPolygon(scaled_coords)
        if poly.is_empty:
            continue
            
        # Get bounding box (use floor/ceil to avoid clipping polygon corners)
        minx, miny, maxx, maxy = poly.bounds
        minx, miny = int(math.floor(minx)), int(math.floor(miny))
        maxx, maxy = int(math.ceil(maxx)), int(math.ceil(maxy))
        
        # Clamp to canvas
        minx = max(0, minx)
        miny = max(0, miny)
        maxx = min(W, maxx)
        maxy = min(H, maxy)
        
        target_w = maxx - minx
        target_h = maxy - miny
        
        if target_w <= 0 or target_h <= 0:
            continue
        
        # Compute polygon in local (bounding box) coordinates
        poly_local = ShapelyPolygon([(x - minx, y - miny) for x, y in scaled_coords])
            
        img_source = images_pixel[idx]
        if img_source is None:
            continue
        
        fg_smart = normalize_foreground_bbox(
            img_dict.get('foreground', None),
            img_source.shape[1],
            img_source.shape[0],
        )

        meta = {
            'idx': idx, 'img_dict': img_dict, 'scaled_coords': scaled_coords,
            'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy,
            'target_w': target_w, 'target_h': target_h,
            'fg_smart': fg_smart, 'img_source': img_source,
            'poly_local': poly_local,
            'warp_job_idx': -1  # -1 = no warp job for this cell
        }

        src_h, src_w = img_source.shape[:2]
        
        # Add margins around fg (same as backup's adjust_inner_rec)
        enlarged_inner, _ = adjust_inner_rec([0, 0, src_w, src_h], fg_smart)
        
        outer_rect_src = [(0,0), (src_w,0), (src_w,src_h), (0,src_h)]
        inner_rect_src = [
            (enlarged_inner[0], enlarged_inner[1]),
            (enlarged_inner[2], enlarged_inner[1]),
            (enlarged_inner[2], enlarged_inner[3]),
            (enlarged_inner[0], enlarged_inner[3])
        ]
        outer_rect_dst = [(0,0), (target_w,0), (target_w,target_h), (0,target_h)]
        
        # Proportional mapping: fg lands at same relative position in target
        # → dst_pts_init ≈ inner_rect_dst → minimal deformation (spec requirement)
        new_x1 = enlarged_inner[0] / src_w * target_w
        new_x2 = enlarged_inner[2] / src_w * target_w
        new_y1 = enlarged_inner[1] / src_h * target_h
        new_y2 = enlarged_inner[3] / src_h * target_h
        
        inner_rect_dst = [
            (new_x1, new_y1), (new_x2, new_y1),
            (new_x2, new_y2), (new_x1, new_y2)
        ]
        
        sal_map = precomputed_saliency.get(img_dict['filename'], None)
        warp_idx = len(warp_jobs)
        job_args = (warp_idx, img_source, outer_rect_src, inner_rect_src,
                   outer_rect_dst, inner_rect_dst, sal_map, target_w, target_h,
                   img_dict['filename'], warp_grid_dir)
        warp_jobs.append(job_args)
        meta['warp_job_idx'] = warp_idx

        job_metadata.append(meta)

    # Phase 2: Run warp jobs in parallel
    warp_results = {}  # job_idx -> warped_img or None
    if warp_jobs:
        import time as _time
        from concurrent.futures import ThreadPoolExecutor

        t0 = _time.time()
        num_workers = min(len(warp_jobs), max(1, os.cpu_count() or 4))
        print(f"[ParallelWarp] Processing {len(warp_jobs)} warp jobs with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = list(executor.map(_single_warp_job, warp_jobs))
        for job_idx, warped in futures:
            warp_results[job_idx] = warped
        
        elapsed = _time.time() - t0
        success = sum(1 for v in warp_results.values() if v is not None)
        failed_job_ids = [jid for jid, val in warp_results.items() if val is None]
        print(f"[Warp] Done: {success}/{len(warp_jobs)} warped successfully ({elapsed:.1f}s)")
        if failed_job_ids:
            failed_names = [warp_jobs[jid][9] for jid in failed_job_ids[:10]]
            print(f"[Warp][WARN] Failed jobs: {len(failed_job_ids)} | sample={failed_names}")

    # Create directory for warped saliency masks (for accurate Ma/Ms evaluation)
    warped_sal_dir = os.path.join(output_dir, "warped_saliency_masks")
    os.makedirs(warped_sal_dir, exist_ok=True)
    pre_warp_dir = os.path.join(warped_sal_dir, "pre_warp_crops")
    post_warp_dir = os.path.join(warped_sal_dir, "post_warp_crops")
    os.makedirs(pre_warp_dir, exist_ok=True)
    os.makedirs(post_warp_dir, exist_ok=True)
    
    # Phase 3: Composite results onto canvas
    composited_count = 0
    skipped_count = 0
    for meta in job_metadata:
        idx = meta['idx']
        img_dict = meta['img_dict']
        scaled_coords = meta['scaled_coords']
        minx, miny = meta['minx'], meta['miny']
        maxx, maxy = meta['maxx'], meta['maxy']
        target_w, target_h = meta['target_w'], meta['target_h']
        fg_smart = meta['fg_smart']
        img_source = meta['img_source']

        cropped_img = None
        warped_sal = None
        crop_debug = {}
        pre_crop = None

        wjidx = meta.get('warp_job_idx', -1)
        if wjidx >= 0:
            warp_result = warp_results.get(wjidx, None)
            if warp_result is not None:
                cropped_img, warped_sal = warp_result
                crop_debug = {'method': 'warp'}

        if cropped_img is None:
            print(f"  [WARN] Skip cell {idx} ({img_dict.get('filename', 'unknown')}): warp failed and warp-only mode is enabled.")
            skipped_count += 1
            continue
        retention_item = {
            "filename": str(img_dict.get("filename", "")),
            "bbox_coverage": None,
            "bbox_fit_fallback": bool(crop_debug.get("bbox_fit_fallback", False)),
            "rotation_angle": int(crop_debug.get("rotation_angle", 0))
        }
        
        # Create local polygon mask
        local_poly_pts = []
        for x, y in scaled_coords:
            lx = int(round(x - minx))
            ly = int(round(y - miny))
            lx = max(0, min(target_w - 1, lx))
            ly = max(0, min(target_h - 1, ly))
            local_poly_pts.append([lx, ly])
        
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(local_poly_pts)], 255)
        # Dilate 2px to fill sub-pixel gaps between adjacent Voronoi cells
        _gap_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, _gap_kernel, iterations=1)
        
        # Apply mask to cropped image
        cropped_img[:, :, 3] = mask  # Set alpha channel

        # Save pre-warp and post-warp crops for inspection
        try:
            pre_crop, _ = smart_cover_crop(img_source, fg_smart, target_w, target_h)
        except Exception:
            pre_crop = None

        if pre_crop is not None:
            if pre_crop.shape[0] != target_h or pre_crop.shape[1] != target_w:
                pre_crop = cv2.resize(pre_crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            if pre_crop.shape[2] == 3:
                alpha = np.ones((target_h, target_w, 1), dtype=np.uint8) * 255
                pre_crop = np.concatenate([pre_crop, alpha], axis=2)
            pre_crop[:, :, 3] = mask

            base_name = os.path.splitext(img_dict.get('filename', f'image_{idx}.png'))[0]
            pre_path = os.path.join(pre_warp_dir, base_name + ".png")
            cv2.imwrite(pre_path, pre_crop)

        base_name = os.path.splitext(img_dict.get('filename', f'image_{idx}.png'))[0]
        post_path = os.path.join(post_warp_dir, base_name + ".png")
        cv2.imwrite(post_path, cropped_img)
        
        # Save warped saliency mask for accurate Ma/Ms evaluation
        if warped_sal is not None:
            # Resize warped saliency to match cropped image size
            if warped_sal.shape[:2] != (target_h, target_w):
                warped_sal_resized = cv2.resize(warped_sal, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                warped_sal_resized = warped_sal
            
            # Apply same polygon mask to warped saliency
            if len(warped_sal_resized.shape) == 2:
                warped_sal_source = warped_sal_resized > 127
            else:
                warped_sal_source = warped_sal_resized[:,:,0] > 127
            warped_sal_bool = warped_sal_source & (mask > 0)

            sal_total = int(warped_sal_source.sum())
            if sal_total > 0:
                retention_item["bbox_coverage"] = float(warped_sal_bool.sum() / sal_total)
            
            # Convert to uint8 with 255 for white
            warped_sal_masked = warped_sal_bool.astype(np.uint8) * 255
            
            # Create full-canvas warped saliency mask
            full_warped_sal = np.zeros((H, W), dtype=np.uint8)
            full_warped_sal[miny:maxy, minx:maxx] = warped_sal_masked
            
            # Save to disk
            fname = img_dict.get('filename', f'image_{idx}.png')
            sal_path = os.path.join(warped_sal_dir, fname)
            cv2.imwrite(sal_path, full_warped_sal)

        bbox_retention.append(retention_item)
        
        # Paste to canvas
        roi = canvas[miny:maxy, minx:maxx]
        
        # Blend using alpha
        alpha = cropped_img[:, :, 3:4] / 255.0
        for c in range(3):
            roi[:, :, c] = (cropped_img[:, :, c] * alpha[:, :, 0] + 
                           roi[:, :, c] * (1 - alpha[:, :, 0])).astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], cropped_img[:, :, 3])
        
        canvas[miny:maxy, minx:maxx] = roi
        composited_count += 1
    
    # Save bbox retention summary
    valid_cov = [b["bbox_coverage"] for b in bbox_retention if b.get("bbox_coverage") is not None]
    avg_coverage = float(sum(valid_cov) / len(valid_cov)) if valid_cov else 0.0
    retention_path = join(output_dir, "bbox_retention.json")
    with open(retention_path, "w") as f:
        json.dump({
            "avg_coverage": avg_coverage,
            "count": len(valid_cov),
            "items": bbox_retention
        }, f, indent=2)
    
    # Apply shape mask to clip collage to original shape boundary
    shape_mask_path = join(output_dir, 'shape_mask_refined.png')
    if os.path.isfile(shape_mask_path):
        shape_mask = cv2.imread(shape_mask_path, cv2.IMREAD_GRAYSCALE)
        if shape_mask is not None:
            # Resize to canvas size
            if shape_mask.shape[0] != canvas.shape[0] or shape_mask.shape[1] != canvas.shape[1]:
                shape_mask = cv2.resize(shape_mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Clip alpha channel to shape mask
            canvas[:, :, 3] = np.minimum(canvas[:, :, 3], shape_mask)
            print(f"  [Voronoi] Applied shape mask from {shape_mask_path}")
    
    # Save - canvas is already BGRA from OpenCV, write directly
    cv2.imwrite(join(output_dir, 'collage.png'), canvas)
    
    # Save Voronoi collage with explicit white separators between adjacent cells.
    canvas_white_space = _add_voronoi_white_space(canvas, layout, thickness=4)
    cv2.imwrite(join(output_dir, 'collage_white_space.png'), canvas_white_space)
    
    assigned_parts = [im.get('assigned_part') for im in layout.get('images', [])]
    valid_assigned_parts = [p for p in assigned_parts if isinstance(p, int)]
    unique_parts = len(set(valid_assigned_parts))
    duplicate_assignments = len(valid_assigned_parts) - unique_parts
    print(
        f"  [Voronoi] Collage composition summary: attempted={len(job_metadata)}, "
        f"composited={composited_count}, skipped={skipped_count}, "
        f"unique_parts={unique_parts}, duplicate_assignments={duplicate_assignments}"
    )

def render_collage(input_image_collection_folder, output_dir, scaling_factor, enable_debug=False, skip_warp=False, warp_mask_dir=None):
    """
    Render collage from layout.
    
    Args:
        input_image_collection_folder: Folder with source images
        output_dir: Output directory for collage.png and collage_white_space.png
        scaling_factor: Magnification factor
        enable_debug: If True, create warp_debug_visualization folder with intermediate results
        skip_warp: Kept for API compatibility, ignored in warp-only mode
    """
    import os
    
    if skip_warp:
        print("[render_collage] skip_warp is ignored: pipeline is warp-only.")

    with open(join(output_dir, 'slicing_result.json'), 'r') as f:
        layout = json.load(f)

    if layout.get('layout_type') != 'voronoi':
        raise ValueError(
            "render_collage now supports Voronoi-only pipeline. "
            "Expected layout_type='voronoi' in slicing_result.json."
        )

    _render_voronoi_collage(
        layout,
        input_image_collection_folder,
        output_dir,
        scaling_factor,
        skip_warp=skip_warp,
        warp_mask_dir=warp_mask_dir,
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Render collage with optional debug visualization')
    parser.add_argument('input_dir', help='Input image collection folder')
    parser.add_argument('output_dir', help='Output directory for collage')
    parser.add_argument('scaling_factor', type=int, help='Magnification factor')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization of warp process')
    
    args = parser.parse_args()
    
    render_collage(args.input_dir, args.output_dir, args.scaling_factor, enable_debug=args.debug)
    
    if args.debug:
        debug_path = join(args.output_dir, 'warp_debug_visualization')
        print(f"\n[SUCCESS] Debug visualizations saved to: {debug_path}")
        print(f"  - 01_saliency_map_*.png: U2-Net saliency detection results")
        print(f"  - 02_mesh_grid_*.png: Source mesh grid overlaid on image")
        print(f"  - 03_salient_regions_source_*.png: Detected salient objects")
        print(f"  - 04_mesh_transformation_*.png: Mesh optimization (magenta=initial, red=optimized)")
        print(f"  - 05_warped_result_*.png: Before/after warp comparison")
