"""
Evaluation module for CAST collage layout metrics.

Metrics:
    Ma  - Saliency Area: fraction of shape covered by salient regions
    Mc  - Compactness (white space): fraction of shape NOT covered by images
    Mo  - Non-overlapping: pairwise overlap penalty between cells
    Mn  - Correlation Preservation (AIC): mean distance image vs category centroid
    Mn_timeline - Disabled for free-assignment runs; kept as NaN for output compatibility
    Mt  - Disabled for free-assignment runs; kept as NaN for output compatibility
    Ms  - Saliency Loss: fraction of salient pixels lost due to overlap
    LQ  - Layout Quality: fraction of shape covered by images (= 1 - Mc)
    DRE - Data Representation Error: MAE between target and actual cell sizes

Usage:
    python evaluation.py --output_dir tests/output_banana --shape tests/output_banana/_voronoi_temp.png
"""
# ...existing code...
import os
import glob
import json
import csv
import math
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

# -------------------------
# Utility IO / conversions
# -------------------------
def _image_info_name(img_info: Dict) -> str:
    """Resolve image filename from slicing metadata across schema variants."""
    name = img_info.get("filename") or img_info.get("name") or ""
    if name:
        return os.path.basename(name)
    path = img_info.get("path") or ""
    return os.path.basename(path) if path else ""


def _load_mask_as_bool(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load image mask and return boolean array (H,W). If image has alpha, use alpha>0."""
    im = Image.open(path).convert("RGBA")
    if target_size is not None and im.size != (target_size[1], target_size[0]):
        im = im.resize((target_size[1], target_size[0]), resample=Image.NEAREST)
    arr = np.array(im)  # (H,W,4)
    alpha = arr[..., 3]
    if alpha.sum() > 0:
        mask = alpha > 0
    else:
        # fallback: use luminance threshold on RGB
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
        mask = lum > 10
    return mask.astype(bool)

def _load_shape_mask(path: str) -> np.ndarray:
    """Load layout shape image and return boolean mask where pixels belong to shape.
    
    Handles both:
    - RGBA images: alpha > 0 = shape
    - Grayscale/binary masks: white (>127) = shape (e.g. _voronoi_temp.png)
    """
    im = Image.open(path)
    # prefer alpha channel
    if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
        arr = im.convert("RGBA")
        alpha = np.array(arr)[..., 3]
        mask = alpha > 0
    else:
        # Grayscale: white = foreground/shape
        gray = im.convert("L")
        a = np.array(gray)
        mask = a > 127
    return mask.astype(bool)

def _centroid_of_mask(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.5, 0.5)
    cx = xs.mean()
    cy = ys.mean()
    return (cx, cy)

# -------------------------
# Metric implementations
# -------------------------
def saliency_area(shape_mask: np.ndarray, image_saliency_masks: List[np.ndarray]) -> float:
    PX = shape_mask.sum()
    if PX == 0:
        return 0.0
    union_sal = np.zeros_like(shape_mask, dtype=bool)
    for s in image_saliency_masks:
        union_sal |= (s & shape_mask)
    return float(union_sal.sum() / PX)

def compactness(shape_mask: np.ndarray, image_masks: List[np.ndarray]) -> float:
    PX = shape_mask.sum()
    if PX == 0:
        return 0.0
    union_imgs = np.zeros_like(shape_mask, dtype=bool)
    for m in image_masks:
        union_imgs |= (m & shape_mask)
    Pw = (shape_mask & ~union_imgs).sum()
    return float(Pw / PX)

def non_overlapping_constraint(shape_mask: np.ndarray, image_masks: List[np.ndarray]) -> float:
    PX = shape_mask.sum()
    if PX == 0:
        return 0.0
    if len(image_masks) == 0:
        return 0.0
    stack = np.stack([m.astype(np.uint8) for m in image_masks], axis=0)  # (N,H,W)
    stack *= shape_mask.astype(np.uint8)
    cover_count = stack.sum(axis=0)
    pairwise = (cover_count * (cover_count - 1) // 2).astype(np.int64)
    Po = pairwise.sum()
    return float(Po / PX)

def correlation_preservation_timeline(
    image_locations: List[Tuple[float, float]],
    reference_reading_slots: List[Tuple[float, float]],
) -> float:
    """Timeline variant of Mn (no AIC categories).

    Assumes ``images`` in slicing_result.json are in chronological order.
    ``reference_reading_slots[k]`` is the normalized centroid of the k-th panel
    when panels are sorted by reading order (top-to-bottom, then left-to-right).

    M_n^time = (1/N) sum_i || L_i - R_i || — lower is better (closer to strict
    timeline = reading-flow alignment).
    """
    if not image_locations or not reference_reading_slots:
        return 0.0
    n = min(len(image_locations), len(reference_reading_slots))
    if n == 0:
        return 0.0
    return correlation_preservation(image_locations[:n], reference_reading_slots[:n])


def correlation_preservation(image_locations: List[Tuple[float, float]],
                              category_centroids: List[Tuple[float, float]]) -> float:
    if not image_locations or not category_centroids:
        return 0.0
    img_loc = np.array(image_locations, dtype=float)
    cat = np.array(category_centroids, dtype=float)
    if img_loc.shape != cat.shape:
        # fallback: if lengths differ, truncate to smallest
        n = min(len(img_loc), len(cat))
        img_loc = img_loc[:n]
        cat = cat[:n]
    diffs = np.linalg.norm(img_loc - cat, axis=1)
    return float(diffs.mean())

def saliency_loss(image_saliency_masks: List[np.ndarray]) -> float:
    if not image_saliency_masks:
        return 0.0
    total_si = sum([s.sum() for s in image_saliency_masks])
    if total_si == 0:
        return 0.0
    u = np.zeros_like(image_saliency_masks[0], dtype=bool)
    for s in image_saliency_masks:
        u |= s
    return float(1.0 - (u.sum() / total_si))

# -------------------------
# Additional metrics requested: Layout Quality (LQ) and Data Representation Error (DRE)
# -------------------------
def layout_quality(shape_mask: np.ndarray, object_masks: List[np.ndarray]) -> float:
    """Layout Quality = area( union(object_masks) ∩ shape ) / area(shape)

    Returns a float in [0,1]."""
    PX = shape_mask.sum()
    if PX == 0:
        return 0.0
    covered = np.zeros_like(shape_mask, dtype=bool)
    for m in object_masks:
        covered |= (m & shape_mask)
    return float(covered.sum() / PX)


def data_representation_error(target_sizes: Optional[List[float]], actual_masks: List[np.ndarray],
                              shape_mask: np.ndarray) -> float:
    """Compute MAE between target sizes and actual sizes.

    - target_sizes: list of desired sizes (normalized or absolute). If None -> returns NaN.
    - actual_sizes: will be computed as object area normalized by shape area.
    """
    if target_sizes is None:
        return float('nan')
    PX = float(shape_mask.sum())
    if PX == 0:
        return float('nan')
    actual = []
    for m in actual_masks:
        actual_area = float(m.sum())
        actual.append(actual_area / PX)
    tgt = np.array(target_sizes, dtype=float)
    act = np.array(actual, dtype=float)
    # Truncate or pad to smallest length
    n = min(len(tgt), len(act))
    if n == 0:
        return float('nan')
    return float(np.mean(np.abs(tgt[:n] - act[:n])))

# -------------------------
# High-level evaluator
# -------------------------
def evaluate_all(shape_mask: np.ndarray,
                 image_masks: List[np.ndarray],
                 image_saliency_masks: Optional[List[np.ndarray]],
                 image_locations: List[Tuple[float, float]],
                 category_centroids: Optional[List[Tuple[float, float]]],
                 target_sizes: Optional[List[float]] = None) -> Dict[str, float]:
    """Compute all metrics and return a dict including the two new metrics LQ and DRE.

    target_sizes: optional list of target sizes (normalized) used for DRE; if None DRE is NaN.
    """
    if image_saliency_masks is None:
        image_saliency_masks = [m.copy() for m in image_masks]
    if category_centroids is None:
        category_centroids = image_locations.copy()
    Ma = saliency_area(shape_mask, image_saliency_masks)
    Mc = compactness(shape_mask, image_masks)
    Mo = non_overlapping_constraint(shape_mask, image_masks)
    Mn = correlation_preservation(image_locations, category_centroids)
    Ms = saliency_loss(image_saliency_masks)
    LQ = layout_quality(shape_mask, image_masks)
    DRE = data_representation_error(target_sizes, image_masks, shape_mask)
    return {"Ma": Ma, "Mc": Mc, "Mo": Mo, "Mn": Mn, "Ms": Ms, "LQ": LQ, "DRE": DRE}

# -------------------------
# Pipeline integration (from run.py output)
# -------------------------
def _load_cell_masks_from_slicing(slicing_path: str, shape_h: int, shape_w: int) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[int]]:
    """Extract per-cell boolean masks from slicing_result.json polygon coords.
    
    Returns:
        cell_masks: list of (H,W) bool arrays, one per image (ordered by image index)
        cell_centroids: list of (cx_norm, cy_norm) for each image
        assigned_parts: list of part indices per image
    """
    import cv2
    with open(slicing_path, "r") as f:
        data = json.load(f)
    
    images = data.get("images", [])
    parts = data.get("parts", [])
    
    cell_masks = []
    cell_centroids = []
    assigned_parts = []
    
    for img_info in images:
        part_idx = img_info.get("assigned_part")
        assigned_parts.append(part_idx)
        
        if part_idx is None or part_idx >= len(parts):
            # No valid cell — empty mask
            cell_masks.append(np.zeros((shape_h, shape_w), dtype=bool))
            cell_centroids.append((0.5, 0.5))
            continue
        
        part = parts[part_idx]
        # Use unbuffered coords for Mo (no overlap by definition)
        # Fall back to buffered coords if unbuffered not available
        coords = part.get("coords_unbuffered", part.get("coords", []))
        
        if len(coords) < 3:
            cell_masks.append(np.zeros((shape_h, shape_w), dtype=bool))
            cell_centroids.append((0.5, 0.5))
            continue
        
        # Draw polygon mask
        pts = np.array(coords, dtype=np.float32)
        pts_int = np.round(pts).astype(np.int32)
        mask = np.zeros((shape_h, shape_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_int], 255)
        cell_masks.append(mask > 0)
        
        # Centroid (normalized)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            cell_centroids.append((float(xs.mean() / shape_w), float(ys.mean() / shape_h)))
        else:
            cell_centroids.append((0.5, 0.5))
    
    return cell_masks, cell_centroids, assigned_parts


def _extract_category_from_filename(filename: str) -> Optional[str]:
    """Extract category from ImageNet-style filename (nXXXXXXXX_YYYY.ext).
    
    Examples:
        n01531178_7126.jpg -> n01531178
        n02398521_980.jpg  -> n02398521
        frame_001.jpg      -> None (no category)
    
    Returns category ID or None if not in expected format.
    """
    import re
    # Match ImageNet synset pattern: nXXXXXXXX at start of filename
    match = re.match(r'^(n\d{8})_', filename)
    if match:
        return match.group(1)
    return None


def _compute_category_centroids(
    images_info: List[Dict],
    cell_centroids: List[Tuple[float, float]]
) -> Optional[List[Tuple[float, float]]]:
    """Compute category centroids for Mn metric (AIC dataset).
    
    For each image, finds its category from filename, then computes the
    centroid of all images in that category.
    
    Returns:
        List of (cx, cy) per image, where each is the centroid of its category.
        Returns None if no categories found (fallback to timeline mode).
    """
    if len(images_info) != len(cell_centroids):
        return None
    
    # Extract categories
    categories = []
    for img_info in images_info:
        fname = _image_info_name(img_info)
        cat = _extract_category_from_filename(fname)
        categories.append(cat)
    
    # Check if any categories found
    if all(c is None for c in categories):
        return None
    
    # Compute centroid per category
    from collections import defaultdict
    cat_locations = defaultdict(list)
    for i, cat in enumerate(categories):
        if cat is not None:
            cat_locations[cat].append(cell_centroids[i])
    
    # Compute mean centroid per category
    cat_centroids = {}
    for cat, locs in cat_locations.items():
        locs_arr = np.array(locs, dtype=float)
        cat_centroids[cat] = tuple(locs_arr.mean(axis=0))
    
    # Map each image to its category centroid
    result = []
    for i, cat in enumerate(categories):
        if cat is not None and cat in cat_centroids:
            result.append(cat_centroids[cat])
        else:
            # Fallback: use own location if no category
            result.append(cell_centroids[i])
    
    return result


def _timeline_reading_reference_centroids(
    slicing_data: Dict, shape_h: int, shape_w: int
) -> List[Tuple[float, float]]:
    """Centroids of assigned Voronoi cells sorted by reading order (cy, then cx), normalized."""
    parts = slicing_data.get("parts", [])
    images = slicing_data.get("images", [])
    if not parts or not images:
        return []

    assigned = set()
    for im in images:
        p = im.get("assigned_part")
        if p is not None and 0 <= p < len(parts):
            coords = parts[p].get("coords", [])
            if len(coords) >= 3:
                assigned.add(p)

    if not assigned:
        return []

    def norm_centroid(pidx: int) -> Tuple[float, float]:
        coords = parts[pidx].get("coords", [])
        if len(coords) < 3:
            return (0.5, 0.5)
        pts = np.array(coords, dtype=float)
        cx = float(pts[:, 0].mean() / shape_w)
        cy = float(pts[:, 1].mean() / shape_h)
        return (cx, cy)

    sorted_parts = sorted(assigned, key=lambda p: (norm_centroid(p)[1], norm_centroid(p)[0]))
    return [norm_centroid(p) for p in sorted_parts]


def _part_centroid_normalized(part: Dict, shape_h: int, shape_w: int) -> Optional[Tuple[float, float]]:
    """Normalized centroid of a part polygon; returns None if invalid."""
    coords = part.get("coords_unbuffered", part.get("coords", []))
    if len(coords) < 3:
        return None
    pts = np.array(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    cx = float(pts[:, 0].mean() / max(shape_w, 1))
    cy = float(pts[:, 1].mean() / max(shape_h, 1))
    if not np.isfinite(cx) or not np.isfinite(cy):
        return None
    return (cx, cy)


def _reading_order_ids(
    id_to_centroid: Dict[int, Optional[Tuple[float, float]]],
    ids: List[int],
) -> List[int]:
    """Mirror the layout-time row-major ordering on normalized centroids."""
    centroids = []
    invalid_ids = []
    for cid in ids:
        pt = id_to_centroid.get(cid)
        if pt is None:
            invalid_ids.append(cid)
            continue
        cx, cy = pt
        if not np.isfinite(cx) or not np.isfinite(cy):
            invalid_ids.append(cid)
            continue
        centroids.append((cid, cx, cy))

    if not centroids:
        return list(ids)

    ys = [c[2] for c in centroids]
    y_min, y_max = min(ys), max(ys)
    y_range = max(y_max - y_min, 1e-6)
    num_rows = max(3, math.ceil(math.sqrt(len(centroids))))
    band_height = y_range / num_rows

    rows = [[] for _ in range(num_rows)]
    for cid, cx, cy in centroids:
        row_idx = int((cy - y_min) / band_height)
        row_idx = min(num_rows - 1, max(0, row_idx))
        rows[row_idx].append((cid, cx, cy))

    ordered = []
    for row in rows:
        if not row:
            continue
        row_sorted = sorted(row, key=lambda item: item[1])
        ordered.extend([item[0] for item in row_sorted])

    if invalid_ids:
        ordered.extend(invalid_ids)

    if len(ordered) < len(ids):
        missing = [cid for cid in ids if cid not in ordered]
        ordered.extend(missing)

    return ordered


def zigzag_order_preservation(
    initial_centroids: List[Tuple[float, float]],
    final_part_centroids: Dict[int, Optional[Tuple[float, float]]],
    active_cell_ids: List[int],
) -> float:
    """Mean normalized rank drift of cell identities under reading order."""
    if not initial_centroids or not active_cell_ids:
        return float("nan")

    valid_ids = [cid for cid in active_cell_ids if 0 <= cid < len(initial_centroids)]
    if len(valid_ids) <= 1:
        return 0.0

    init_map = {cid: tuple(initial_centroids[cid]) for cid in valid_ids}
    final_map = {cid: final_part_centroids.get(cid) for cid in valid_ids}

    init_order = _reading_order_ids(init_map, valid_ids)
    final_order = _reading_order_ids(final_map, valid_ids)
    init_rank = {cid: rank for rank, cid in enumerate(init_order)}
    final_rank = {cid: rank for rank, cid in enumerate(final_order)}

    denom = max(len(valid_ids) - 1, 1)
    drifts = [abs(final_rank[cid] - init_rank[cid]) / denom for cid in valid_ids]
    return float(np.mean(drifts))


def _load_saliency_masks_from_isnet(isnet_masks_dir: str, images_info: List[Dict],
                                     shape_h: int, shape_w: int,
                                     cell_masks: List[np.ndarray]) -> List[np.ndarray]:
    """Load ISNet saliency masks and intersect with cell masks.
    
    For each image, loads its ISNet mask, resizes to cell bounding box,
    then intersects with the cell polygon mask.
    """
    import cv2
    saliency_masks = []
    
    for i, img_info in enumerate(images_info):
        fname = _image_info_name(img_info)
        mask_path = None
        if fname:
            stem, ext = os.path.splitext(fname)
            ext = ext.lower()
            probe_exts = ['.png', '.jpg', '.jpeg']
            if ext in probe_exts:
                probe_exts = [ext] + [e for e in probe_exts if e != ext]

            candidates = [
                os.path.join(isnet_masks_dir, fname),
            ]
            for e in probe_exts:
                candidates.append(os.path.join(isnet_masks_dir, stem + e))
                candidates.append(os.path.join(isnet_masks_dir, stem + '_mask' + e))

            for cand in candidates:
                if os.path.isfile(cand):
                    mask_path = cand
                    break
        
        if mask_path:
            sal = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if sal is not None:
                # Resize saliency to match shape dimensions
                sal_resized = cv2.resize(sal, (shape_w, shape_h), interpolation=cv2.INTER_NEAREST)
                sal_bool = (sal_resized > 127) & cell_masks[i]
                saliency_masks.append(sal_bool)
                continue
        
        # Fallback: use cell mask as saliency
        saliency_masks.append(cell_masks[i].copy())
    
    return saliency_masks


def _load_warped_saliency_masks(output_dir: str, images_info: List[Dict],
                                  shape_h: int, shape_w: int) -> Optional[List[np.ndarray]]:
    """Load actual warped saliency masks from collage assembly output.
    
    Tries to load warped saliency masks saved during collage assembly.
    These reflect the actual saliency in the final collage after warp/crop.
    
    Returns:
        List of warped saliency masks if found, None otherwise.
    """
    import cv2
    warped_sal_dir = os.path.join(output_dir, "warped_saliency_masks")
    if not os.path.isdir(warped_sal_dir):
        return None
    
    warped_masks = []
    for img_info in images_info:
        fname = _image_info_name(img_info)
        if not fname:
            return None
        
        # Try to load warped saliency mask
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        probe_exts = ['.png', '.jpg', '.jpeg']
        if ext in probe_exts:
            probe_exts = [ext] + [e for e in probe_exts if e != ext]
        candidates = [os.path.join(warped_sal_dir, fname)]
        for e in probe_exts:
            candidates.append(os.path.join(warped_sal_dir, base + e))

        mask_path = None
        for cand in candidates:
            if os.path.isfile(cand):
                mask_path = cand
                break
        if mask_path is None:
            return None
        
        sal = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if sal is None:
            return None
        
        # Resize to shape dimensions if needed
        if sal.shape != (shape_h, shape_w):
            sal = cv2.resize(sal, (shape_w, shape_h), interpolation=cv2.INTER_NEAREST)
        
        warped_masks.append(sal > 127)
    
    return warped_masks if len(warped_masks) == len(images_info) else None


def evaluate_run_output(output_dir: str, shape_path: str,
                        mask_folder: Optional[str] = None,
                        save_json: bool = True, save_csv: bool = True) -> Dict[str, float]:
    """
    Evaluate a CAST run output folder.
    
    Args:
        output_dir: Path to run output (e.g. tests/output_banana)
                    Must contain: slicing_result.json, collage.png
        shape_path: Path to shape mask (e.g. _voronoi_temp.png)
                    White=foreground, black=background
        mask_folder: Optional path to folder with per-frame saliency masks.
                     If provided, uses these instead of ISNet masks.
                     Masks should have same filenames as frames.
    """
    import cv2
    
    # 1. Load shape mask
    shape_mask = _load_shape_mask(shape_path)
    H, W = shape_mask.shape
    print(f"[Eval] Shape mask: {W}x{H}, foreground: {shape_mask.sum()/(H*W)*100:.1f}%")
    
    # 2. Load Voronoi cell masks from slicing_result.json polygon coords
    #    These are the original Voronoi cells (no buffer/dilation) → no overlap by definition
    slicing_path = os.path.join(output_dir, "slicing_result.json")
    if not os.path.isfile(slicing_path):
        raise FileNotFoundError(f"slicing_result.json not found in {output_dir}")
    
    cell_masks, cell_centroids, assigned_parts = _load_cell_masks_from_slicing(slicing_path, H, W)
    print(f"[Eval] Loaded {len(cell_masks)} cell masks from slicing_result.json")
    
    with open(slicing_path, "r") as f:
        slicing_data = json.load(f)
    images_info = slicing_data.get("images", [])
    parts_info = slicing_data.get("parts", [])

    valid_assignments = [
        p for p in assigned_parts
        if isinstance(p, int) and 0 <= p < len(parts_info)
    ]
    invalid_assignment_count = len(assigned_parts) - len(valid_assignments)
    unique_assigned_parts = len(set(valid_assignments))
    duplicate_assignments = len(valid_assignments) - unique_assigned_parts
    print(
        f"[Eval] Assignment integrity: images={len(images_info)}, parts={len(parts_info)}, "
        f"unique_assigned_parts={unique_assigned_parts}, duplicate_assignments={duplicate_assignments}, "
        f"invalid_assignments={invalid_assignment_count}"
    )
    
    # 3. Load collage alpha as coverage mask
    collage_path = os.path.join(output_dir, "collage.png")
    collage_coverage = None
    if os.path.isfile(collage_path):
        collage = cv2.imread(collage_path, cv2.IMREAD_UNCHANGED)
        if collage is not None and collage.ndim == 3 and collage.shape[2] == 4:
            alpha = collage[:, :, 3]
            if alpha.shape != (H, W):
                alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_NEAREST)
            collage_coverage = alpha > 0
            covered_pct = collage_coverage.sum() / (H * W) * 100
            print(f"[Eval] Collage coverage: {covered_pct:.1f}% of canvas")
    
    # 4. Saliency masks: prioritize warped saliency, then user-provided, then ISNet, then cell masks
    # Priority 0: Warped saliency masks (actual saliency in final collage)
    saliency_masks = _load_warped_saliency_masks(output_dir, images_info, H, W)
    if saliency_masks is not None:
        print(f"[Eval] Loaded WARPED saliency masks (actual collage saliency)")
    # Priority 1: User-provided mask folder
    elif mask_folder and os.path.isdir(mask_folder):
        saliency_masks = _load_saliency_masks_from_isnet(
            mask_folder, images_info, H, W, cell_masks)
        print(f"[Eval] Loaded saliency masks from user folder: {mask_folder}")
        print(f"[Eval] WARNING: Using original masks, not warped. Ma/Ms may not reflect actual collage.")
    else:
        # Priority 2: ISNet output from run.py/run_eval.py
        isnet_masks_dir = os.path.join(output_dir, "isnet_output", "saliency_masks")
        if not (os.path.isdir(isnet_masks_dir) and len(os.listdir(isnet_masks_dir)) > 0):
            legacy = os.path.join(output_dir, "isnet_output", "masks_filtered")
            if os.path.isdir(legacy) and len(os.listdir(legacy)) > 0:
                isnet_masks_dir = legacy
        if os.path.isdir(isnet_masks_dir) and len(os.listdir(isnet_masks_dir)) > 0:
            saliency_masks = _load_saliency_masks_from_isnet(
                isnet_masks_dir, images_info, H, W, cell_masks)
            print(f"[Eval] Loaded saliency masks from ISNet output")
            print(f"[Eval] WARNING: Using original masks, not warped. Ma/Ms may not reflect actual collage.")
        else:
            # Priority 3: Fallback to cell masks
            saliency_masks = [m.copy() for m in cell_masks]
            print(f"[Eval] No saliency masks found, using cell masks as fallback")
    
    # 5. Target sizes from slicing_result (normalized cell areas)
    shape_area = float(shape_mask.sum())
    target_sizes = None
    parts = slicing_data.get("parts", [])
    if parts:
        # Compute target size as polygon area / shape area
        target_sizes = []
        for i, img_info in enumerate(images_info):
            if i < len(cell_masks):
                target_sizes.append(float(cell_masks[i].sum()) / max(shape_area, 1))
            else:
                target_sizes.append(0.0)
    
    # 6. Try to extract category centroids from filenames (for AIC/ImageNet datasets)
    category_centroids = _compute_category_centroids(images_info, cell_centroids)
    if category_centroids is not None:
        categories = [
            _extract_category_from_filename(_image_info_name(img))
            for img in images_info
        ]
        n_cats = len({c for c in categories if c is not None})
        print(f"[Eval] Found {n_cats} categories from filenames (ImageNet format)")
    else:
        print(f"[Eval] No categories found in filenames, Mn will use timeline mode")
    
    # 6.5. Compute validity metrics (for 3-loss model evaluation)
    num_cells = len(cell_masks)
    num_invalid = sum(1 for mask in cell_masks if mask.sum() == 0)
    num_missing = sum(1 for i, mask in enumerate(cell_masks) if mask.sum() == 0 and i < len(images_info))
    
    print(
        f"[Eval] Cell-mask validity: {num_cells - num_invalid}/{num_cells} valid, "
        f"invalid={num_invalid}"
    )
    
    # 7. Compute metrics
    metrics = evaluate_all(
        shape_mask=shape_mask,
        image_masks=cell_masks,
        image_saliency_masks=saliency_masks,
        image_locations=cell_centroids,
        category_centroids=category_centroids,
        target_sizes=target_sizes
    )

    # For final reporting, make Mc/Mo reflect the FINAL collage output.
    # This avoids inflated overlap/whitespace from duplicated part assignments
    # in slicing metadata after remapping.
    if collage_coverage is not None:
        collage_mask = collage_coverage.astype(bool)
        metrics["Mc"] = compactness(shape_mask, [collage_mask])
        metrics["Mo"] = non_overlapping_constraint(shape_mask, [collage_mask])
        metrics["LQ"] = layout_quality(shape_mask, [collage_mask])
        print("[Eval] Using collage alpha for final Mc/Mo/LQ")
    
    # Add validity metrics
    metrics["num_cells"] = num_cells
    metrics["num_invalid"] = num_invalid
    metrics["num_missing"] = num_missing
    metrics["invalid_ratio"] = num_invalid / max(num_cells, 1)

    # Order-based metrics are intentionally disabled for free-assignment runs.
    # Keep the keys so existing CSV/JSON readers and ablation scripts remain compatible.
    metrics["Mn_timeline"] = float('nan')
    metrics["Mt"] = float('nan')
    print("[Eval] Order metrics disabled: free image-cell assignment has no timeline target.")

    # 7. Extra metrics from collage alpha
    if collage_coverage is not None:
        coverage_in_shape = float((collage_coverage & shape_mask).sum()) / max(shape_area, 1)
        white_space_in_shape = float((shape_mask & ~collage_coverage).sum()) / max(shape_area, 1)
        metrics["collage_coverage"] = coverage_in_shape
        metrics["white_space"] = white_space_in_shape
    
    # 8. Bbox retention from bbox_retention.json
    bbox_ret_path = os.path.join(output_dir, "bbox_retention.json")
    if os.path.isfile(bbox_ret_path):
        with open(bbox_ret_path, "r") as f:
            bbox_ret = json.load(f)
        metrics["avg_bbox_coverage"] = bbox_ret.get("avg_coverage", float('nan'))
    
    # Save outputs
    if save_json:
        outj = os.path.join(output_dir, "evaluation_metrics.json")
        with open(outj, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Eval] Saved: {outj}")
    if save_csv:
        outcsv = os.path.join(output_dir, "evaluation_metrics.csv")
        with open(outcsv, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])
        print(f"[Eval] Saved: {outcsv}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.4f}")
        else:
            print(f"  {k:25s} = {v}")
    print("="*50)
    
    return metrics

# -------------------------
# CLI runner
# -------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Evaluate CAST collage output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with ISNet masks from output_dir
  python evaluation.py --output_dir output/ --shape shape.png
  
  # Evaluate with user-provided saliency masks
  python evaluation.py --output_dir output/ --shape shape.png --mask_folder masks/
  
Required files in output_dir:
  - slicing_result.json  (cell polygons and assignments)
  - collage.png          (final collage with alpha channel)
  
Optional files (if --mask_folder not provided):
  - isnet_output/saliency_masks/*.jpg/png  (per-frame saliency masks)
        """
    )
    ap.add_argument("--output_dir", required=True,
                    help="Run output folder containing slicing_result.json and collage.png")
    ap.add_argument("--shape", required=True,
                    help="Shape mask image (white=foreground, black=background)")
    ap.add_argument("--mask_folder", default=None,
                    help="Folder with per-frame saliency masks (same filenames as frames). "
                         "If not provided, uses ISNet masks from output_dir/isnet_output/saliency_masks/")
    ap.add_argument("--no_save_csv", action="store_true",
                    help="Skip saving CSV output")
    ap.add_argument("--no_save_json", action="store_true",
                    help="Skip saving JSON output")
    args = ap.parse_args()
    
    evaluate_run_output(
        args.output_dir, args.shape,
        mask_folder=args.mask_folder,
        save_json=not args.no_save_json,
        save_csv=not args.no_save_csv
    )