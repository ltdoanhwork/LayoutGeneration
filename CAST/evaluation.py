"""
Evaluation module for CAST collage layout metrics.

Metrics:
    Ma  - Saliency Area: fraction of shape covered by salient regions
    Mc  - Compactness (white space): fraction of shape NOT covered by images
    Mo  - Non-overlapping: pairwise overlap penalty between cells
    Mn  - Correlation Preservation (AIC): mean distance image vs category centroid
    Mn_timeline - Keyframe/timeline: mean distance vs reading-order slot (no categories)
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
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

# -------------------------
# Utility IO / conversions
# -------------------------
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
        coords = part.get("coords", [])
        
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
        fname = img_info.get("filename", "")
        mask_path = os.path.join(isnet_masks_dir, fname) if fname else None
        
        if mask_path and os.path.isfile(mask_path):
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


def evaluate_run_output(output_dir: str, shape_path: str,
                        save_json: bool = True, save_csv: bool = True) -> Dict[str, float]:
    """
    Evaluate a CAST run output folder.
    
    Args:
        output_dir: Path to run output (e.g. tests/output_banana)
                    Must contain: slicing_result.json, collage.png
        shape_path: Path to shape mask (e.g. _voronoi_temp.png)
                    White=foreground, black=background
    """
    import cv2
    
    # 1. Load shape mask
    shape_mask = _load_shape_mask(shape_path)
    H, W = shape_mask.shape
    print(f"[Eval] Shape mask: {W}x{H}, foreground: {shape_mask.sum()/(H*W)*100:.1f}%")
    
    # 2. Load cell polygon masks from slicing_result.json
    slicing_path = os.path.join(output_dir, "slicing_result.json")
    if not os.path.isfile(slicing_path):
        raise FileNotFoundError(f"slicing_result.json not found in {output_dir}")
    
    cell_masks, cell_centroids, assigned_parts = _load_cell_masks_from_slicing(slicing_path, H, W)
    print(f"[Eval] Loaded {len(cell_masks)} cell masks from slicing_result.json")
    
    # 3. Load collage alpha as coverage mask
    collage_path = os.path.join(output_dir, "collage.png")
    collage_coverage = None
    if os.path.isfile(collage_path):
        collage = cv2.imread(collage_path, cv2.IMREAD_UNCHANGED)
        if collage is not None and collage.shape[2] == 4:
            alpha = collage[:, :, 3]
            if alpha.shape != (H, W):
                alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_NEAREST)
            collage_coverage = alpha > 0
            covered_pct = collage_coverage.sum() / (H * W) * 100
            print(f"[Eval] Collage coverage: {covered_pct:.1f}% of canvas")
    
    # 4. Saliency masks: try ISNet masks, fallback to cell masks
    with open(slicing_path, "r") as f:
        slicing_data = json.load(f)
    images_info = slicing_data.get("images", [])
    
    isnet_masks_dir = os.path.join(output_dir, "isnet_output", "saliency_masks")
    if not (os.path.isdir(isnet_masks_dir) and len(os.listdir(isnet_masks_dir)) > 0):
        legacy = os.path.join(output_dir, "isnet_output", "masks_filtered")
        if os.path.isdir(legacy) and len(os.listdir(legacy)) > 0:
            isnet_masks_dir = legacy
    if os.path.isdir(isnet_masks_dir) and len(os.listdir(isnet_masks_dir)) > 0:
        saliency_masks = _load_saliency_masks_from_isnet(
            isnet_masks_dir, images_info, H, W, cell_masks)
        print(f"[Eval] Loaded saliency masks from ISNet")
    else:
        saliency_masks = [m.copy() for m in cell_masks]
        print(f"[Eval] No ISNet masks found, using cell masks as saliency")
    
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
    
    # 6. Compute metrics
    metrics = evaluate_all(
        shape_mask=shape_mask,
        image_masks=cell_masks,
        image_saliency_masks=saliency_masks,
        image_locations=cell_centroids,
        category_centroids=None,
        target_sizes=target_sizes
    )

    reading_refs = _timeline_reading_reference_centroids(slicing_data, H, W)
    metrics["Mn_timeline"] = correlation_preservation_timeline(cell_centroids, reading_refs)
    if len(cell_centroids) != len(reading_refs):
        print(
            f"[Eval] Mn_timeline: len(images)={len(cell_centroids)} vs "
            f"reading_slots={len(reading_refs)} — using min length for mean"
        )

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
    ap = argparse.ArgumentParser(description="Evaluate CAST collage output")
    ap.add_argument("--output_dir", required=True,
                    help="Run output folder (e.g. tests/output_banana)")
    ap.add_argument("--shape", required=True,
                    help="Shape mask image (e.g. tests/output_banana/_voronoi_temp.png)")
    ap.add_argument("--no_save_csv", action="store_true")
    ap.add_argument("--no_save_json", action="store_true")
    args = ap.parse_args()
    
    evaluate_run_output(
        args.output_dir, args.shape,
        save_json=not args.no_save_json,
        save_csv=not args.no_save_csv
    )