"""COLLA Layout Pipeline — Voronoi (EVAL MODE)

Like run.py but skips U2-Net and ISNet entirely.
Inputs are pre-computed:
  - A binary shape mask (white=foreground)
  - A folder of frame images
  - A folder of per-frame binary masks (white=salient region, same filenames as frames)

The per-frame masks are used to:
  1. Extract bounding boxes (replaces ISNet detection)
  2. Provide saliency masks for content-aware warp
  3. Feed evaluation.py (copied to isnet_output/saliency_masks/)

All other config is identical to run.py (FROZEN).

Usage:
    python run_eval.py <shape_mask> <frame_folder> <mask_folder> <output_dir> <scale> [flags]

Optional flags:
    --no-filter-frames-by-isnet / --filter-frames-by-isnet
    --ablation=MODE  (MODE: full, wo_cap, wo_cvt, wo_fea)

Example:
        python run_eval.py masks/shape.png frames/ frames_masks/ output/ 2 --ablation=wo_cvt
"""
import matplotlib
matplotlib.use('Agg')
import shape_decomposition as sd
import sas_optimization as so
import collage_assembly as ca
import sys
import cv2
import os
import json
import numpy as np
import time

# ==============================================================================
# FROZEN CONFIGURATION — IDENTICAL TO run.py
# ==============================================================================
CFG_ENABLE_DEBUG = False
CFG_USE_OBJECT_DETECTION = False
CFG_DETECTION_THRESHOLD = 0.1
CFG_USE_VORONOI_LAYOUT = True
CFG_USE_TIMELINE_ORDER = False
CFG_SKIP_WARP = False
CFG_NUM_ITERATIONS = 100
CFG_DEBUG_EVERY = 10
CFG_FILTER_NO_DETECTION = False
CFG_FILTER_FRAMES_BY_ISNET = False
CFG_PROB_CSV_PATH = None
CFG_JSON_DIR = None

# Mask-based bbox filtering thresholds (same as ISNet filtering in run.py)
CFG_MIN_BBOX_RATIO = 0.1
CFG_MIN_BBOX_COUNT = 1


def _bbox_from_mask(mask_gray):
    """Extract bounding box from a binary mask (white=foreground).
    
    Returns:
        bbox: [x1, y1, x2, y2] or None if mask is empty
        pixel_area: number of foreground pixels
    """
    if mask_gray is None:
        return None, 0
    
    # Threshold
    _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    
    # Union bounding box of all contours
    x1, y1 = mask_gray.shape[1], mask_gray.shape[0]
    x2, y2 = 0, 0
    pixel_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        pixel_area += area
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        x1 = min(x1, rx)
        y1 = min(y1, ry)
        x2 = max(x2, rx + rw)
        y2 = max(y2, ry + rh)
    
    if x2 <= x1 or y2 <= y1:
        return None, 0
    
    return [int(x1), int(y1), int(x2), int(y2)], int(pixel_area)


def _build_summary_from_masks(frame_folder, mask_folder):
    """Build a summary dict (same format as ISNet summary.json) from per-frame masks.
    
    Matches frames to masks by filename (stem must match).
    
    Returns:
        summary: dict with 'frames' list, same schema as ISNet output
    """
    # Collect frame files
    frame_exts = ('.jpg', '.jpeg', '.png')
    frame_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(frame_exts)
    ])
    
    # Collect mask files (index by stem)
    mask_files = {}
    if os.path.isdir(mask_folder):
        for f in os.listdir(mask_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                stem = os.path.splitext(f)[0]
                mask_files[stem] = f
    
    summary = {
        'frames': [],
        'total_objects': 0,
        'params': {
            'source': 'precomputed_masks',
            'mask_folder': mask_folder,
        }
    }
    
    no_mask_count = 0
    for frame_name in frame_files:
        stem = os.path.splitext(frame_name)[0]
        
        # Read frame to get dimensions
        frame_path = os.path.join(frame_folder, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"  [WARN] Cannot read frame: {frame_name}, skipping")
            continue
        img_h, img_w = img.shape[:2]
        
        # Find matching mask
        mask_name = mask_files.get(stem)
        objects = []
        
        if mask_name:
            mask_path = os.path.join(mask_folder, mask_name)
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is not None:
                # Resize mask to frame size if needed
                if mask_gray.shape[:2] != (img_h, img_w):
                    mask_gray = cv2.resize(mask_gray, (img_w, img_h),
                                           interpolation=cv2.INTER_NEAREST)
                
                bbox, pixel_area = _bbox_from_mask(mask_gray)
                if bbox is not None:
                    bbox_w = bbox[2] - bbox[0]
                    bbox_h = bbox[3] - bbox[1]
                    objects.append({
                        'bbox': bbox,
                        'pixel_area': pixel_area,
                        'bbox_area': bbox_w * bbox_h,
                        'size': [bbox_w, bbox_h],
                        'confidence': 1.0,
                    })
        else:
            no_mask_count += 1
        
        summary['frames'].append({
            'name': frame_name,
            'num_objects': len(objects),
            'frame_size': [img_w, img_h],
            'objects': objects,
        })
        summary['total_objects'] += len(objects)
    
    n_frames = len(summary['frames'])
    summary['avg_objects_per_frame'] = (
        summary['total_objects'] / n_frames if n_frames > 0 else 0
    )
    
    if no_mask_count > 0:
        print(f"  [WARN] {no_mask_count}/{n_frames} frames have no matching mask "
              f"(will use full-image bbox)")
    
    return summary


def _sync_saliency_masks(frame_folder, mask_folder, kept_frames, out_dir):
    """Copy/resize per-frame masks into saliency directory for warp + evaluation.
    
    For frames without a mask, generate a bbox-based saliency mask (same as run.py).
    """
    os.makedirs(out_dir, exist_ok=True)
    
    mask_files = {}
    if os.path.isdir(mask_folder):
        for f in os.listdir(mask_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                stem = os.path.splitext(f)[0]
                mask_files[stem] = f
    
    for fi in kept_frames:
        fname = fi.get('name')
        if not fname:
            continue
        
        stem = os.path.splitext(fname)[0]
        fs = fi.get('frame_size') or fi.get('img_size', [960, 540])
        img_w, img_h = int(fs[0]), int(fs[1])
        
        mask_name = mask_files.get(stem)
        if mask_name:
            # Load user mask
            src_path = os.path.join(mask_folder, mask_name)
            mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Resize to frame size if needed
                if mask.shape[:2] != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h),
                                      interpolation=cv2.INTER_NEAREST)
                # Ensure binary
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join(out_dir, fname), mask)
                continue
        
        # Fallback: bbox-based saliency (for frames without user-provided mask)
        # Note: In eval mode, this should rarely happen since user provides all masks
        objs = fi.get('objects', [])
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pad_ratio = 0.10
        for obj in objs:
            x1, y1, x2, y2 = obj['bbox']
            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * pad_ratio), int(bh * pad_ratio)
            x1p = max(0, x1 - px)
            y1p = max(0, y1 - py)
            x2p = min(img_w, x2 + px)
            y2p = min(img_h, y2 + py)
            mask[y1p:y2p, x1p:x2p] = 255
        cv2.imwrite(os.path.join(out_dir, fname), mask)


class _TeeIO:
    """Duplicate writes to terminal and a log file (stdout/stderr)."""

    __slots__ = ('_a', '_b')

    def __init__(self, a, b):
        self._a, self._b = a, b

    def write(self, s):
        if not s:
            return
        self._a.write(s)
        self._b.write(s)

    def flush(self):
        self._a.flush()
        self._b.flush()

    def fileno(self):
        return self._a.fileno()

    def isatty(self):
        return getattr(self._a, 'isatty', lambda: False)()


if __name__ == '__main__':
    start = time.time()

    # ========================================================================
    # ARGUMENT PARSING
    # ========================================================================
    if len(sys.argv) < 6:
        print("Usage: python run_eval.py <shape_mask> <frame_folder> <mask_folder> <output_dir> <scale> [flags]")
        print()
        print("  shape_mask   : Binary mask PNG (white=foreground shape)")
        print("  frame_folder : Folder of frame images (jpg/png)")
        print("  mask_folder  : Folder of per-frame binary masks (same filenames)")
        print("  output_dir   : Output directory")
        print("  scale        : Scaling factor (int, e.g. 2)")
        print()
        print("Optional flags:")
        print("  --no-filter-frames-by-isnet / --filter-frames-by-isnet")
        print("  --ablation=MODE  (MODE: full, wo_cap, wo_cvt, wo_fea)")
        print()
        print("All other parameters are FROZEN (same as run.py).")
        sys.exit(1)

    input_shape = sys.argv[1]
    input_image_collection_folder = sys.argv[2]
    input_mask_folder = sys.argv[3]
    output_dir = sys.argv[4]
    try:
        scaling_factor = int(sys.argv[5])
    except ValueError:
        scaling_factor = 2

    # Parse optional flags (parity with run.py)
    filter_frames_by_isnet = CFG_FILTER_FRAMES_BY_ISNET
    ablation_mode = None
    for arg in sys.argv[6:]:
        if arg == '--no-filter-frames-by-isnet':
            filter_frames_by_isnet = False
        elif arg == '--filter-frames-by-isnet':
            filter_frames_by_isnet = True
        elif arg.startswith('--ablation='):
            ablation_mode = arg.split('=', 1)[1]
            valid_ablation_modes = [
                'full', 'wo_cap', 'wo_cvt', 'wo_fea',
            ]
            if ablation_mode not in valid_ablation_modes:
                print(f"[ERROR] Invalid ablation mode: {ablation_mode}")
                print("Valid modes: " + ", ".join(valid_ablation_modes))
                sys.exit(1)
            if ablation_mode == 'full':
                ablation_mode = None

    # Load frozen config
    enable_debug = CFG_ENABLE_DEBUG
    use_object_detection = CFG_USE_OBJECT_DETECTION
    detection_threshold = CFG_DETECTION_THRESHOLD
    use_voronoi_layout = CFG_USE_VORONOI_LAYOUT
    use_timeline_order = CFG_USE_TIMELINE_ORDER
    skip_warp = CFG_SKIP_WARP
    num_iterations = CFG_NUM_ITERATIONS
    debug_every = CFG_DEBUG_EVERY
    filter_no_detection = CFG_FILTER_NO_DETECTION
    min_bbox_ratio = CFG_MIN_BBOX_RATIO
    min_bbox_count = CFG_MIN_BBOX_COUNT
    prob_csv_path = CFG_PROB_CSV_PATH
    json_dir = CFG_JSON_DIR

    os.makedirs(output_dir, exist_ok=True)

    _log_path = os.path.join(output_dir, 'run_eval.log')
    _log_fp = open(_log_path, 'w', encoding='utf-8')
    _stdout_bak, _stderr_bak = sys.stdout, sys.stderr
    sys.stdout = _TeeIO(_stdout_bak, _log_fp)
    sys.stderr = _TeeIO(_stderr_bak, _log_fp)
    try:
        import datetime as _dt
        print(f"[LOG] Mirror log: {_log_path}")
        print(f"[LOG] Started: {_dt.datetime.now().isoformat(timespec='seconds')}")
        print(f"[LOG] argv: {sys.argv}")
        print("=" * 60)
        print("COLLA LAYOUT PIPELINE — EVAL MODE (no U2-Net, no ISNet)")
        print("=" * 60)
        print(f"[CONFIG] Shape mask       : {input_shape}")
        print(f"[CONFIG] Frame folder     : {input_image_collection_folder}")
        print(f"[CONFIG] Mask folder      : {input_mask_folder}")
        print(f"[CONFIG] Output dir       : {output_dir}")
        print(f"[CONFIG] Scaling factor   : {scaling_factor}")
        print(f"[CONFIG] Voronoi layout   : {use_voronoi_layout} (FROZEN)")
        print(f"[CONFIG] Timeline order   : {use_timeline_order} (FROZEN)")
        print(f"[CONFIG] Num iterations   : {num_iterations} (FROZEN)")
        print(f"[CONFIG] Skip warp        : {skip_warp} (FROZEN)")
        print(f"[CONFIG] Saliency source  : pre-computed masks (EVAL MODE)")
        print(f"[CONFIG] Min bbox ratio   : {min_bbox_ratio} (FROZEN)")
        print(f"[CONFIG] Min bbox count   : {min_bbox_count} (FROZEN)")
        print(f"[CONFIG] Filter frames by bbox rules: "
              f"{'ON' if filter_frames_by_isnet else 'OFF'}")
        if ablation_mode:
            print(f"[CONFIG] Ablation mode    : {ablation_mode} (EXPERIMENT)")
        else:
            print(f"[CONFIG] Ablation mode    : full (all losses enabled)")
        print("=" * 60)

        step_times = {}

        # ========================================================================
        # STEP 0: Load shape mask directly (no U2-Net)
        # ========================================================================
        t_step = time.time()
        shape_mask_path = os.path.join(output_dir, "shape_mask_refined.png")

        print(f"\n[STEP 0] Loading shape mask (EVAL: no U2-Net): {input_shape}")
        raw = cv2.imread(input_shape, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot load shape mask: {input_shape}")
        if raw.ndim == 3:
            if raw.shape[2] == 4:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
            else:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        mask_refined = np.asarray(raw, dtype=np.uint8)
        if mask_refined.max() <= 1:
            mask_refined = (mask_refined * 255).astype(np.uint8)
        _, mask_bin = cv2.threshold(mask_refined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_refined = mask_bin
        print(f"  Mask size: {mask_refined.shape[1]}x{mask_refined.shape[0]}")
        cv2.imwrite(shape_mask_path, mask_refined)
        print(f"  Saved shape mask to: {shape_mask_path}")

        step_times['Step 0 (mask)'] = time.time() - t_step
        t_step = time.time()

        # ========================================================================
        # STEP 0.5: Build summary from pre-computed masks (replaces ISNet)
        # ========================================================================
        print(f"\n[STEP 0.5] Building summary from pre-computed masks (no ISNet)")
        print(f"  Frame folder: {input_image_collection_folder}")
        print(f"  Mask folder : {input_mask_folder}")

        summary = _build_summary_from_masks(
            input_image_collection_folder, input_mask_folder
        )

        # Save summary.json (same location as ISNet would)
        isnet_output_dir = os.path.join(output_dir, 'isnet_output')
        isnet_saliency_dir = os.path.join(isnet_output_dir, 'saliency_masks')
        isnet_summary = os.path.join(isnet_output_dir, 'summary.json')
        filtered_summary = os.path.join(isnet_output_dir, 'filtered_summary.json')
        os.makedirs(isnet_output_dir, exist_ok=True)

        with open(isnet_summary, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Built summary: {len(summary['frames'])} frames, "
              f"{summary['total_objects']} objects")

        # Frame filtering (same logic as run.py)
        if filter_frames_by_isnet:
            print(f"\n[STEP 0.5b] Filtering frames "
                  f"(min_bbox_count={min_bbox_count}, min_bbox_ratio={min_bbox_ratio})...")

            kept, dropped = [], []
            for frame_info in summary['frames']:
                fs = frame_info.get('frame_size', [960, 540])
                img_w, img_h = fs[0], fs[1]
                img_area = img_w * img_h
                objects = frame_info.get('objects', [])

                valid_objects = [
                    o for o in objects
                    if min_bbox_ratio * img_area <= o.get('bbox_area', 0)
                ]

                if len(valid_objects) >= min_bbox_count:
                    kept.append({**frame_info,
                                 'objects': valid_objects,
                                 'num_objects': len(valid_objects)})
                else:
                    print(f"    [DROP] {frame_info.get('name', '<unknown>')}: "
                          f"raw={len(objects)} valid={len(valid_objects)}")
                    dropped.append(frame_info['name'])

            print(f"  Kept   : {len(kept)} frames")
            print(f"  Dropped: {len(dropped)} frames")
            if dropped:
                print(f"  Dropped frames: {dropped[:10]}"
                      + (" ..." if len(dropped) > 10 else ""))
        else:
            print("\n[STEP 0.5b] Filter frames by bbox rules: OFF — keeping all frames")
            kept = list(summary['frames'])
            dropped = []

        filtered = {
            'frames': kept,
            'total_objects': sum(len(fi['objects']) for fi in kept),
            'params': summary.get('params', {}),
            'filter': {
                'enabled': filter_frames_by_isnet,
                'min_bbox_ratio': min_bbox_ratio,
                'min_bbox_count': min_bbox_count,
                'kept': len(kept),
                'dropped': len(dropped),
            }
        }
        with open(filtered_summary, 'w') as f:
            json.dump(filtered, f, indent=2)
        print(f"  Saved filtered summary: {filtered_summary}")

        if len(kept) == 0:
            print("[ERROR] No frames left after filtering!")
            sys.exit(1)

        # Sync saliency masks for warp + evaluation
        _sync_saliency_masks(
            input_image_collection_folder, input_mask_folder,
            kept, isnet_saliency_dir
        )
        warp_mask_dir = isnet_saliency_dir
        summary_json_path = filtered_summary
        print(f"  → saliency_masks/ : {isnet_saliency_dir} ({len(kept)} frames)")
        print(f"  → summary-json    : {summary_json_path}")

        step_times['Step 0.5 (masks)'] = time.time() - t_step
        t_step = time.time()

        # ========================================================================
        # STEP 1: Shape Decomposition
        # ========================================================================
        print(f"\n[STEP 1] Shape decomposition")
        sd.generate_cuts(shape_mask_path, output_dir)

        # ========================================================================
        # STEP 2: Voronoi Layout Optimization
        # ========================================================================
        step_times['Step 1 (decomp)'] = time.time() - t_step
        t_step = time.time()
        use_timeline_order = False  # Free image-cell assignment; no timeline/order constraint
        print(f"\n[STEP 2] Voronoi layout optimization ({num_iterations} iters)")
        so.optimization(
            shape_mask_path,
            None,
            output_dir,
            image_folder=input_image_collection_folder,
            use_object_detection=use_object_detection,
            detection_threshold=detection_threshold,
            use_voronoi_layout=use_voronoi_layout,
            use_timeline_order=use_timeline_order,
            num_iterations=num_iterations,
            prob_csv_path=prob_csv_path,
            json_dir=json_dir,
            filter_no_detection=filter_no_detection,
            debug_every=debug_every,
            summary_json=summary_json_path,
            allow_empty_detection=True,
            ablation_mode=ablation_mode,
        )

        # ========================================================================
        # STEP 3: Collage Assembly & Rendering
        # ========================================================================
        step_times['Step 2 (voronoi)'] = time.time() - t_step
        t_step = time.time()
        print(f"\n[STEP 3] Collage assembly (warp_mask_dir={warp_mask_dir})")
        ca.render_collage(
            input_image_collection_folder,
            output_dir,
            scaling_factor,
            enable_debug=enable_debug,
            skip_warp=skip_warp,
            warp_mask_dir=warp_mask_dir,
        )

        # ========================================================================
        # STEP 4: Pipeline visualization
        # ========================================================================
        print(f"\n[STEP 4] Generating pipeline visualization...")
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 4, figsize=(24, 6))

            # Panel 1: Input shape mask
            mask_img = cv2.imread(input_shape, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                axes[0].imshow(mask_img, cmap='gray')
            axes[0].set_title('Input (shape mask)', fontsize=14)
            axes[0].axis('off')

            # Panel 2: Refined shape mask
            mask_refined_img = cv2.imread(shape_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_refined_img is not None:
                axes[1].imshow(mask_refined_img, cmap='gray')
            axes[1].set_title('Shape mask (refined)', fontsize=14)
            axes[1].axis('off')

            # Panel 3: Voronoi layout
            voronoi_debug_candidates = [
                os.path.join(output_dir, 'voronoi_debug_3_cells_after_opt.png'),
                os.path.join(output_dir, 'voronoi_debug_3_cells.png'),
                os.path.join(output_dir, 'voronoi_debug_2_cells_before_opt.png'),
                os.path.join(output_dir, '_voronoi_temp.png'),
            ]
            slicing_path = os.path.join(output_dir, 'slicing_result.json')
            if os.path.isfile(slicing_path):
                with open(slicing_path, 'r') as f:
                    slicing = json.load(f)

                has_voronoi_debug = False
                layout_vis = None
                for cand in voronoi_debug_candidates:
                    if os.path.isfile(cand):
                        layout_vis = cv2.imread(cand)
                        if layout_vis is not None:
                            has_voronoi_debug = True
                            break
                if not has_voronoi_debug and mask_refined_img is not None:
                    layout_vis = np.zeros((mask_refined_img.shape[0], mask_refined_img.shape[1], 3), dtype=np.uint8)
                elif not has_voronoi_debug:
                    layout_vis = np.zeros((1024, 1024, 3), dtype=np.uint8)

                if layout_vis is not None:
                    parts = slicing.get('parts', [])
                    images = slicing.get('images', [])

                    part_to_timeline = {}
                    for img_idx, img_info in enumerate(images):
                        part_idx = img_info.get('assigned_part')
                        if isinstance(part_idx, int):
                            part_to_timeline[part_idx] = img_idx

                    def _gradient_bgr(order_idx, total_items):
                        anchors_rgb = np.array([
                            [25.0, 32.0, 72.0],
                            [32.0, 94.0, 166.0],
                            [46.0, 154.0, 145.0],
                            [170.0, 190.0, 110.0],
                            [240.0, 180.0, 70.0],
                            [203.0, 83.0, 42.0],
                        ], dtype=np.float32)
                        if total_items <= 1:
                            t = 0.5
                        else:
                            t = float(order_idx) / float(max(1, total_items - 1))
                        t = 0.08 + 0.84 * t
                        t = t ** 0.92
                        pos = t * (len(anchors_rgb) - 1)
                        lo = int(np.floor(pos))
                        hi = min(lo + 1, len(anchors_rgb) - 1)
                        a = pos - lo
                        rgb = (1.0 - a) * anchors_rgb[lo] + a * anchors_rgb[hi]
                        luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                        rgb = 0.88 * rgb + 0.12 * luma
                        return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

                    base_layout_vis = layout_vis.copy() if has_voronoi_debug else np.zeros_like(layout_vis)
                    draw_items = []
                    for i, part in enumerate(parts):
                        coords = np.array(part['coords'], dtype=np.int32)
                        timeline_idx = part_to_timeline.get(i, i)
                        if not has_voronoi_debug:
                            color = _gradient_bgr(timeline_idx, max(len(images), len(parts)))
                            cv2.fillPoly(base_layout_vis, [coords], color)
                        m = cv2.moments(coords)
                        if m['m00'] != 0:
                            cx = int(m['m10'] / m['m00'])
                            cy = int(m['m01'] / m['m00'])
                        else:
                            cx, cy = int(coords[:, 0].mean()), int(coords[:, 1].mean())
                        draw_items.append((coords, i, cx, cy))

                    layout_vis = base_layout_vis.copy()

                    # Fill cracks (same as run.py)
                    if mask_refined_img is not None:
                        if (mask_refined_img.shape[0] != layout_vis.shape[0]
                                or mask_refined_img.shape[1] != layout_vis.shape[1]):
                            shape_mask_vis = cv2.resize(
                                mask_refined_img,
                                (layout_vis.shape[1], layout_vis.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        else:
                            shape_mask_vis = mask_refined_img

                        in_shape = shape_mask_vis > 127
                        is_black = np.all(layout_vis < 10, axis=2)
                        is_white = np.all(layout_vis > 245, axis=2)
                        is_hole = is_black | is_white
                        seed_mask = in_shape & (~is_hole)
                        gap_mask = in_shape & is_hole
                        if np.any(gap_mask) and np.any(seed_mask):
                            src = np.where(seed_mask, 0, 1).astype(np.uint8)
                            _, labels = cv2.distanceTransformWithLabels(
                                src, cv2.DIST_L2, 5,
                                labelType=cv2.DIST_LABEL_PIXEL,
                            )
                            seed_y, seed_x = np.where(seed_mask)
                            nearest_idx = labels[gap_mask] - 1
                            nearest_idx = np.clip(nearest_idx, 0, len(seed_y) - 1)
                            layout_vis[gap_mask] = layout_vis[
                                seed_y[nearest_idx], seed_x[nearest_idx],
                            ]

                    for coords, cell_idx, cx, cy in draw_items:
                        if not has_voronoi_debug:
                            cv2.polylines(layout_vis, [coords], True, (255, 255, 255), 1)
                        label = str(cell_idx + 1)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                        tx, ty = int(cx - tw / 2), int(cy + th / 2)
                        cv2.putText(layout_vis, label, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(layout_vis, label, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                    axes[2].imshow(cv2.cvtColor(layout_vis, cv2.COLOR_BGR2RGB))
            axes[2].set_title('Voronoi Layout', fontsize=14)
            axes[2].axis('off')

            # Panel 4: Final collage
            collage_path = os.path.join(output_dir, 'collage.png')
            if os.path.isfile(collage_path):
                col_img = cv2.imread(collage_path, cv2.IMREAD_UNCHANGED)
                if col_img is not None:
                    if col_img.ndim == 3 and col_img.shape[2] == 4:
                        col_img = cv2.cvtColor(col_img, cv2.COLOR_BGRA2RGBA)
                    else:
                        col_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)
                    axes[3].imshow(col_img)
            axes[3].set_title('Final Collage', fontsize=14)
            axes[3].axis('off')

            plt.tight_layout()
            viz_path = os.path.join(output_dir, 'pipeline_visualization.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {viz_path}")
        except Exception as e:
            print(f"  [WARN] Visualization failed: {e}")

        step_times['Step 3 (warp)'] = time.time() - t_step

        # ========================================================================
        # STEP 5: Auto-evaluation
        # ========================================================================
        t_step = time.time()
        print(f"\n[STEP 5] Running evaluation...")
        try:
            from evaluation import evaluate_run_output
            metrics = evaluate_run_output(
                output_dir,
                shape_mask_path,
                mask_folder=isnet_saliency_dir,
            )
        except Exception as e:
            print(f"  [WARN] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        step_times['Step 5 (eval)'] = time.time() - t_step

        print("\n[DONE] Eval pipeline completed successfully!")
        total = time.time() - start
        print(f"Total time: {total:.2f}s")
        print("  Per-step breakdown:")
        for name, dt in step_times.items():
            pct = dt / total * 100 if total > 0 else 0
            print(f"    {name}: {dt:.1f}s ({pct:.0f}%)")
    finally:
        sys.stdout = _stdout_bak
        sys.stderr = _stderr_bak
        try:
            _log_fp.flush()
        except Exception:
            pass
        _log_fp.close()
