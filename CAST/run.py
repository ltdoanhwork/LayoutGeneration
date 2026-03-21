"""COLLA Layout Pipeline — Voronoi (v4 HARDENED)

This is a PRODUCTION-HARDENED version with all parameters fixed.
Do NOT modify parameters here — they are frozen.

Usage:
    python run.py <shape_image_or_mask> <image_folder> <output_dir> <scale> [flags]

First positional arg:
  Default: RGB image → U2-Net produces the layout shape mask (swap model in utils/get_mask if needed).
  With --shape-is-mask: path to a grayscale/BGR mask PNG (foreground white/255) → skips U2-Net.

Optional flags:
  --shape-is-mask / --shape-is-image
  --no-filter-frames-by-isnet / --filter-frames-by-isnet

Example:
    python run.py inputs/shape.png /path/to/keyframes output_dir 2
    python run.py inputs/my_mask.png /path/to/keyframes output_dir 2 --shape-is-mask
"""
import matplotlib
matplotlib.use('Agg')
import shape_decomposition as sd
import sas_optimization as so
import collage_assembly as ca
from utils.get_mask import predict_mask, preprocess_image, refine_mask, extract_object
import sys
import cv2
import os
import json
import copy
import numpy as np
import time

try:
    from isnet_detector import SimpleISNetDetector
    HAS_ISNET = True
except Exception as e:
    HAS_ISNET = False
    print(f"[WARN] Could not import isnet_detector: {type(e).__name__}: {e}")


# ==============================================================================
# FROZEN CONFIGURATION — DO NOT MODIFY
# ==============================================================================
# These parameters are frozen for reproducible production runs.
# Edit only if you know exactly what you're doing.

CFG_ENABLE_DEBUG = False              # Was: --debug
CFG_USE_OBJECT_DETECTION = False      # Was: --object-detection
CFG_DETECTION_THRESHOLD = 0.22        # Was: --det-thresh=
CFG_USE_VORONOI_LAYOUT = True         # Was: --voronoi-layout (forced True)
CFG_USE_TIMELINE_ORDER = True         # Was: --timeline-order (forced True)
CFG_SKIP_WARP = False                 # Was: --no-warp
CFG_NUM_ITERATIONS = 400              # Was: --num-iterations=
CFG_DEBUG_EVERY = 0                   # Was: --debug-every=
CFG_FILTER_NO_DETECTION = False       # Was: --filter-no-detection

# First positional arg: False = RGB → U2-Net mask; True = mask PNG, skip U2-Net
CFG_SHAPE_INPUT_IS_MASK = False

# ISNet detection parameters (used when --run-isnet was enabled)
CFG_RUN_ISNET = True                  # Always run ISNet detection
CFG_ISNET_THRESHOLD = 0.3
CFG_ISNET_MIN_AREA = 500
CFG_MIN_BBOX_RATIO = 0.005             # min bbox_area / img_area
CFG_MIN_BBOX_COUNT = 0                # min detections per frame
CFG_FILTER_FRAMES_BY_ISNET = True     # Step 0.5b; override with argv

# ISNet weights path (relative to CAST root)
CFG_ISNET_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'isnet-detector', 'weights', 'isnetis.ckpt'
)

# Optional paths (set to None if not used)
CFG_PROB_CSV_PATH = None              # Was: --prob-csv=
CFG_JSON_DIR = None                   # Was: --json-dir=
# summary_json and warp_mask_dir are auto-set from ISNet output


def _bbox_saliency_mask(img_h, img_w, objects):
    """Create a saliency mask from ISNet bounding boxes.
    Each bbox region is set to 255 (high saliency); the rest is 0.
    Boxes are slightly padded (10%) to give the warp some context.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pad_ratio = 0.10
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad_ratio), int(bh * pad_ratio)
        x1p = max(0, x1 - px)
        y1p = max(0, y1 - py)
        x2p = min(img_w, x2 + px)
        y2p = min(img_h, y2 + py)
        mask[y1p:y2p, x1p:x2p] = 255
    return mask


def _draw_bbox_detection_vis(img_rgb, objects):
    """BGR image with green rectangles for each detection (for inspection only)."""
    vis_bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    for o in objects:
        x1, y1, x2, y2 = [int(v) for v in o['bbox']]
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis_bgr


def _sync_saliency_masks_for_kept_frames(kept, out_dir):
    """Persist per-frame bbox saliency (grayscale) for evaluation and warp; align with filtered_summary."""
    os.makedirs(out_dir, exist_ok=True)
    keep_names = {fi['name'] for fi in kept if fi.get('name')}
    for fn in os.listdir(out_dir):
        low = fn.lower()
        if low.endswith(('.png', '.jpg', '.jpeg')) and fn not in keep_names:
            try:
                os.remove(os.path.join(out_dir, fn))
            except OSError:
                pass
    for fi in kept:
        fname = fi.get('name')
        if not fname:
            continue
        fs = fi.get('frame_size') or fi.get('img_size', [960, 540])
        img_w, img_h = int(fs[0]), int(fs[1])
        objs = [{'bbox': o['bbox']} for o in fi.get('objects', [])]
        mask = _bbox_saliency_mask(img_h, img_w, objs)
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
    # ARGUMENT PARSING — MINIMAL (positional only)
    # ========================================================================
    if len(sys.argv) < 5:
        print("Usage: python run.py <shape_image_or_mask> <image_folder> <output_dir> <scale> [flags]")
        print("  --shape-is-mask     arg1 is a mask PNG (skip U2-Net)")
        print("  --shape-is-image    arg1 is RGB for U2-Net (default)")
        print("  --no-filter-frames-by-isnet / --filter-frames-by-isnet")
        print("\nSee CFG_* constants at top of run.py.")
        sys.exit(1)

    input_shape = sys.argv[1]
    input_image_collection_folder = sys.argv[2]
    output_dir = sys.argv[3]
    try:
        scaling_factor = int(sys.argv[4])
    except ValueError:
        scaling_factor = 2

    filter_frames_by_isnet = CFG_FILTER_FRAMES_BY_ISNET
    shape_input_is_mask = CFG_SHAPE_INPUT_IS_MASK
    for arg in sys.argv[5:]:
        if arg == '--no-filter-frames-by-isnet':
            filter_frames_by_isnet = False
        elif arg == '--filter-frames-by-isnet':
            filter_frames_by_isnet = True
        elif arg in ('--shape-is-mask', '--direct-shape-mask'):
            shape_input_is_mask = True
        elif arg in ('--shape-is-image', '--u2net-shape'):
            shape_input_is_mask = False

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
    run_isnet = CFG_RUN_ISNET
    isnet_threshold = CFG_ISNET_THRESHOLD
    isnet_min_area = CFG_ISNET_MIN_AREA
    min_bbox_ratio = CFG_MIN_BBOX_RATIO
    min_bbox_count = CFG_MIN_BBOX_COUNT
    isnet_weights = CFG_ISNET_WEIGHTS
    prob_csv_path = CFG_PROB_CSV_PATH
    json_dir = CFG_JSON_DIR

    os.makedirs(output_dir, exist_ok=True)

    _log_path = os.path.join(output_dir, 'run.log')
    _log_fp = open(_log_path, 'w', encoding='utf-8')
    _stdout_bak, _stderr_bak = sys.stdout, sys.stderr
    sys.stdout = _TeeIO(_stdout_bak, _log_fp)
    sys.stderr = _TeeIO(_stderr_bak, _log_fp)
    _sal_temp_dir = None
    try:
        import datetime as _dt
        print(f"[LOG] Mirror log: {_log_path}")
        print(f"[LOG] Started: {_dt.datetime.now().isoformat(timespec='seconds')}")
        print(f"[LOG] argv: {sys.argv}")
        print("=" * 60)
        print("COLLA LAYOUT PIPELINE (Voronoi v4 HARDENED)")
        print("=" * 60)
        print(f"[CONFIG] Input shape      : {input_shape}")
        print(f"[CONFIG] Image folder     : {input_image_collection_folder}")
        print(f"[CONFIG] Output dir       : {output_dir}")
        print(f"[CONFIG] Scaling factor   : {scaling_factor}")
        print(f"[CONFIG] Voronoi layout   : {use_voronoi_layout} (FROZEN)")
        print(f"[CONFIG] Timeline order   : {use_timeline_order} (FROZEN)")
        print(f"[CONFIG] Num iterations   : {num_iterations} (FROZEN)")
        print(f"[CONFIG] Skip warp        : {skip_warp} (FROZEN)")
        print(f"[CONFIG] Shape input      : "
              f"{'mask file (no U2-Net)' if shape_input_is_mask else 'RGB image → U2-Net'}")
        print(f"[CONFIG] Saliency source  : ISNet bbox (FROZEN)")
        print(f"[CONFIG] Min bbox ratio   : {min_bbox_ratio} (FROZEN)")
        print(f"[CONFIG] Min bbox count   : {min_bbox_count} (FROZEN)")
        if run_isnet:
            print(f"[CONFIG] Run ISNet        : True (FROZEN)")
            print(f"[CONFIG] ISNet weights    : {isnet_weights}")
            print(f"[CONFIG] ISNet threshold  : {isnet_threshold} (FROZEN)")
            print(f"[CONFIG] ISNet min area   : {isnet_min_area} (FROZEN)")
            print(f"[CONFIG] Filter frames by ISNet bbox rules: "
                  f"{'ON' if filter_frames_by_isnet else 'OFF'}")
        print("=" * 60)

        # ========================================================================
        # STEP 0: Shape mask from U2-Net (RGB) or direct mask file
        # ========================================================================
        step_times = {}
        t_step = time.time()
        shape_mask_path = os.path.join(output_dir, "shape_mask_refined.png")

        if shape_input_is_mask:
            print(f"\n[STEP 0] Loading direct shape mask (U2-Net skipped): {input_shape}")
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
        else:
            print(f"\n[STEP 0] Processing layout shape image (U2-Net): {input_shape}")
            image = cv2.imread(input_shape)
            if image is None:
                raise FileNotFoundError(f"Cannot load input shape image: {input_shape}")

            print(f"  Image shape: {image.shape}")
            inputs, orig_h, orig_w = preprocess_image(image)
            print(f"  Original size: {orig_w}x{orig_h}")
            print("  Predicting mask with U2-Net (change model in utils/get_mask if needed)...")
            from utils.get_mask import net
            pred_mask = predict_mask(net, inputs)
            print("  Refining mask...")
            mask_refined = refine_mask(pred_mask, orig_h, orig_w)
            cv2.imwrite(shape_mask_path, mask_refined)
            print(f"  Saved refined mask to: {shape_mask_path}")

        step_times['Step 0 (mask)'] = time.time() - t_step
        t_step = time.time()

        # ========================================================================
        # STEP 0.5: ISNet Detection (FROZEN ENABLED)
        # ========================================================================
        warp_mask_dir = None
        summary_json_path = None

        if run_isnet:
            if not HAS_ISNET:
                print("[ERROR] isnet-detector not installed! Run: cd isnet_detector && pip install -e .")
                sys.exit(1)

            isnet_output_dir = os.path.join(output_dir, 'isnet_output')
            isnet_saliency_dir = os.path.join(isnet_output_dir, 'saliency_masks')
            isnet_bbox_vis_dir = os.path.join(isnet_output_dir, 'bbox_detection')
            isnet_summary = os.path.join(isnet_output_dir, 'summary.json')
            filtered_summary = os.path.join(isnet_output_dir, 'filtered_summary.json')

            if os.path.isfile(isnet_summary):
                print(f"[STEP 0.5] Found summary.json, skipping ISNet forward pass")
                print(f"  summary.json : {isnet_summary}")
                with open(isnet_summary) as f:
                    summary = json.load(f)
            else:
                print(f"[STEP 0.5] Running ISNet detection (bbox mode)...")
                print(f"  Input : {input_image_collection_folder}")
                print(f"  Output: {isnet_output_dir}")

                detector = SimpleISNetDetector(
                    model_path=isnet_weights,
                    device="cuda:0",
                    use_u2net=False,          # ISNet only
                    img_size=1024,            # FROZEN: 1024 for quality
                )

                from pathlib import Path
                os.makedirs(isnet_bbox_vis_dir, exist_ok=True)

                frames = sorted(Path(input_image_collection_folder).glob('*.jpg')) + \
                         sorted(Path(input_image_collection_folder).glob('*.png'))
                frames = sorted(frames)
                print(f"  Processing {len(frames)} frames...")

                summary = {
                    'frames': [],
                    'total_objects': 0,
                    'params': {
                        'threshold': isnet_threshold,
                        'min_area': isnet_min_area,
                        'use_u2net': False,
                    }
                }

                from PIL import Image as _PIL_Image
                for idx, frame_path in enumerate(frames, 1):
                    img_np = np.array(_PIL_Image.open(str(frame_path)).convert('RGB'))
                    img_h, img_w = img_np.shape[:2]

                    # ISNet detection → bboxes
                    objects, mask_binary = detector.detect_objects(
                        str(frame_path),
                        threshold=isnet_threshold,
                        min_area=isnet_min_area,
                        merge_kernel=11,   # FROZEN: match CLI default
                        adaptive_morph=True,
                    )

                    det_list = [
                        {
                            'bbox': list(o['bbox']),
                            'pixel_area': o['pixel_area'],
                            'bbox_area': o['bbox_area'],
                            'size': list(o['size']),
                            'confidence': o['confidence'],
                        }
                        for o in objects
                    ]
                    vis_bgr = _draw_bbox_detection_vis(img_np, det_list)
                    cv2.imwrite(
                        os.path.join(isnet_bbox_vis_dir, frame_path.name),
                        vis_bgr,
                    )

                    summary['frames'].append({
                        'name': frame_path.name,
                        'num_objects': len(objects),
                        'frame_size': [img_w, img_h],
                        'objects': det_list,
                    })
                    summary['total_objects'] += len(objects)

                    if idx % 10 == 0 or idx == len(frames):
                        print(f"    [{idx}/{len(frames)}] {frame_path.name}: "
                              f"{len(objects)} objects")

                summary['avg_objects_per_frame'] = (
                    summary['total_objects'] / len(frames) if frames else 0
                )
                with open(isnet_summary, 'w') as f:
                    json.dump(summary, f, indent=2)

                del detector
                import torch
                torch.cuda.empty_cache()
                print(f"  ISNet done: {summary['total_objects']} objects across "
                      f"{len(frames)} frames")

            # Frame filtering (optional)
            if filter_frames_by_isnet:
                print(f"\n[STEP 0.5b] Filtering frames "
                      f"(min_bbox_count={min_bbox_count}, min_bbox_ratio={min_bbox_ratio})...")

                kept, dropped = [], []
                for frame_info in summary['frames']:
                    fs = frame_info.get('frame_size') or frame_info.get('img_size', [960, 540])
                    img_w, img_h = fs[0], fs[1]
                    img_area = img_w * img_h
                    objects = frame_info.get('objects', [])

                    valid_objects = [
                        o for o in objects
                        if min_bbox_ratio * img_area
                             <= o.get('bbox_area', 0)
                    ]

                    if len(valid_objects) >= min_bbox_count:
                        kept.append({**frame_info,
                                     'objects': valid_objects,
                                     'num_objects': len(valid_objects)})
                    else:
                        dropped.append(frame_info['name'])

                print(f"  Kept   : {len(kept)} frames")
                print(f"  Dropped: {len(dropped)} frames")
                if dropped:
                    print(f"  Dropped frames: {dropped[:10]}"
                          + (" ..." if len(dropped) > 10 else ""))
            else:
                print("\n[STEP 0.5b] Filter frames by ISNet bbox rules: OFF — keeping all frames")
                kept = copy.deepcopy(summary['frames'])
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
                print("[ERROR] No frames left after ISNet step (summary empty or all filtered out)!")
                sys.exit(1)

            _sync_saliency_masks_for_kept_frames(kept, isnet_saliency_dir)
            warp_mask_dir = isnet_saliency_dir
            summary_json_path = filtered_summary
            print(f"  → saliency_masks/ : {isnet_saliency_dir} ({len(kept)} frames, for eval + warp)")
            print(f"  → summary-json    : {summary_json_path}")

            step_times['Step 0.5 (isnet)'] = time.time() - t_step
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
        use_timeline_order = True  # Enforce v3 behavior
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
            allow_empty_detection=run_isnet and (not filter_frames_by_isnet),
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

            img_shape = cv2.imread(input_shape)
            if img_shape is not None:
                if img_shape.ndim == 2 or (img_shape.ndim == 3 and img_shape.shape[2] == 1):
                    axes[0].imshow(img_shape.squeeze(), cmap='gray')
                else:
                    axes[0].imshow(cv2.cvtColor(img_shape, cv2.COLOR_BGR2RGB))
            axes[0].set_title(
                'Input (mask file)' if shape_input_is_mask else 'Input (RGB)', fontsize=14)
            axes[0].axis('off')

            mask_img = cv2.imread(shape_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                axes[1].imshow(mask_img, cmap='gray')
            axes[1].set_title(
                'Shape mask (from file)' if shape_input_is_mask else 'Shape mask (U2-Net)',
                fontsize=14)
            axes[1].axis('off')

            voronoi_debug = os.path.join(output_dir, '_voronoi_temp.png')
            slicing_path = os.path.join(output_dir, 'slicing_result.json')
            if os.path.isfile(slicing_path):
                with open(slicing_path, 'r') as f:
                    slicing = json.load(f)
                layout_vis = (cv2.imread(voronoi_debug)
                              if os.path.isfile(voronoi_debug)
                              else np.zeros((mask_img.shape[0], mask_img.shape[1], 3),
                                            dtype=np.uint8))
                if layout_vis is not None:
                    colors = plt.cm.tab20(
                        np.linspace(0, 1, len(slicing.get('parts', []))))
                    for i, part in enumerate(slicing.get('parts', [])):
                        coords = np.array(part['coords'], dtype=np.int32)
                        color = tuple(int(c * 255) for c in colors[i][:3])
                        cv2.polylines(layout_vis, [coords], True, color, 2)
                    axes[2].imshow(cv2.cvtColor(layout_vis, cv2.COLOR_BGR2RGB))
            axes[2].set_title('Voronoi Layout', fontsize=14)
            axes[2].axis('off')

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

        print("\n[DONE] Pipeline completed successfully!")
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
