#!/usr/bin/env python3
"""
Batch process films for ISNet detection and filtering only.
Creates folder structure with images and masks after ISNet filtering and box filtering.
Stops after preprocessing - does NOT run layout generation.

Usage:
    python batch_isnet_only.py
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path

# Keep keyframe input structure aligned with run_ablation.sh
DATA_ROOT = "/home/serverai/ltdoanh/LayoutGeneration/FINAL_data"
KF_DIR = f"{DATA_ROOT}/keyframe/inference_recerr_batch_deduplicate"
MASK_DIR = f"{DATA_ROOT}/input_layout/input_custom_mask/cropped"
CAST_DIR = "/home/serverai/ltdoanh/LayoutGeneration/CAST_loss"

DEFAULT_ISNET_WEIGHT = f"{CAST_DIR}/isnet-detector/weights/isnetis.ckpt"
GENERAL_ISNET_WEIGHT = (
    "/home/serverai/ltdoanh/LayoutGeneration/CAST_loss/isnet-detector/weights/isnet-general-use.pth"
)

# Video -> mask mapping aligned with run_inference.sh
VIDEO_MASK = {
    "Your_name": "Your_name.png",
    "Nobody": "Nobody.png",
    "Kpop_demon_hunter": "Kpop_demon_hunter.png",
    "Zootopia": "Zootopia.png",
    "Inside_out": "Inside_out.png",
    "Quintessential": "Quintessential.png",
    "Stranger_thing": "Stranger_thing.png",
    "Golden": "Golden.png",
    "Luca": "Luca.png",
    "Bocchi_the_rock": "Bocchi_the_rock.png",
    "Umaru": "Umaru.png",
    "Onepiece": "Onepiece.png",
    "Spider_man": "Spider_man.png",
    "Avatar3": "Avatar3.png",
    "Project_hail_mary": "Project_hail_mary.png",
    "Squirrel": "Squirrel.png",
    "Moana": "Moana.png",
}

GENERAL_VIDEOS = {
    "Onepiece",
    "Spider_man",
    "Avatar3",
    "Project_hail_mary",
    "Squirrel",
    "Moana",
}


def has_image_files(folder: Path) -> bool:
    return any(folder.glob("*.jpg")) or any(folder.glob("*.png"))


def resolve_isnet_weight(video_name: str, override: str | None = None) -> str:
    if override:
        return override
    return GENERAL_ISNET_WEIGHT if video_name in GENERAL_VIDEOS else DEFAULT_ISNET_WEIGHT


def infer_video_name_from_keyframes_dir(keyframes_dir: Path) -> str:
    model_dir_names = {"recerr", "v11", "vsumm", "llmvs"}
    if keyframes_dir.name.lower() in model_dir_names and keyframes_dir.parent.name:
        return keyframes_dir.parent.name
    return keyframes_dir.name


def build_single_keyframes_config(
    keyframes_dir: Path,
    mask_root: Path,
    filter_isnet: bool,
    isnet_weight_override: str | None = None,
) -> dict | None:
    if not keyframes_dir.is_dir() or not has_image_files(keyframes_dir):
        print(f"[ERROR] Keyframes folder missing or empty: {keyframes_dir}")
        return None

    video_name = infer_video_name_from_keyframes_dir(keyframes_dir)
    mask_name = VIDEO_MASK.get(video_name, f"{video_name}.png")
    layout_path = mask_root / mask_name
    if not layout_path.is_file():
        print(f"[ERROR] Missing mask for {video_name}: {layout_path}")
        return None

    return {
        "name": video_name,
        "layout": str(layout_path),
        "keyframes": str(keyframes_dir),
        "scale": 2,
        "filter_isnet": filter_isnet,
        "isnet_weight": resolve_isnet_weight(video_name, isnet_weight_override),
    }


def discover_film_configs(
    kf_root: Path,
    mask_root: Path,
    filter_isnet: bool,
    isnet_weight_override: str | None = None,
) -> list[dict]:
    """Build configs by scanning <kf_root>/<video>/keyframes folders."""
    configs: list[dict] = []
    skipped: list[str] = []

    if not kf_root.exists() or not kf_root.is_dir():
        print(f"[ERROR] Keyframe root not found: {kf_root}")
        return configs

    for video_dir in sorted(kf_root.iterdir()):
        if not video_dir.is_dir():
            continue

        keyframes_dir = video_dir / "recerr"
        if not keyframes_dir.is_dir() or not has_image_files(keyframes_dir):
            skipped.append(f"{video_dir.name}: missing/empty keyframes")
            continue

        mask_name = VIDEO_MASK.get(video_dir.name, f"{video_dir.name}.png")
        layout_path = mask_root / mask_name
        if not layout_path.is_file():
            skipped.append(f"{video_dir.name}: missing mask ({layout_path})")
            continue

        configs.append(
            {
                "name": video_dir.name,
                "layout": str(layout_path),
                "keyframes": str(keyframes_dir),
                "scale": 2,
                "filter_isnet": filter_isnet,
                "isnet_weight": resolve_isnet_weight(video_dir.name, isnet_weight_override),
            }
        )

    print(f"Discovered {len(configs)} valid video configs from {kf_root}")
    if skipped:
        print("Skipped entries:")
        for item in skipped:
            print(f"  - {item}")
    return configs


def run_isnet_only(config, base_output_dir):
    """Run ISNet detection and filtering only for a single film configuration."""
    print(f"\n{'='*60}")
    print(f"Processing: {config['name']} (ISNet only)")
    print(f"{'='*60}")
    
    # Create temporary output directory for ISNet processing
    temp_output = f"/tmp/isnet_{config['name']}"
    os.makedirs(temp_output, exist_ok=True)
    
    # Build command for ISNet-only processing
    # We'll use a modified script that stops after ISNet
    cmd = [
        "python", "isnet_only.py",  # New script that only does ISNet
        config["layout"],
        config["keyframes"],
        temp_output,
        str(config["scale"]),
        "--shape-is-mask",
        f"--isnet-weights={config['isnet_weight']}",
    ]
    
    # Add ISNet filter flag
    if config["filter_isnet"]:
        cmd.append("--filter-frames-by-isnet")
    else:
        cmd.append("--no-filter-frames-by-isnet")
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            cwd=CAST_DIR,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout per film
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully processed {config['name']}")
            
            # Copy processed images and masks to organized output
            copy_isnet_data(config, temp_output, base_output_dir)
            
            # Clean up temp directory
            shutil.rmtree(temp_output, ignore_errors=True)
            return True
            
        else:
            print(f"✗ Failed to process {config['name']}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout processing {config['name']}")
        return False
    except Exception as e:
        print(f"✗ Error processing {config['name']}: {e}")
        return False

def create_isnet_only_script():
    """Create a standalone ISNet-only script."""
    script_content = '''#!/usr/bin/env python3
"""
ISNet-only script - runs detection and filtering only.
Standalone version without importing run.py.
"""

import sys
import os
import cv2
import json
import time
import numpy as np
import copy
import shutil
from pathlib import Path
from PIL import Image as PILImage

if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) < 5:
        print("Usage: python isnet_only.py <shape> <images> <output> <scale> [--shape-is-mask] [--filter-frames-by-isnet] [--isnet-weights=/path/to/weights]")
        sys.exit(1)
        
    # Set up arguments
    input_shape = sys.argv[1]
    input_image_collection_folder = sys.argv[2]
    output_dir = sys.argv[3]
    scaling_factor = int(sys.argv[4])
    
    # Parse flags
    shape_input_is_mask = False
    filter_frames_by_isnet = True
    isnet_weights = "/home/serverai/ltdoanh/LayoutGeneration/CAST_loss/isnet-detector/weights/isnetis.ckpt"
    
    for arg in sys.argv[5:]:
        if arg == "--shape-is-mask":
            shape_input_is_mask = True
        elif arg == "--filter-frames-by-isnet":
            filter_frames_by_isnet = True
        elif arg == "--no-filter-frames-by-isnet":
            filter_frames_by_isnet = False
        elif arg.startswith("--isnet-weights="):
            isnet_weights = arg.split("=", 1)[1].strip()
    
    print("ISNet-ONLY PROCESSING")
    print("=" * 50)
    
    # Import ISNet detector
    try:
        from isnet_detector import SimpleISNetDetector
        HAS_ISNET = True
    except Exception as e:
        HAS_ISNET = False
        print(f"[ERROR] isnet-detector not available: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 0: Shape mask (load directly since --shape-is-mask)
    print(f"\\n[STEP 0] Loading shape mask: {input_shape}")
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
    
    shape_mask_path = os.path.join(output_dir, "shape_mask_refined.png")
    cv2.imwrite(shape_mask_path, mask_refined)
    print(f"  Saved shape mask: {shape_mask_path}")
    
    # Step 0.5: ISNet Detection
    print(f"\\n[STEP 0.5] Running ISNet detection...")
    
    isnet_output_dir = os.path.join(output_dir, 'isnet_output')
    isnet_saliency_dir = os.path.join(isnet_output_dir, 'saliency_masks')
    isnet_bbox_vis_dir = os.path.join(isnet_output_dir, 'bbox_detection')
    isnet_summary = os.path.join(isnet_output_dir, 'summary.json')
    filtered_summary = os.path.join(isnet_output_dir, 'filtered_summary.json')
    
    os.makedirs(isnet_bbox_vis_dir, exist_ok=True)
    os.makedirs(isnet_saliency_dir, exist_ok=True)
    
    # ISNet parameters (from CAST run.py)
    isnet_threshold = 0.1  # From CAST config
    isnet_min_area = 0     # From CAST config
    min_bbox_ratio = 0.05   # From CAST config
    min_bbox_count = 1     # From CAST config
    
    detector = SimpleISNetDetector(
        model_path=isnet_weights,
        device="cuda:0",
        use_u2net=False,
        img_size=1024,
    )
    
    frames = sorted(Path(input_image_collection_folder).glob('*.jpg')) + \\
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
    
    def _draw_bbox_detection_vis(img_np, det_list):
        """Draw bounding boxes on image."""
        vis = img_np.copy()
        for det in det_list:
            x, y, w, h = det['bbox']
            cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        return vis

    def _build_center_fallback_mask(img_h, img_w):
        """Create 50% center rectangle mask (same spirit as run.py fallback)."""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        margin = 0.25  # 50% center region
        x1 = int(img_w * margin)
        x2 = int(img_w * (1 - margin))
        y1 = int(img_h * margin)
        y2 = int(img_h * (1 - margin))
        mask[y1:y2, x1:x2] = 255
        return mask
    
    for idx, frame_path in enumerate(frames, 1):
        img_np = np.array(PILImage.open(str(frame_path)).convert('RGB'))
        img_h, img_w = img_np.shape[:2]
        
        # ISNet detection
        objects, mask_binary = detector.detect_objects(
            str(frame_path),
            threshold=isnet_threshold,
            min_area=isnet_min_area,
            merge_kernel=11,
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
        
        # Save saliency mask
        if mask_binary is not None:
            cv2.imwrite(
                os.path.join(isnet_saliency_dir, frame_path.name),
                mask_binary
            )
        
        summary['frames'].append({
            'name': frame_path.name,
            'num_objects': len(objects),
            'frame_size': [img_w, img_h],
            'objects': det_list,
        })
        summary['total_objects'] += len(objects)
        
        if idx % 10 == 0 or idx == len(frames):
            print(f"    [{idx}/{len(frames)}] {frame_path.name}: {len(objects)} objects")
    
    with open(isnet_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    
    del detector
    import torch
    torch.cuda.empty_cache()
    print(f"  ISNet done: {summary['total_objects']} objects across {len(frames)} frames")
    
    # Frame filtering
    if filter_frames_by_isnet:
        print(f"\\n[STEP 0.5b] Filtering frames...")
        kept, dropped = [], []
        for frame_info in summary['frames']:
            fs = frame_info.get('frame_size') or frame_info.get('img_size', [960, 540])
            img_w, img_h = fs[0], fs[1]
            img_area = img_w * img_h
            objects = frame_info.get('objects', [])
            
            valid_objects = [
                o for o in objects
                if min_bbox_ratio * img_area <= o.get('bbox_area', 0)
            ]
            
            if len(valid_objects) >= min_bbox_count:
                kept.append({**frame_info, 'objects': valid_objects, 'num_objects': len(valid_objects)})
            else:
                print(f"    [DROP] {frame_info.get('name', '<unknown>')}: raw={len(objects)} valid={len(valid_objects)}")
                dropped.append(frame_info['name'])
        
        print(f"  Kept: {len(kept)} frames")
        print(f"  Dropped: {len(dropped)} frames")
    else:
        print("\\n[STEP 0.5b] No filtering - keeping all frames")
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
    
    # Copy filtered images and masks to output directories
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print(f"\\n[STEP 0.5c] Copying filtered images and masks...")
    for frame_info in kept:
        frame_name = frame_info['name']
        
        # Copy original image
        src_img = os.path.join(input_image_collection_folder, frame_name)
        dst_img = os.path.join(images_dir, frame_name)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        
        # Copy saliency mask
        src_mask = os.path.join(isnet_saliency_dir, frame_name)
        dst_mask = os.path.join(masks_dir, frame_name)
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)
        else:
            # Match run.py behavior: if no detection but frame kept, write center fallback mask.
            fs = frame_info.get('frame_size') or frame_info.get('img_size')
            if fs and len(fs) >= 2:
                img_w, img_h = int(fs[0]), int(fs[1])
            else:
                img = cv2.imread(src_img)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
            fallback_mask = _build_center_fallback_mask(img_h, img_w)
            cv2.imwrite(dst_mask, fallback_mask)
    
    print(f"  ✓ Copied {len(kept)} images and masks")
    print(f"\\n✓ ISNet-only processing completed!")
    print(f"  Images: {images_dir}")
    print(f"  Masks: {masks_dir}")
'''
    
    script_path = f"{CAST_DIR}/isnet_only.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"Created ISNet-only script: {script_path}")

def copy_isnet_data(config, temp_output, base_output_dir):
    """Copy processed images and masks from temp output to organized directory."""
    # Create output directory for this film
    film_output_dir = os.path.join(base_output_dir, config['name'])
    os.makedirs(film_output_dir, exist_ok=True)
    
    # Source directories from temp output
    images_dir = os.path.join(temp_output, "images")
    masks_dir = os.path.join(temp_output, "masks")
    
    # Destination directories
    dest_images = os.path.join(film_output_dir, "images")
    dest_masks = os.path.join(film_output_dir, "masks")
    
    # Copy images if they exist
    if os.path.exists(images_dir):
        print(f"  Copying images from {images_dir}")
        if os.path.exists(dest_images):
            shutil.rmtree(dest_images)
        shutil.copytree(images_dir, dest_images)
        
        image_files = [f for f in os.listdir(dest_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Copied {len(image_files)} images")
    else:
        print(f"  ⚠ Images directory not found: {images_dir}")
    
    # Copy masks if they exist
    if os.path.exists(masks_dir):
        print(f"  Copying masks from {masks_dir}")
        if os.path.exists(dest_masks):
            shutil.rmtree(dest_masks)
        shutil.copytree(masks_dir, dest_masks)
        
        mask_files = [f for f in os.listdir(dest_masks) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Copied {len(mask_files)} masks")
    else:
        print(f"  ⚠ Masks directory not found: {masks_dir}")
    
    # Save configuration info
    config_file = os.path.join(film_output_dir, "config.json")
    with open(config_file, 'w') as f:
        clean_config = {
            "name": config["name"],
            "scale": config["scale"],
            "filter_isnet": config["filter_isnet"],
            "isnet_weight": config.get("isnet_weight", ""),
            "original_layout": config["layout"],
            "original_keyframes": config["keyframes"],
            "cast_output": config.get("output", ""),
            "processing_type": "isnet_only"
        }
        json.dump(clean_config, f, indent=2)
    
    print(f"  ✓ Saved data to {film_output_dir}")

def main():
    """Main function to process all films with ISNet only."""
    parser = argparse.ArgumentParser(
        description="Auto-scan keyframe folders and export ISNet images/masks per video"
    )
    parser.add_argument("--keyframe-root", default=KF_DIR, help="Root folder containing <video>/recerr")
    parser.add_argument(
        "--keyframes-dir",
        default=None,
        help="Direct folder containing keyframe images for one video, e.g. <root>/Swapped/llmvs",
    )
    parser.add_argument("--mask-dir", default=MASK_DIR, help="Folder containing shape masks")
    parser.add_argument(
        "--isnet-weight",
        default=None,
        help="Override ISNet weight path for all processed inputs",
    )
    parser.add_argument(
        "--filter-frames-by-isnet",
        dest="filter_frames_by_isnet",
        action="store_true",
        help="Filter frames by ISNet bbox rules (default)",
    )
    parser.add_argument(
        "--no-filter-frames-by-isnet",
        dest="filter_frames_by_isnet",
        action="store_false",
        help="Keep all frames; if a frame has no detected object, generate center rectangle mask",
    )
    parser.set_defaults(filter_frames_by_isnet=True)
    parser.add_argument(
        "--output",
        default=f"{DATA_ROOT}/batch_isnet_only",
        help="Single output root folder for all videos",
    )
    args = parser.parse_args()

    print("CAST Batch ISNet-Only Processor")
    print("=" * 60)
    print("This script runs ISNet detection and filtering ONLY")
    print("No shape decomposition or layout generation")
    print("=" * 60)

    # Base output directory for organized data
    base_output_dir = args.output
    os.makedirs(base_output_dir, exist_ok=True)

    if args.keyframes_dir:
        config = build_single_keyframes_config(
            Path(args.keyframes_dir),
            Path(args.mask_dir),
            filter_isnet=args.filter_frames_by_isnet,
            isnet_weight_override=args.isnet_weight,
        )
        film_configs = [config] if config else []
    else:
        film_configs = discover_film_configs(
            Path(args.keyframe_root),
            Path(args.mask_dir),
            filter_isnet=args.filter_frames_by_isnet,
            isnet_weight_override=args.isnet_weight,
        )

    print(f"Output directory: {base_output_dir}")
    print(f"Processing {len(film_configs)} films...")

    if not film_configs:
        print("No valid video config found. Nothing to process.")
        return

    # Create the ISNet-only script once.
    create_isnet_only_script()

    # Process each film
    successful = 0
    failed = 0

    for config in film_configs:
        try:
            ok = run_isnet_only(config, base_output_dir)
            if ok:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Critical error processing {config['name']}: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH ISNET-ONLY PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total films: {len(film_configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {base_output_dir}")
    
    # List processed films
    if successful > 0:
        print(f"\nProcessed films:")
        for config in film_configs:
            film_dir = os.path.join(base_output_dir, config['name'])
            if os.path.exists(film_dir):
                images_count = len([f for f in os.listdir(os.path.join(film_dir, "images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(os.path.join(film_dir, "images")) else 0
                masks_count = len([f for f in os.listdir(os.path.join(film_dir, "masks")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(os.path.join(film_dir, "masks")) else 0
                print(f"  - {config['name']}: {images_count} images, {masks_count} masks")

if __name__ == "__main__":
    main()
