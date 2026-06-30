#!/usr/bin/env python3
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
    print(f"\n[STEP 0] Loading shape mask: {input_shape}")
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
    print(f"\n[STEP 0.5] Running ISNet detection...")
    
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
        print(f"\n[STEP 0.5b] Filtering frames...")
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
        print("\n[STEP 0.5b] No filtering - keeping all frames")
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
    
    print(f"\n[STEP 0.5c] Copying filtered images and masks...")
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
    print(f"\n✓ ISNet-only processing completed!")
    print(f"  Images: {images_dir}")
    print(f"  Masks: {masks_dir}")
