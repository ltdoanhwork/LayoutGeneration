#!/usr/bin/env python3
"""
Generate synthetic test cases for ablation study of 3 loss functions.
Design adapted for Publication-ready Figures:
  - Neutral solid gray frames to highlight red overlaps in final evaluation.
  - Thematic layout masks: Water Drop, Fish, Leaf.

  L_cap_res  — residual capacity matching (cell area ∝ saliency importance)
  L_cvt_norm — normalised centroidal regularity (compact, non-wandering cells)
  L_fea      — one-sided feasibility penalty (min 70% of target per cell)
"""

import os
import json
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------

def create_layout_mask(width: int, height: int, mask_type: str) -> Image.Image:
    """
    Create layout shape mask (binary) for cell arrangement.
    Shapes are chosen to reflect an environmental/water theme.
    """
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = width // 2, height // 2
    R = min(width, height)

    if mask_type == 'water_drop':
        # Water drop shape (perfect for testing gravity/bottom-heavy layouts)
        drop_r = int(0.25 * R)
        drop_cy = cy + int(0.1 * R)
        # Bottom circle
        draw.ellipse([cx - drop_r, drop_cy - drop_r, cx + drop_r, drop_cy + drop_r], fill=255)
        # Top triangle pointing up
        top_tip_y = cy - int(0.4 * R)
        draw.polygon([
            (cx, top_tip_y),
            (cx - drop_r, drop_cy),
            (cx + drop_r, drop_cy)
        ], fill=255)

    elif mask_type == 'fish':
        # Fish silhouette (testing elongated horizontal layouts)
        body_w, body_h = int(0.35 * R), int(0.20 * R)
        draw.ellipse([cx - body_w, cy - body_h, cx + body_w, cy + body_h], fill=255)
        tail_pts = [
            (cx - body_w + 10, cy),
            (cx - body_w - int(0.25 * R), cy - int(0.15 * R)),
            (cx - body_w - int(0.25 * R), cy + int(0.15 * R)),
        ]
        draw.polygon(tail_pts, fill=255)
        eye_r = int(0.04 * R)
        draw.ellipse([cx + int(0.15 * R), cy - eye_r, 
                      cx + int(0.15 * R) + 2*eye_r, cy + eye_r], fill=0)

    elif mask_type == 'leaf':
        # Leaf shape (testing symmetric/diagonal layouts)
        leaf_h = int(0.65 * R)
        leaf_w = int(0.25 * R)
        pts = [
            (cx, cy - leaf_h//2),           # top tip
            (cx + leaf_w, cy - leaf_h//6),  # right upper
            (cx + leaf_w//2, cy),           # right middle
            (cx + leaf_w, cy + leaf_h//6),  # right lower
            (cx, cy + leaf_h//2),           # bottom tip
            (cx - leaf_w, cy + leaf_h//6),  # left lower
            (cx - leaf_w//2, cy),           # left middle
            (cx - leaf_w, cy - leaf_h//6),  # left upper
        ]
        draw.polygon(pts, fill=255)
        vein_w = int(0.015 * R)
        draw.line([(cx, cy - leaf_h//2 + 5), (cx, cy + leaf_h//2 - 5)], fill=0, width=vein_w)

    elif mask_type == 'rectangle':
        # Baseline control
        pad = int(0.15 * R)
        draw.rectangle([pad, pad, width - pad, height - pad], fill=255)

    else:
        raise ValueError(f"Unknown mask_type: {mask_type!r}")

    return img


def create_frame_mask(width: int, height: int, mask_shape: str = 'rectangle') -> Image.Image:
    """
    Create frame-level mask (smaller rectangle inside frame).
    This simulates the saliency/object area for capacity matching.
    """
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = width // 2, height // 2
    
    if mask_shape == 'rectangle':
        rw, rh = int(width * 0.60), int(height * 0.60)
    elif mask_shape == 'small_rect':
        rw, rh = int(width * 0.40), int(height * 0.40)
    elif mask_shape == 'large_rect':
        rw, rh = int(width * 0.75), int(height * 0.75)
    else:
        rw, rh = int(width * 0.60), int(height * 0.60)
        
    x1, y1 = cx - rw//2, cy - rh//2
    draw.rectangle([x1, y1, x1 + rw, y1 + rh], fill=255)
    return img


# ---------------------------------------------------------------------------
# Frame generator (Publication-Ready Neutral Palette)
# ---------------------------------------------------------------------------

def make_simple_frame(fw: int, fh: int, frame_idx: int) -> Image.Image:
    """
    Create a neutral solid gray rectangle frame.
    Avoids gradients so that overlap visualization (red/blue) in evaluation is clear.
    """
    # Base canvas: Solid neutral gray (#808080)
    bg_color = (128, 128, 128)
    img = Image.new('RGB', (fw, fh), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw a thin white border to separate individual frames slightly
    draw.rectangle([0, 0, fw-1, fh-1], outline=(255, 255, 255), width=2)
    
    # Inner subject box (darker gray to represent the 'saliency' object visually)
    rw, rh = int(fw * 0.50), int(fh * 0.50)
    cx, cy = fw // 2, fh // 2
    x1, y1 = cx - rw // 2, cy - rh // 2
    draw.rectangle([x1, y1, x1 + rw, y1 + rh], fill=(100, 100, 100), outline=(200, 200, 200), width=1)
    
    # Simple frame label
    label = f"{frame_idx:02d}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    
    # White text with black outline for readability
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((15 + dx, 15 + dy), label, font=font, fill=(0, 0, 0))
    draw.text((15, 15), label, font=font, fill=(255, 255, 255))
    
    return img


# ---------------------------------------------------------------------------
# Test Case Utilities
# ---------------------------------------------------------------------------

def get_frame_mask_shape(n: int, style: str) -> List[str]:
    if style == 'varying':
        n_large = max(1, n * 3 // 10)
        n_small = max(1, n * 3 // 10)
        n_med = n - n_large - n_small
        return ['large_rect'] * n_large + ['rectangle'] * n_med + ['small_rect'] * n_small
    if style == 'mixed':
        return ['large_rect' if i % 2 == 0 else 'small_rect' for i in range(n)]
    return ['rectangle'] * n


def create_test_case(
    case_name: str, num_frames: int, layout_mask_type: str,
    frame_style: str, description: str,
    output_base: str,
    mask_size=(700, 500), frame_size=(400, 300)
):
    case_dir = os.path.join(output_base, case_name)
    frames_dir = os.path.join(case_dir, 'frames')
    mask_dir = os.path.join(case_dir, 'mask')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Layout mask
    layout_mask = create_layout_mask(*mask_size, layout_mask_type)
    layout_mask_path = os.path.join(case_dir, 'shape_mask.png')
    layout_mask.save(layout_mask_path)

    fw, fh = frame_size
    frame_mask_shapes = get_frame_mask_shape(num_frames, frame_style)
    frame_infos = []

    for i, mask_shape in enumerate(frame_mask_shapes):
        # Create identical gray frames
        img = make_simple_frame(fw, fh, i)
        # Using PNG instead of JPG ensures no compression artifacts on the borders
        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        img.save(frame_path)
        
        # Save corresponding frame mask
        frame_mask = create_frame_mask(fw, fh, mask_shape)
        mask_path = os.path.join(mask_dir, f'frame_{i:03d}_mask.png')
        frame_mask.save(mask_path)
        
        # Calculate bbox
        mask_arr = np.array(frame_mask)
        ys, xs = np.where(mask_arr > 128)
        if len(xs) > 0 and len(ys) > 0:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            capacity = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (fw * fh)
        else:
            bbox = [0, 0, fw, fh]
            capacity = 1.0

        frame_infos.append({
            'filename': f'frame_{i:03d}.png',
            'mask_filename': f'frame_{i:03d}_mask.png',
            'bbox': bbox,
            'capacity': round(capacity, 4),
            'mask_shape': mask_shape,
        })

    metadata = {
        'case_name': case_name,
        'description': description,
        'num_frames': num_frames,
        'layout_mask_type': layout_mask_type,
        'frame_style': frame_style,
        'mask_size': list(mask_size),
        'frame_size': list(frame_size),
        'layout_mask_path': layout_mask_path,
        'frames_dir': frames_dir,
        'mask_dir': mask_dir,
        'frame_infos': frame_infos,
    }
    
    with open(os.path.join(case_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    fg_px = int(np.array(layout_mask).sum() // 255)
    fg_pct = fg_px / (mask_size[0] * mask_size[1]) * 100
    print(f"  {case_name:<18} {num_frames:>2}f  layout={layout_mask_type:<10} fg={fg_pct:>4.1f}%")
    
    return case_dir


if __name__ == '__main__':
    OUTPUT_BASE = '/home/serverai/ltdoanh/LayoutGeneration/FINAL_data/synthetic_tests'

    # Thematic Test Cases: Environmental Focus
    TEST_CASES = [
        ('test_cap',       10, 'water_drop', 'varying', 'L_cap_res: varying mask sizes, water drop layout'),
        ('test_cvt',       12, 'fish',       'equal',   'L_cvt_norm: uniform frames, fish layout'),
        ('test_fea',        8, 'leaf',       'equal',   'L_fea: uniform frames, leaf layout'),
        ('test_combined',  10, 'rectangle',  'mixed',   'All 3 losses: mixed masks, baseline control'),
    ]

    print("=" * 70)
    print("  Generating Publication-Ready Synthetic Test Cases")
    print("=" * 70)
    
    for case_name, n, layout_mask, style, desc in TEST_CASES:
        create_test_case(case_name, n, layout_mask, style, desc, OUTPUT_BASE)
        
    print("=" * 70)
    print(f"  Output saved to: {OUTPUT_BASE}")