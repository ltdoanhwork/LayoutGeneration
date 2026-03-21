"""
Debug visualization utilities for collage assembly.
Provides step-by-step visualization of the warp process.
"""

import cv2
import numpy as np
import os
from os.path import join


def create_debug_dir(base_output_dir):
    """Create debug visualization directory."""
    debug_dir = join(base_output_dir, "warp_debug_visualization")
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def save_step_debug(debug_dir, image_id, step_num, step_name, img, extra_info=""):
    """Save a single step visualization."""
    if debug_dir is None:
        return
    
    # Convert to BGR for saving
    if img is None:
        return
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = img.copy()
    
    # Add text overlay
    h, w = img_bgr.shape[:2]
    # Create header
    header_h = 40
    result = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    result[:header_h] = [40, 40, 40]
    result[header_h:] = img_bgr
    
    cv2.putText(result, f"STEP {step_num}: {step_name}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if extra_info:
        cv2.putText(result, extra_info, (10, h + header_h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    path = join(debug_dir, f"step{step_num}_{step_name.replace(' ', '_')}_{image_id}.png")
    cv2.imwrite(path, result)
    print(f"[DEBUG STEP {step_num}] {step_name}: {path}")


def visualize_saliency_map(image, saliency_map, debug_dir, image_id=""):
    """Save saliency map visualization."""
    # Normalize saliency for visualization
    sal_vis = (saliency_map * 255).astype(np.uint8)
    sal_colored = cv2.applyColorMap(sal_vis, cv2.COLORMAP_JET)
    
    # Blend with original
    if image.shape[2] == 4:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = image[:, :, :3]
    
    blended = cv2.addWeighted(image_bgr, 0.5, sal_colored, 0.5, 0)
    
    output_path = join(debug_dir, f"01_saliency_map_{image_id}.png")
    cv2.imwrite(output_path, blended)
    print(f"[DEBUG] Saved saliency map: {output_path}")


def visualize_mesh_grid(image, src_pts, debug_dir, image_id="", grid_size=16):
    """Visualize source mesh grid on original image."""
    if image.shape[2] == 4:
        viz_img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        viz_img = image.copy()
    
    # Draw mesh points
    for pt in src_pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < viz_img.shape[1] and 0 <= y < viz_img.shape[0]:
            cv2.circle(viz_img, (x, y), 3, (0, 255, 0), -1)
    
    # Draw mesh lines (grid)
    h, w = image.shape[:2]
    for y in range(0, h, grid_size):
        cv2.line(viz_img, (0, y), (w, y), (0, 200, 0), 1)
    for x in range(0, w, grid_size):
        cv2.line(viz_img, (x, 0), (x, h), (0, 200, 0), 1)
    
    output_path = join(debug_dir, f"02_mesh_grid_{image_id}.png")
    cv2.imwrite(output_path, viz_img)
    print(f"[DEBUG] Saved mesh grid: {output_path}")


def visualize_salient_regions(image, inner_src_box, inner_dest_box, outer_rectangle_dest, 
                             outer_rectangle_source, debug_dir, image_id=""):
    """Visualize salient regions and target boxes."""
    if image.shape[2] == 4:
        viz_img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        viz_img = image.copy()
    
    # Draw source salient box
    x1, y1, x2, y2 = int(inner_src_box[0]), int(inner_src_box[1]), int(inner_src_box[2]), int(inner_src_box[3])
    cv2.rectangle(viz_img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue: salient region
    
    # Draw source outer box
    outer_pts = np.array(outer_rectangle_source, dtype=np.int32)
    cv2.polylines(viz_img, [outer_pts], True, (0, 255, 0), 2)  # Green: outer box
    
    output_path = join(debug_dir, f"03_salient_regions_source_{image_id}.png")
    cv2.imwrite(output_path, viz_img)
    print(f"[DEBUG] Saved salient regions (source): {output_path}")


def visualize_mesh_transformation(src_pts, dst_pts_init, dst_pts_optimized, target_shape, 
                                 debug_dir, image_id=""):
    """Visualize mesh transformation from source to target."""
    h_dst, w_dst = target_shape
    
    # Create visualization canvas
    canvas = np.ones((h_dst, w_dst, 3), dtype=np.uint8) * 255
    
    # Draw initial destination points (magenta)
    for pt in dst_pts_init:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w_dst and 0 <= y < h_dst:
            cv2.circle(canvas, (x, y), 2, (255, 0, 255), -1)  # Magenta: initial
    
    # Draw optimized destination points (red)
    for pt in dst_pts_optimized:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w_dst and 0 <= y < h_dst:
            cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)  # Red: optimized
    
    # Draw connections from initial to optimized
    for init_pt, opt_pt in zip(dst_pts_init, dst_pts_optimized):
        x1, y1 = int(init_pt[0]), int(init_pt[1])
        x2, y2 = int(opt_pt[0]), int(opt_pt[1])
        if (0 <= x1 < w_dst and 0 <= y1 < h_dst and 
            0 <= x2 < w_dst and 0 <= y2 < h_dst):
            cv2.line(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1)
    
    output_path = join(debug_dir, f"04_mesh_transformation_{image_id}.png")
    cv2.imwrite(output_path, canvas)
    print(f"[DEBUG] Saved mesh transformation (magenta=initial, red=optimized): {output_path}")


def visualize_warped_result(original_image, warped_image, debug_dir, image_id="", 
                            polygon_coords=None, detection_box=None, dest_box=None):
    """
    Compare original vs warped image, and show final result cut into shape.
    
    Args:
        original_image: Original source image
        warped_image: Warped result image
        debug_dir: Debug directory
        image_id: Image identifier
        polygon_coords: Optional polygon coordinates (local coords) to show shape cutout
        detection_box: Optional [x1, y1, x2, y2] detection box in source image
        dest_box: Optional [x1, y1, x2, y2] destination box in warped image
    """
    if original_image.shape[2] == 4:
        orig_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
    else:
        orig_bgr = original_image[:, :, :3]
    
    if warped_image.shape[2] == 4:
        warp_bgr = cv2.cvtColor(warped_image, cv2.COLOR_RGBA2BGR)
    else:
        warp_bgr = warped_image[:, :, :3]
    
    h_warp, w_warp = warp_bgr.shape[:2]
    
    # Draw detection box on original (if provided)
    orig_with_box = orig_bgr.copy()
    if detection_box is not None:
        x1, y1, x2, y2 = [int(v) for v in detection_box]
        cv2.rectangle(orig_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(orig_with_box, "Detection", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw destination box on warped (if provided)
    warp_with_box = warp_bgr.copy()
    if dest_box is not None:
        x1, y1, x2, y2 = [int(v) for v in dest_box]
        cv2.rectangle(warp_with_box, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(warp_with_box, "Dest Box", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # Draw centroid crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(warp_with_box, (cx-20, cy), (cx+20, cy), (0, 0, 255), 2)
        cv2.line(warp_with_box, (cx, cy-20), (cx, cy+20), (0, 0, 255), 2)
    
    # Create shape cutout visualization (if polygon provided)
    shape_cutout = None
    if polygon_coords is not None and len(polygon_coords) > 2:
        # Create polygon mask
        mask = np.zeros((h_warp, w_warp), dtype=np.uint8)
        pts = np.array(polygon_coords, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Apply mask to warped image
        shape_cutout = warp_bgr.copy()
        shape_cutout[mask == 0] = [40, 40, 40]  # Dark gray background for non-shape areas
        
        # Draw polygon outline
        cv2.polylines(shape_cutout, [pts], True, (0, 255, 255), 2)  # Yellow outline
        
        # Mark shape centroid
        M = cv2.moments(pts)
        if M["m00"] > 0:
            shape_cx = int(M["m10"] / M["m00"])
            shape_cy = int(M["m01"] / M["m00"])
            cv2.circle(shape_cutout, (shape_cx, shape_cy), 8, (0, 0, 255), -1)
            cv2.putText(shape_cutout, "Shape Center", (shape_cx+10, shape_cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Resize original to match warped for comparison
    orig_resized = cv2.resize(orig_with_box, (w_warp, h_warp))
    
    # Create comparison image
    if shape_cutout is not None:
        # 3-panel: Original | Warped | Shape Cutout
        comparison = np.hstack([orig_resized, warp_with_box, shape_cutout])
        labels = ["ORIGINAL (detection)", "WARPED (dest box)", "SHAPE CUTOUT"]
    else:
        # 2-panel: Original | Warped
        comparison = np.hstack([orig_resized, warp_with_box])
        labels = ["ORIGINAL", "WARPED"]
    
    # Add labels at top
    label_height = 30
    labeled = np.zeros((comparison.shape[0] + label_height, comparison.shape[1], 3), dtype=np.uint8)
    labeled[label_height:] = comparison
    labeled[:label_height] = [50, 50, 50]  # Dark header
    
    # Add text labels
    panel_width = w_warp
    for i, label in enumerate(labels):
        x_pos = i * panel_width + 10
        cv2.putText(labeled, label, (x_pos, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    output_path = join(debug_dir, f"05_warped_result_{image_id}.png")
    cv2.imwrite(output_path, labeled)
    print(f"[DEBUG] Saved warped result: {output_path}")


def visualize_patch_placement(canvas, patch, patch_origin, image_id, debug_dir, step_num=8):
    """
    Visualize where a patch will be placed on the canvas.
    
    Args:
        canvas: Full canvas (can be current state with previous patches)
        patch: The patch to place
        patch_origin: (row_start, col_start) where patch will be placed
        image_id: Image identifier
        debug_dir: Debug output directory
        step_num: Step number for filename
    """
    if debug_dir is None:
        return
    
    row_start, col_start = patch_origin
    patch_h, patch_w = patch.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]
    
    # Create visualization
    if canvas.shape[2] == 4:
        canvas_vis = cv2.cvtColor(canvas.copy(), cv2.COLOR_RGBA2BGR)
    else:
        canvas_vis = canvas[:,:,:3].copy()
    
    # Draw where patch will go (RED rectangle)
    r_end = min(row_start + patch_h, canvas_h)
    c_end = min(col_start + patch_w, canvas_w)
    
    cv2.rectangle(canvas_vis, (col_start, row_start), (c_end, r_end), (0, 0, 255), 3)
    
    # Add info text
    info = f"Patch: {patch_w}x{patch_h} at ({col_start}, {row_start})"
    cv2.putText(canvas_vis, info, (col_start, row_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw crosshair at patch center
    cx = col_start + patch_w // 2
    cy = row_start + patch_h // 2
    cv2.line(canvas_vis, (cx - 30, cy), (cx + 30, cy), (255, 0, 0), 2)
    cv2.line(canvas_vis, (cx, cy - 30), (cx, cy + 30), (255, 0, 0), 2)
    
    path = join(debug_dir, f"step{step_num}_patch_placement_{image_id}.png")
    cv2.imwrite(path, canvas_vis)
    print(f"[DEBUG STEP {step_num}] Patch placement: {path}")


def visualize_incremental_composite(canvas, patches_placed, debug_dir, step_name="incremental"):
    """
    Save current state of canvas showing all patches placed so far.
    
    Args:
        canvas: Current canvas state
        patches_placed: Number of patches placed
        debug_dir: Debug output directory
        step_name: Name for the step
    """
    if debug_dir is None:
        return
    
    if canvas.shape[2] == 4:
        canvas_vis = cv2.cvtColor(canvas.copy(), cv2.COLOR_RGBA2BGR)
    else:
        canvas_vis = canvas[:,:,:3].copy()
    
    # Add info overlay
    info = f"Patches placed: {patches_placed}"
    cv2.putText(canvas_vis, info, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    path = join(debug_dir, f"composite_{step_name}_{patches_placed:03d}.png")
    cv2.imwrite(path, canvas_vis)
    print(f"[DEBUG] Saved incremental composite ({patches_placed} patches): {path}")
