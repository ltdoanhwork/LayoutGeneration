#!/usr/bin/env python3
"""
Advanced BBox Refinement with:
- Radial expansion from center (gaussian weighting)
- KDE or Meanshift for threshold detection
- Distance-based weighting (suppress far from center)
- Selective percentile (not taking all details)
- HSV normalization for color analysis
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.ndimage import label as scipy_label
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.cluster import MeanShift, estimate_bandwidth


class BBoxRefinement:
    """Refine bounding boxes based on visual importance with advanced methods"""
    
    def __init__(self, device="cuda"):
        self.device = device
    
    def refine_bbox(self, image, bbox, weights=(0.4, 0.4, 0.2), 
                    lambda_=10, eta=0.9, tau=30, expansion_margin=0.1,
                    search_margin=0.2, use_kde=True, distance_weight=True,
                    percentile_threshold=75, use_mean_shift=True):
        """
        Advanced BBox refinement with radial expansion and KDE thresholding
        
        Args:
            image: numpy array (H, W, 3) in BGR
            bbox: [x1, y1, x2, y2] initial bbox from SAM2
            weights: (w_entropy, w_color, w_edge)
            lambda_: temperature for softmax
            eta: coverage threshold (0.9 = top 90%)
            tau: edge detection threshold
            expansion_margin: context margin
            search_margin: margin to search beyond bbox
            use_kde: use KDE for threshold detection
            distance_weight: apply gaussian weighting from bbox center
            percentile_threshold: if not use_kde, use this percentile
            use_mean_shift: use mean shift clustering to grow/shrink per direction
        
        Returns:
            refined_bbox, importance_map, info dict
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h_img, w_img = image.shape[:2]
        
        # Ensure bbox is within image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)
        
        # Step 0: Create search region
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0
        
        search_margin_x = int(bbox_w * search_margin)
        search_margin_y = int(bbox_h * search_margin)
        
        search_x1 = max(0, x1 - search_margin_x)
        search_y1 = max(0, y1 - search_margin_y)
        search_x2 = min(w_img, x2 + search_margin_x)
        search_y2 = min(h_img, y2 + search_margin_y)
        
        # Step 1: Extract search region
        search_region = image[search_y1:search_y2, search_x1:search_x2].copy()
        h, w = search_region.shape[:2]
        
        # Calculate offset of original bbox within search region
        bbox_offset_x = x1 - search_x1
        bbox_offset_y = y1 - search_y1
        bbox_in_search = [
            bbox_offset_x, 
            bbox_offset_y, 
            bbox_offset_x + bbox_w, 
            bbox_offset_y + bbox_h
        ]
        
        # Create mask for original bbox area
        mask = np.zeros((h, w), dtype=np.float32)
        mask[
            max(0, int(bbox_in_search[1])):min(h, int(bbox_in_search[3])),
            max(0, int(bbox_in_search[0])):min(w, int(bbox_in_search[2]))
        ] = 1.0
        
        # Step 1b: Create distance map from bbox center (gaussian weighting)
        if distance_weight:
            center_x = (bbox_in_search[0] + bbox_in_search[2]) / 2.0
            center_y = (bbox_in_search[1] + bbox_in_search[3]) / 2.0
            
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            dist_map = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
            
            # Gaussian decay from center
            max_dist = np.sqrt(w**2 + h**2) / 2.0
            gaussian_decay = np.exp(-(dist_map**2) / (2 * (max_dist/2)**2))
        else:
            gaussian_decay = np.ones((h, w), dtype=np.float32)
        
        # Step 2: Compute feature maps
        region = search_region
        entropy_map = self._compute_entropy_map(region, window_size=20)
        color_var_map = self._compute_color_variance_map_hsv(region, window_size=20)
        edge_map = self._compute_edge_density_map(region, tau=tau)
        
        # Step 3: Apply mask + distance decay
        # Inside bbox: 2.0x, outside: 0.3x, then multiply by gaussian decay
        importance_boost = (mask * 2.0 + (1 - mask) * 0.3) * gaussian_decay
        
        # Step 4: Normalize to [0, 1]
        entropy_norm = self._normalize(entropy_map) * importance_boost
        color_norm = self._normalize(color_var_map) * importance_boost
        edge_norm = self._normalize(edge_map) * importance_boost
        
        # Step 5: Fuse with weights
        w_e, w_v, w_d = weights
        importance_map = w_e * entropy_norm + w_v * color_norm + w_d * edge_norm
        
        # Normalize final map
        importance_map = self._normalize(importance_map)
        
        # Step 6: Apply thresholding (KDE or percentile-based)
        if use_kde:
            threshold = self._compute_kde_threshold(importance_map)
        else:
            threshold = np.percentile(importance_map[mask > 0.5], percentile_threshold)
        
        # Binary mask of important pixels
        important_pixels = (importance_map >= threshold).astype(np.float32)
        
        # Step 7: Find connected components (to handle isolated pixels)
        labeled, num_features = scipy_label(important_pixels)
        
        # Keep only largest component(s) that contain bbox center
        if num_features > 0:
            center_label = labeled[
                int((bbox_in_search[1] + bbox_in_search[3]) / 2),
                int((bbox_in_search[0] + bbox_in_search[2]) / 2)
            ]
            if center_label > 0:
                important_pixels = (labeled == center_label).astype(np.float32)
        
        # Step 8: Get coordinates of important pixels
        coords = np.argwhere(important_pixels > 0.5)
        
        mean_shift_used = False
        if len(coords) > 0 and use_mean_shift:
            # Use mean shift to find optimal expansion directions
            min_y, min_x, max_y, max_x = self._mean_shift_bbox_refinement(
                coords, importance_map, 
                bbox_in_search,
                h, w
            )
            mean_shift_used = True
        elif len(coords) > 0:
            # Simple min/max approach
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
        else:
            # Fallback: use original bbox
            min_y, min_x = int(bbox_in_search[1]), int(bbox_in_search[0])
            max_y, max_x = int(bbox_in_search[3]), int(bbox_in_search[2])
        
        # Step 9: Ensure we cover original bbox
        min_x = min(min_x, int(bbox_in_search[0]))
        min_y = min(min_y, int(bbox_in_search[1]))
        max_x = max(max_x, int(bbox_in_search[2]))
        max_y = max(max_y, int(bbox_in_search[3]))
        
        # Step 10: Add context margin
        margin_x = int((max_x - min_x) * expansion_margin)
        margin_y = int((max_y - min_y) * expansion_margin)
        
        min_x = max(0, min_x - margin_x)
        max_x = min(w - 1, max_x + margin_x)
        min_y = max(0, min_y - margin_y)
        max_y = min(h - 1, max_y + margin_y)
        
        # Convert back to image coordinates
        refined_bbox = [
            search_x1 + min_x,
            search_y1 + min_y,
            search_x1 + max_x,
            search_y1 + max_y
        ]
        
        info = {
            'original_bbox': bbox,
            'search_region': [search_x1, search_y1, search_x2, search_y2],
            'refined_bbox': refined_bbox,
            'expansion_ratio': (
                (refined_bbox[2] - refined_bbox[0]) * (refined_bbox[3] - refined_bbox[1]) /
                ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            ) if ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) > 0 else 1.0,
            'kde_threshold': float(threshold),
            'important_pixel_count': int(len(coords)),
            'eta': eta,
            'use_kde': use_kde,
            'mean_shift_used': mean_shift_used,
        }
        
        return refined_bbox, importance_map, info
    
    def _mean_shift_bbox_refinement(self, coords, importance_map, 
                                    bbox_in_search, h, w, n_clusters=6):
        """
        Use mean shift clustering to intelligently grow/shrink bbox
        
        Co/giãn theo 4 hướng (N/S/E/W) dựa trên high-value clusters
        """
        
        if len(coords) < 5:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            return min_y, min_x, max_y, max_x
        
        # Step 1: Subsample coords for speed (max 500 points)
        if len(coords) > 500:
            indices = np.random.choice(len(coords), 500, replace=False)
            coords_sample = coords[indices]
            importance_sample = np.array([importance_map[c[0], c[1]] for c in coords[indices]])
        else:
            coords_sample = coords
            importance_sample = np.array([importance_map[c[0], c[1]] for c in coords])
        
        # Step 2: Quick bandwidth estimation
        bandwidth = max(5, int(np.std(coords_sample) / 2))
        
        # Step 3: Apply mean shift (fast version)
        try:
            ms = MeanShift(bandwidth=bandwidth, n_iter=20)
            labels = ms.fit_predict(coords_sample)
            cluster_centers = ms.cluster_centers_
        except:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            return min_y, min_x, max_y, max_x
        
        # Step 4: Get bbox center
        bbox_cx = (bbox_in_search[0] + bbox_in_search[2]) / 2.0
        bbox_cy = (bbox_in_search[1] + bbox_in_search[3]) / 2.0
        
        # Step 5: Find extreme points in each direction
        extremes = {
            'top': bbox_in_search[1],      # y_min
            'bottom': bbox_in_search[3],   # y_max
            'left': bbox_in_search[0],     # x_min
            'right': bbox_in_search[2],    # x_max
        }
        
        # For each cluster, weight by importance and find direction extremes
        for idx, center in enumerate(cluster_centers):
            cy, cx = center
            
            # Weight by importance of this cluster
            mask = (labels == idx)
            if mask.sum() == 0:
                continue
            
            weight = np.mean(importance_sample[mask])
            
            # Find extremes in this cluster
            cluster_coords = coords_sample[mask]
            cluster_cy = cluster_coords[:, 0]
            cluster_cx = cluster_coords[:, 1]
            
            # Top: high importance → extend; low importance → shrink
            if cy < bbox_cy:
                y_top = cluster_cy.min()
                extremes['top'] = min(extremes['top'], int(y_top)) if weight > 0.3 else max(extremes['top'], int(y_top))
            
            # Bottom
            if cy > bbox_cy:
                y_bottom = cluster_cy.max()
                extremes['bottom'] = max(extremes['bottom'], int(y_bottom)) if weight > 0.3 else min(extremes['bottom'], int(y_bottom))
            
            # Left
            if cx < bbox_cx:
                x_left = cluster_cx.min()
                extremes['left'] = min(extremes['left'], int(x_left)) if weight > 0.3 else max(extremes['left'], int(x_left))
            
            # Right
            if cx > bbox_cx:
                x_right = cluster_cx.max()
                extremes['right'] = max(extremes['right'], int(x_right)) if weight > 0.3 else min(extremes['right'], int(x_right))
        
        # Step 6: Ensure cover original bbox
        min_y = min(extremes['top'], int(bbox_in_search[1]))
        max_y = max(extremes['bottom'], int(bbox_in_search[3]))
        min_x = min(extremes['left'], int(bbox_in_search[0]))
        max_x = max(extremes['right'], int(bbox_in_search[2]))
        
        return min_y, min_x, max_y, max_x
    
    def _compute_entropy_map(self, region, window_size=20):
        """Compute local Shannon entropy - optimized"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Use block histogram instead of pixel-wise for speed
        entropy_map = np.zeros((h, w), dtype=np.float32)
        pad = window_size // 2
        
        gray_padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        
        # Subsample computation for speed
        step = max(1, window_size // 4)
        
        for i in range(0, h, step):
            for j in range(0, w, step):
                window = gray_padded[i:i+window_size, j:j+window_size]
                hist, _ = np.histogram(window, bins=128, range=(0, 256))
                hist = hist / (window_size * window_size + 1e-10)
                entropy_val = -np.sum(hist * np.log2(hist + 1e-10))
                entropy_map[i:i+step, j:j+step] = entropy_val
        
        # Interpolate to fill gaps
        if step > 1:
            entropy_map = cv2.resize(entropy_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return entropy_map
    
    def _compute_color_variance_map(self, region, window_size=20):
        """Compute color variance in HSV space"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, w = region.shape[:2]
        var_map = np.zeros((h, w), dtype=np.float32)
        
        pad = window_size // 2
        hsv_padded = np.pad(hsv, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        
        for i in range(h):
            for j in range(w):
                window = hsv_padded[i:i+window_size, j:j+window_size, :]
                var_sum = sum([window[:,:,c].var() for c in range(3)])
                var_map[i, j] = var_sum / 3.0
        
        return var_map
    
    def _compute_color_variance_map_hsv(self, region, window_size=20):
        """
        Compute normalized color variance in HSV space - optimized
        """
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, w = region.shape[:2]
        var_map = np.zeros((h, w), dtype=np.float32)
        
        pad = window_size // 2
        hsv_padded = np.pad(hsv, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        
        # Subsample for speed
        step = max(1, window_size // 4)
        
        for i in range(0, h, step):
            for j in range(0, w, step):
                window = hsv_padded[i:i+window_size, j:j+window_size, :]
                
                # Hue: normalize and handle circularity
                hue = window[:,:,0] / 180.0
                hue_var = np.var(np.sin(hue * 2 * np.pi)) + np.var(np.cos(hue * 2 * np.pi))
                
                # Saturation and Value
                sat_var = np.var(window[:,:,1] / 255.0)
                val_var = np.var(window[:,:,2] / 255.0)
                
                var_val = 0.2 * hue_var + 0.4 * sat_var + 0.4 * val_var
                var_map[i:i+step, j:j+step] = var_val
        
        # Interpolate
        if step > 1:
            var_map = cv2.resize(var_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return var_map
    
    def _compute_edge_density_map(self, region, tau=30):
        """Compute edge density using Sobel"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_map = (magnitude > tau).astype(np.float32)
        return edge_map
    
    def _normalize(self, x):
        """Normalize to [0, 1]"""
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + 1e-8)
    
    def _compute_kde_threshold(self, importance_map, bw_method=0.15):
        """
        Compute threshold using KDE + peak detection
        
        Idea: histogram of importance values, smooth with KDE,
        find local minima to separate foreground from background
        """
        values = importance_map.flatten()
        
        # Fit KDE
        kde = gaussian_kde(values, bw_method=bw_method)
        
        # Evaluate KDE on grid
        x_eval = np.linspace(0, 1, 500)
        density = kde(x_eval)
        
        # Find peaks in KDE (local maxima)
        peaks, _ = find_peaks(density, height=0)
        
        if len(peaks) > 1:
            # Find valley (local minimum) between first two peaks
            peak1, peak2 = peaks[0], peaks[1]
            valley_region = x_eval[peak1:peak2]
            valley_density = density[peak1:peak2]
            valley_idx = np.argmin(valley_density)
            threshold = valley_region[valley_idx]
        else:
            # Fallback: use 70th percentile
            threshold = np.percentile(values, 70)
        
        return float(threshold)
    
    def visualize_refinement(self, image, original_bbox, refined_bbox, 
                            importance_map, output_path=None):
        """
        Create visualization showing:
        1. Original bbox
        2. Importance heatmap
        3. Refined bbox
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Image with original bbox
        img_orig = image.copy()
        x1, y1, x2, y2 = [int(c) for c in original_bbox]
        cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green
        axes[0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Original BBox\n({x2-x1}x{y2-y1})')
        axes[0].axis('off')
        
        # Heatmap
        heatmap_colored = cv2.applyColorMap(
            (importance_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        axes[1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Importance Heatmap')
        axes[1].axis('off')
        
        # Image with refined bbox
        img_refined = image.copy()
        x1, y1, x2, y2 = [int(c) for c in refined_bbox]
        cv2.rectangle(img_refined, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red
        axes[2].imshow(cv2.cvtColor(img_refined, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Refined BBox\n({x2-x1}x{y2-y1})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"[VIZ] Saved visualization to {output_path}")
        
        plt.close()
    
    def create_heatmap_overlay(self, image, importance_map, output_path=None):
        """Create heatmap overlay on image"""
        # Normalize importance map to match region size
        importance_norm = (importance_map * 255).astype(np.uint8)
        
        # Resize heatmap to match image if needed
        if importance_norm.shape[:2] != image.shape[:2]:
            importance_norm = cv2.resize(
                importance_norm, 
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Ensure 3 channels
        if len(importance_norm.shape) == 2:
            importance_norm = cv2.cvtColor(importance_norm, cv2.COLOR_GRAY2BGR)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(importance_norm, cv2.COLORMAP_JET)
        
        # Blend with original
        overlay = cv2.addWeighted(image, 0.7, heatmap_colored, 0.3, 0)
        
        if output_path:
            cv2.imwrite(output_path, overlay)
            print(f"[HEATMAP] Saved to {output_path}")
        
        return overlay


def test_bbox_refinement():
    """Test advanced refinement on a sample image"""
    import glob
    from pathlib import Path
    
    # Get a test image
    test_folder = "data/samples/keyframe4check/23670_keyframes"
    images = sorted(glob.glob(f"{test_folder}/*.jpg"))[:1]  # First image
    
    if not images:
        print("[ERROR] No test images found")
        return
    
    img_path = images[0]
    print(f"[TEST] Testing on: {img_path}")
    
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    # Create dummy bboxes (simulate SAM2 detections)
    test_bboxes = [
        [int(w*0.1), int(h*0.1), int(w*0.4), int(h*0.4)],
        [int(w*0.5), int(h*0.2), int(w*0.9), int(h*0.6)],
    ]
    
    refiner = BBoxRefinement()
    
    # Create output dir
    output_dir = "debug_output_bbox_refinement_advanced"
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPARING REFINEMENT STRATEGIES")
    print("="*80)
    
    strategies = [
        {
            'name': 'Mean Shift + Distance',
            'params': {
                'weights': (0.4, 0.4, 0.2),
                'use_kde': True,
                'distance_weight': True,
                'use_mean_shift': True,
                'expansion_margin': 0.15,
                'search_margin': 0.25,
            }
        },
        {
            'name': 'Advanced (KDE + Distance)',
            'params': {
                'weights': (0.4, 0.4, 0.2),
                'use_kde': True,
                'distance_weight': True,
                'use_mean_shift': False,
                'expansion_margin': 0.15,
                'search_margin': 0.25,
            }
        },
        {
            'name': 'Conservative (P75)',
            'params': {
                'weights': (0.4, 0.4, 0.2),
                'use_kde': False,
                'distance_weight': True,
                'use_mean_shift': False,
                'percentile_threshold': 75,
                'expansion_margin': 0.1,
                'search_margin': 0.2,
            }
        },
        {
            'name': 'Aggressive (P60)',
            'params': {
                'weights': (0.4, 0.4, 0.2),
                'use_kde': False,
                'distance_weight': True,
                'use_mean_shift': False,
                'percentile_threshold': 60,
                'expansion_margin': 0.2,
                'search_margin': 0.3,
            }
        },
    ]
    
    for bbox_idx, bbox in enumerate(test_bboxes):
        print(f"\n{'─'*80}")
        print(f"BBox {bbox_idx}: {bbox}")
        print(f"{'─'*80}")
        
        results_for_bbox = []
        
        for strategy in strategies:
            strat_name = strategy['name']
            params = strategy['params']
            
            refined, heatmap, info = refiner.refine_bbox(image, bbox, **params)
            
            results_for_bbox.append({
                'name': strat_name,
                'refined': refined,
                'heatmap': heatmap,
                'info': info,
            })
            
            print(f"\n  [{strat_name}]")
            print(f"    Refined: {refined}")
            print(f"    Expansion: {info['expansion_ratio']:.2f}x")
            print(f"    KDE Threshold: {info['kde_threshold']:.3f}")
            print(f"    Important pixels: {info['important_pixel_count']}")
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"BBox {bbox_idx} - Refinement Strategies Comparison", fontsize=16, fontweight='bold')
        
        for idx, result in enumerate(results_for_bbox):
            ax = axes[idx // 2, idx % 2]
            
            # Draw image with refined bbox
            img_viz = image.copy()
            x1, y1, x2, y2 = [int(c) for c in result['refined']]
            cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw original bbox
            ox1, oy1, ox2, oy2 = [int(c) for c in bbox]
            cv2.rectangle(img_viz, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
            
            ax.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
            title = f"{result['name']}\nExpansion: {result['info']['expansion_ratio']:.2f}x"
            ax.set_title(title, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        viz_path = f"{output_dir}/{bbox_idx:02d}_comparison.png"
        plt.savefig(viz_path, dpi=100, bbox_inches='tight')
        print(f"\n  💾 Saved comparison: {viz_path}")
        plt.close()
    
    print(f"\n{'='*80}")
    print(f"✅ DONE - Output directory: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_bbox_refinement()
