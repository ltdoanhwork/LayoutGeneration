"""
Seam Carving algorithm for content-aware image resizing.
Based on the reference implementation, optimized for collage assembly.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class SeamCarver:
    """
    Content-aware image resizing using seam carving.
    Supports both shrinking and expanding with optional protection masks.
    """
    
    def __init__(self, image: np.ndarray, out_height: int, out_width: int, 
                 protect_mask: np.ndarray = None):
        """
        Initialize seam carver.
        
        Args:
            image: Input image (BGR or RGBA), will be converted to float64
            out_height: Target height
            out_width: Target width
            protect_mask: Optional mask (same size as image) where 255 = protected areas
        """
        # Handle RGBA images
        if image.shape[2] == 4:
            self.has_alpha = True
            self.in_image = image[:, :, :3].astype(np.float64)
            self.alpha_channel = image[:, :, 3].astype(np.float64)
        else:
            self.has_alpha = False
            self.in_image = image.astype(np.float64)
            self.alpha_channel = None
        
        self.in_height, self.in_width = self.in_image.shape[:2]
        self.out_height = out_height
        self.out_width = out_width
        
        # Working copy
        self.out_image = np.copy(self.in_image)
        if self.has_alpha:
            self.out_alpha = np.copy(self.alpha_channel)
        
        # Protection mask
        self.protect = protect_mask is not None
        if self.protect:
            self.mask = protect_mask.astype(np.float64)
        else:
            self.mask = None
        
        # Energy constant for protected areas
        self.constant = 1000
        
    def seams_carving(self) -> np.ndarray:
        """
        Perform seam carving to resize image.
        
        Returns:
            Resized image
        """
        # Calculate needed changes
        delta_row = int(self.out_height - self.in_height)
        delta_col = int(self.out_width - self.in_width)
        
        # Process columns (width)
        if delta_col < 0:
            self._seams_removal_horizontal(abs(delta_col))
        elif delta_col > 0:
            self._seams_insertion_horizontal(delta_col)
        
        # Process rows (height) - rotate, process, rotate back
        if delta_row < 0:
            self.out_image = self._rotate_image(self.out_image, ccw=True)
            if self.has_alpha:
                self.out_alpha = self._rotate_mask(self.out_alpha, ccw=True)
            if self.protect and self.mask is not None:
                self.mask = self._rotate_mask(self.mask, ccw=True)
            self._seams_removal_horizontal(abs(delta_row))
            self.out_image = self._rotate_image(self.out_image, ccw=False)
            if self.has_alpha:
                self.out_alpha = self._rotate_mask(self.out_alpha, ccw=False)
        elif delta_row > 0:
            self.out_image = self._rotate_image(self.out_image, ccw=True)
            if self.has_alpha:
                self.out_alpha = self._rotate_mask(self.out_alpha, ccw=True)
            if self.protect and self.mask is not None:
                self.mask = self._rotate_mask(self.mask, ccw=True)
            self._seams_insertion_horizontal(delta_row)
            self.out_image = self._rotate_image(self.out_image, ccw=False)
            if self.has_alpha:
                self.out_alpha = self._rotate_mask(self.out_alpha, ccw=False)
        
        # Recombine with alpha if needed
        if self.has_alpha:
            result = np.zeros((self.out_image.shape[0], self.out_image.shape[1], 4), dtype=np.uint8)
            result[:, :, :3] = np.clip(self.out_image, 0, 255).astype(np.uint8)
            result[:, :, 3] = np.clip(self.out_alpha, 0, 255).astype(np.uint8)
            return result
        else:
            return np.clip(self.out_image, 0, 255).astype(np.uint8)
    
    def _calc_energy_map(self) -> np.ndarray:
        """Calculate energy map using gradient magnitude."""
        b, g, r = cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy
    
    def _cumulative_map_forward(self, energy_map: np.ndarray) -> np.ndarray:
        """Compute cumulative energy map using forward energy."""
        m, n = energy_map.shape
        output = np.copy(energy_map)
        
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    output[row, col] = energy_map[row, col] + min(
                        output[row - 1, col],
                        output[row - 1, col + 1]
                    )
                elif col == n - 1:
                    output[row, col] = energy_map[row, col] + min(
                        output[row - 1, col - 1],
                        output[row - 1, col]
                    )
                else:
                    output[row, col] = energy_map[row, col] + min(
                        output[row - 1, col - 1],
                        output[row - 1, col],
                        output[row - 1, col + 1]
                    )
        return output
    
    def _cumulative_map_backward(self, energy_map: np.ndarray) -> np.ndarray:
        """Compute cumulative energy map using backward energy."""
        m, n = energy_map.shape
        output = np.copy(energy_map)
        
        for row in range(1, m):
            for col in range(n):
                output[row, col] = energy_map[row, col] + np.amin(
                    output[row - 1, max(col - 1, 0): min(col + 2, n)]
                )
        return output
    
    def _find_seam(self, cumulative_map: np.ndarray) -> np.ndarray:
        """Find minimum energy seam."""
        m, n = cumulative_map.shape
        seam = np.zeros(m, dtype=np.int32)
        seam[-1] = np.argmin(cumulative_map[-1])
        
        for row in range(m - 2, -1, -1):
            prev_col = seam[row + 1]
            left = max(prev_col - 1, 0)
            right = min(prev_col + 2, n)
            seam[row] = left + np.argmin(cumulative_map[row, left:right])
        
        return seam
    
    def _delete_seam(self, seam: np.ndarray):
        """Remove a seam from the image."""
        m, n = self.out_image.shape[:2]
        output = np.zeros((m, n - 1, 3), dtype=np.float64)
        
        for row in range(m):
            col = seam[row]
            output[row, :col, :] = self.out_image[row, :col, :]
            output[row, col:, :] = self.out_image[row, col + 1:, :]
        
        self.out_image = output
        
        # Also delete from alpha channel
        if self.has_alpha:
            alpha_out = np.zeros((m, n - 1), dtype=np.float64)
            for row in range(m):
                col = seam[row]
                alpha_out[row, :col] = self.out_alpha[row, :col]
                alpha_out[row, col:] = self.out_alpha[row, col + 1:]
            self.out_alpha = alpha_out
    
    def _delete_seam_on_mask(self, seam: np.ndarray):
        """Remove a seam from the protection mask."""
        if self.mask is None:
            return
        m, n = self.mask.shape
        output = np.zeros((m, n - 1), dtype=np.float64)
        
        for row in range(m):
            col = seam[row]
            output[row, :col] = self.mask[row, :col]
            output[row, col:] = self.mask[row, col + 1:]
        
        self.mask = output
    
    def _add_seam(self, seam: np.ndarray):
        """Insert a seam into the image."""
        m, n = self.out_image.shape[:2]
        output = np.zeros((m, n + 1, 3), dtype=np.float64)
        
        for row in range(m):
            col = seam[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.out_image[row, col:col + 2, ch])
                    output[row, col, ch] = self.out_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 2:, ch] = self.out_image[row, col + 1:, ch]
                else:
                    p = np.average(self.out_image[row, col - 1:col + 1, ch])
                    output[row, :col, ch] = self.out_image[row, :col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
        
        self.out_image = output
        
        # Also add to alpha channel
        if self.has_alpha:
            alpha_out = np.zeros((m, n + 1), dtype=np.float64)
            for row in range(m):
                col = seam[row]
                if col == 0:
                    p = np.average(self.out_alpha[row, col:col + 2])
                    alpha_out[row, col] = self.out_alpha[row, col]
                    alpha_out[row, col + 1] = p
                    alpha_out[row, col + 2:] = self.out_alpha[row, col + 1:]
                else:
                    p = np.average(self.out_alpha[row, col - 1:col + 1])
                    alpha_out[row, :col] = self.out_alpha[row, :col]
                    alpha_out[row, col] = p
                    alpha_out[row, col + 1:] = self.out_alpha[row, col:]
            self.out_alpha = alpha_out
    
    def _add_seam_on_mask(self, seam: np.ndarray):
        """Insert a seam into the protection mask."""
        if self.mask is None:
            return
        m, n = self.mask.shape
        output = np.zeros((m, n + 1), dtype=np.float64)
        
        for row in range(m):
            col = seam[row]
            if col == 0:
                p = np.average(self.mask[row, col:col + 2])
                output[row, col] = self.mask[row, col]
                output[row, col + 1] = p
                output[row, col + 2:] = self.mask[row, col + 1:]
            else:
                p = np.average(self.mask[row, col - 1:col + 1])
                output[row, :col] = self.mask[row, :col]
                output[row, col] = p
                output[row, col + 1:] = self.mask[row, col:]
        
        self.mask = output
    
    def _update_seams(self, remaining_seams: list, current_seam: np.ndarray) -> list:
        """Update seam indices after insertion."""
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output
    
    def _seams_removal_horizontal(self, num_seams: int):
        """Remove seams to reduce width."""
        for _ in range(num_seams):
            energy_map = self._calc_energy_map()
            if self.protect and self.mask is not None:
                energy_map[self.mask > 0] *= self.constant
            cumulative_map = self._cumulative_map_forward(energy_map)
            seam = self._find_seam(cumulative_map)
            self._delete_seam(seam)
            if self.protect:
                self._delete_seam_on_mask(seam)
    
    def _seams_insertion_horizontal(self, num_seams: int):
        """Insert seams to increase width."""
        temp_image = np.copy(self.out_image)
        temp_alpha = np.copy(self.out_alpha) if self.has_alpha else None
        temp_mask = np.copy(self.mask) if self.protect else None
        seams_record = []
        
        # Find seams to duplicate
        for _ in range(num_seams):
            energy_map = self._calc_energy_map()
            if self.protect and self.mask is not None:
                energy_map[self.mask > 0] *= self.constant
            cumulative_map = self._cumulative_map_backward(energy_map)
            seam = self._find_seam(cumulative_map)
            seams_record.append(seam)
            self._delete_seam(seam)
            if self.protect:
                self._delete_seam_on_mask(seam)
        
        # Restore and insert seams
        self.out_image = temp_image
        if self.has_alpha:
            self.out_alpha = temp_alpha
        if self.protect:
            self.mask = temp_mask
        
        for _ in range(len(seams_record)):
            seam = seams_record.pop(0)
            self._add_seam(seam)
            if self.protect:
                self._add_seam_on_mask(seam)
            seams_record = self._update_seams(seams_record, seam)
    
    def _rotate_image(self, image: np.ndarray, ccw: bool = True) -> np.ndarray:
        """Rotate image 90 degrees."""
        if ccw:
            return np.rot90(image, k=1)
        else:
            return np.rot90(image, k=-1)
    
    def _rotate_mask(self, mask: np.ndarray, ccw: bool = True) -> np.ndarray:
        """Rotate 2D mask 90 degrees."""
        if ccw:
            return np.rot90(mask, k=1)
        else:
            return np.rot90(mask, k=-1)


def seam_carve(image: np.ndarray, target_height: int, target_width: int,
               protect_mask: np.ndarray = None) -> np.ndarray:
    """
    Resize image using seam carving.
    
    Args:
        image: Input image (BGR or RGBA)
        target_height: Target height
        target_width: Target width
        protect_mask: Optional protection mask (255 = protected)
        
    Returns:
        Resized image
    """
    carver = SeamCarver(image, target_height, target_width, protect_mask)
    return carver.seams_carving()


def seam_carve_reduce_gap(image: np.ndarray, 
                          detections: list,
                          gap_regions: list,
                          reduction_ratio: float = 0.25,  # Conservative - remove only 25% of gap
                          min_gap_ratio: float = 0.45,    # Only carve if gap > 45% of width
                          max_reduction_ratio: float = 0.12) -> Tuple[np.ndarray, float]:  # Max 12% of width
    """
    Apply seam carving to reduce gaps between detected objects.
    
    NEW LOGIC (CONSERVATIVE): 
    - Only trigger if gap is > 45% of image width
    - Reduce proportionally to gap ratio, max 12% of width
    - This prevents over-aggressive seam carving that creates visible artifacts
    
    Args:
        image: Input image
        detections: List of detection dicts with 'bbox' key
        gap_regions: List of gap dicts from analyze_box_distribution()
        reduction_ratio: How much of gap to remove (0-1), conservative at 0.25
        min_gap_ratio: Min gap-to-width ratio to trigger carving (default 0.45 = 45%)
        max_reduction_ratio: Maximum reduction as ratio of image width (default 12%)
        
    Returns:
        Tuple of (carved_image, scale_factor)
    """
    h, w = image.shape[:2]
    
    if not gap_regions:
        return image, 1.0
    
    # Calculate total gap
    total_gap = sum(g['gap_size'] for g in gap_regions)
    gap_to_width_ratio = total_gap / w
    
    # CONSERVATIVE: Only proceed if gap is significant (> 45% of width)
    if gap_to_width_ratio < min_gap_ratio:
        print(f"[SeamCarving] Gap {total_gap}px = {gap_to_width_ratio*100:.1f}% of width < {min_gap_ratio*100:.0f}% threshold, skipping")
        return image, 1.0
    
    # Calculate reduction based on gap-to-width ratio
    # Only remove reduction_ratio (25%) of the excess gap above threshold
    excess_ratio = gap_to_width_ratio - min_gap_ratio
    target_reduction_ratio = excess_ratio * reduction_ratio
    
    # Cap at max_reduction_ratio (default 12%)
    target_reduction_ratio = min(target_reduction_ratio, max_reduction_ratio)
    
    # Minimum meaningful reduction
    min_reduction_px = 30
    target_reduction = int(w * target_reduction_ratio)
    
    if target_reduction < min_reduction_px:
        print(f"[SeamCarving] Calculated reduction {target_reduction}px < {min_reduction_px}px min, skipping")
        return image, 1.0
    
    target_width = w - target_reduction
    
    print(f"[SeamCarving] Gap analysis:")
    print(f"  - Total gap: {total_gap}px ({gap_to_width_ratio*100:.1f}% of {w}px)")
    print(f"  - Min gap threshold: {min_gap_ratio*100:.0f}%")
    print(f"  - Excess gap: {(gap_to_width_ratio - min_gap_ratio)*100:.1f}%")
    print(f"  - Reduction: {target_reduction}px ({target_reduction_ratio*100:.1f}% of width)")
    print(f"  - New width: {w}px -> {target_width}px")
    
    # Create protection mask for detected objects
    padding = 20  # Protect detected objects with padding
    protect_mask = np.zeros((h, w), dtype=np.float64)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        protect_mask[y1:y2, x1:x2] = 255
    
    # Apply seam carving
    carved = seam_carve(image, h, target_width, protect_mask)
    scale_factor = target_width / w
    
    print(f"[SeamCarving] Done! Scale factor: {scale_factor:.3f}")
    
    return carved, scale_factor


def compute_box_shift_after_carving(original_detections: list,
                                     original_width: int,
                                     carved_width: int,
                                     gap_regions: list) -> list:
    """
    Estimate new box positions after seam carving.
    
    Since seam carving removes pixels from gap regions (low energy),
    boxes after gaps should shift left proportionally.
    
    Args:
        original_detections: Original detection list
        original_width: Original image width
        carved_width: Width after carving
        gap_regions: Gap regions that were carved
        
    Returns:
        Updated detections with adjusted bboxes
    """
    if not gap_regions:
        return original_detections
    
    total_removed = original_width - carved_width
    
    # Sort detections by x position
    sorted_dets = sorted(original_detections, key=lambda d: d['center'][0])
    
    # Calculate cumulative shift for each detection
    updated_dets = []
    cumulative_shift = 0
    gap_idx = 0
    sorted_gaps = sorted(gap_regions, key=lambda g: g['region'][0])
    
    for det in sorted_dets:
        det_copy = dict(det)
        x1, y1, x2, y2 = det['bbox']
        cx = det['center'][0]
        
        # Count gaps to the left of this detection
        while gap_idx < len(sorted_gaps) and sorted_gaps[gap_idx]['region'][2] < cx:
            # This gap is to the left, estimate shift
            gap_contribution = sorted_gaps[gap_idx]['gap_size'] / sum(g['gap_size'] for g in gap_regions)
            cumulative_shift += int(total_removed * gap_contribution)
            gap_idx += 1
        
        # Apply shift
        new_x1 = max(0, x1 - cumulative_shift)
        new_x2 = max(0, x2 - cumulative_shift)
        new_cx = max(0, cx - cumulative_shift)
        
        det_copy['bbox'] = [new_x1, y1, new_x2, y2]
        det_copy['center'] = (new_cx, det['center'][1])
        updated_dets.append(det_copy)
    
    return updated_dets


# ============================================================================
# SMART RETARGETING - Automatic method selection
# ============================================================================

class SmartRetargeter:
    """
    Smart retargeting class that automatically chooses the best method:
    - Simple scaling for minor changes
    - Seam carving for clean backgrounds
    - Content-aware warping for complex scenes or large changes
    
    Based on Multi-operator approach from research papers.
    """
    
    def __init__(self, 
                 seam_cost_threshold: float = 30.0,
                 scale_distortion_threshold: float = 1.25,
                 minor_change_threshold: float = 0.15):
        """
        Args:
            seam_cost_threshold: Energy threshold. Below this -> use seam carving.
                                Above this -> use warping (complex background).
            scale_distortion_threshold: Max scale factor before considering alternatives (1.25 = 125%).
            minor_change_threshold: Scale change < this -> use simple scaling (default 15%).
        """
        self.seam_cost_threshold = seam_cost_threshold
        self.scale_max = scale_distortion_threshold
        self.minor_threshold = minor_change_threshold
    
    def compute_energy_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute energy map (gradient magnitude) - represents pixel importance.
        Fast Sobel filter instead of complex entropy calculations.
        
        Args:
            image: Input image (BGR, RGBA, or grayscale)
            
        Returns:
            Energy map (2D array)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute gradients using Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy = np.abs(grad_x) + np.abs(grad_y)
        
        return energy
    
    def evaluate_seam_cost(self, image: np.ndarray, target_w: int, target_h: int) -> float:
        """
        Estimate "cost" of using seam carving.
        Instead of running actual seam carving (slow), we estimate based on energy.
        
        High energy = complex background = high cost = seam carving will create artifacts
        Low energy = clean background = low cost = seam carving is safe
        
        Args:
            image: Input image
            target_w, target_h: Target dimensions
            
        Returns:
            Cost score (higher = more complex, worse for seam carving)
        """
        h, w = image.shape[:2]
        energy = self.compute_energy_map(image)
        
        cost_score = 0.0
        
        # Evaluate horizontal change (width)
        if target_w != w:
            # Mean energy represents overall complexity
            # For expansion, we need to "invent" new pixels - higher energy = harder
            # For reduction, we remove low-energy seams - higher energy = more distortion
            cost_score += np.mean(energy)
        
        # Evaluate vertical change (height)
        if target_h != h:
            cost_score += np.mean(energy)
        
        # Normalize by expansion/reduction amount
        # Larger changes = higher cost
        width_change_ratio = abs(target_w - w) / w
        height_change_ratio = abs(target_h - h) / h
        change_factor = max(width_change_ratio, height_change_ratio)
        
        cost_score *= (1 + change_factor)
        
        return cost_score
    
    def analyze_background_complexity(self, image: np.ndarray, 
                                     detection_box: Optional[Tuple[int, int, int, int]] = None) -> dict:
        """
        Analyze background complexity to help decision making.
        
        Args:
            image: Input image
            detection_box: Optional (x1, y1, x2, y2) of foreground object to exclude from analysis
            
        Returns:
            Dict with analysis results:
                - mean_energy: Average energy
                - std_energy: Standard deviation (uniformity)
                - edge_density: Ratio of high-energy pixels
                - is_clean: Boolean - is background clean enough for seam carving?
        """
        energy = self.compute_energy_map(image)
        
        # If detection box provided, analyze only background
        if detection_box is not None:
            x1, y1, x2, y2 = detection_box
            mask = np.ones(energy.shape, dtype=bool)
            mask[y1:y2, x1:x2] = False
            bg_energy = energy[mask]
        else:
            bg_energy = energy.flatten()
        
        mean_energy = np.mean(bg_energy)
        std_energy = np.std(bg_energy)
        
        # Edge density: ratio of pixels with high energy (> 75th percentile)
        threshold = np.percentile(bg_energy, 75)
        edge_density = np.sum(bg_energy > threshold) / len(bg_energy)
        
        # Decision: clean background if low mean energy AND low variance AND low edge density
        is_clean = (mean_energy < self.seam_cost_threshold and 
                   std_energy < mean_energy * 0.8 and 
                   edge_density < 0.30)
        
        return {
            'mean_energy': float(mean_energy),
            'std_energy': float(std_energy),
            'edge_density': float(edge_density),
            'is_clean': is_clean
        }
    
    def decide_method(self, image: np.ndarray, 
                     target_w: int, target_h: int,
                     detection_box: Optional[Tuple[int, int, int, int]] = None) -> dict:
        """
        Decide the best retargeting method based on image analysis.
        
        Args:
            image: Input image
            target_w, target_h: Target dimensions
            detection_box: Optional foreground object bbox
            
        Returns:
            Decision dict with:
                - method: 'scale', 'seam_carving', or 'warp'
                - reason: Explanation string
                - cost: Seam carving cost score
                - analysis: Background analysis dict
        """
        h_src, w_src = image.shape[:2]
        
        # Calculate scale ratios
        scale_w = target_w / w_src
        scale_h = target_h / h_src
        max_scale_needed = max(scale_w, scale_h)
        min_scale_needed = min(scale_w, scale_h)
        
        # === CASE 1: Minor change -> Simple scaling is optimal ===
        # If scale change is within minor_threshold (default 15%), just scale
        scale_diff_w = abs(scale_w - 1.0)
        scale_diff_h = abs(scale_h - 1.0)
        max_scale_diff = max(scale_diff_w, scale_diff_h)
        
        if max_scale_diff < self.minor_threshold:
            return {
                'method': 'scale',
                'reason': f'Minor change ({max_scale_diff*100:.1f}% < {self.minor_threshold*100:.0f}%)',
                'cost': 0.0,
                'scale_factor': max_scale_needed,
                'analysis': None
            }
        
        # === CASE 2: Analyze background complexity ===
        analysis = self.analyze_background_complexity(image, detection_box)
        seam_cost = self.evaluate_seam_cost(image, target_w, target_h)
        
        # === CASE 3: Clean background + reasonable scale -> Seam carving ===
        # Only use seam carving if:
        # 1. Background is clean (low energy)
        # 2. Scale is not too extreme (< 1.5x)
        # 3. Not shrinking too much (min scale > 0.7)
        is_reasonable_scale = (max_scale_needed < 1.5 and min_scale_needed > 0.7)
        
        if analysis['is_clean'] and is_reasonable_scale and seam_cost < self.seam_cost_threshold:
            return {
                'method': 'seam_carving',
                'reason': f'Clean background (energy={seam_cost:.1f} < {self.seam_cost_threshold})',
                'cost': seam_cost,
                'scale_factor': max_scale_needed,
                'analysis': analysis
            }
        
        # === CASE 4: Complex background OR extreme scale -> Content-aware warp ===
        # Use warping for:
        # - High energy background (lots of details/edges)
        # - Extreme scale changes
        # - Non-uniform scaling (different w/h ratios)
        
        if seam_cost >= self.seam_cost_threshold:
            reason = f'Complex background (energy={seam_cost:.1f} >= {self.seam_cost_threshold})'
        elif not is_reasonable_scale:
            reason = f'Extreme scale (max={max_scale_needed:.2f}, min={min_scale_needed:.2f})'
        else:
            reason = 'Non-uniform scaling or edge case'
        
        return {
            'method': 'warp',
            'reason': reason,
            'cost': seam_cost,
            'scale_factor': max_scale_needed,
            'analysis': analysis
        }
    
    def solve(self, image: np.ndarray, 
             target_w: int, target_h: int,
             detection_box: Optional[Tuple[int, int, int, int]] = None,
             verbose: bool = True) -> Tuple[Optional[np.ndarray], str, dict]:
        """
        Main entry point: Analyze and decide best method, but DON'T execute.
        
        This function only returns the DECISION. The caller should execute
        the appropriate method (scale/seam_carving/warp) based on the result.
        
        Args:
            image: Input image
            target_w, target_h: Target dimensions
            detection_box: Optional (x1, y1, x2, y2) of foreground object
            verbose: Print decision info
            
        Returns:
            Tuple of (processed_image, method_name, decision_dict)
            - processed_image: None (caller should process based on method)
            - method_name: 'scale', 'seam_carving', or 'warp'
            - decision_dict: Full decision info
        """
        decision = self.decide_method(image, target_w, target_h, detection_box)
        
        if verbose:
            print(f"[SmartRetarget] Decision: {decision['method'].upper()}")
            print(f"  Reason: {decision['reason']}")
            if decision['analysis']:
                analysis = decision['analysis']
                print(f"  Background analysis:")
                print(f"    - Mean energy: {analysis['mean_energy']:.2f}")
                print(f"    - Std energy: {analysis['std_energy']:.2f}")
                print(f"    - Edge density: {analysis['edge_density']*100:.1f}%")
                print(f"    - Is clean: {analysis['is_clean']}")
        
        return None, decision['method'], decision


# ============================================================================
# CONVENIENCE FUNCTIONS for collage_assembly.py
# ============================================================================

def smart_expand_to_fill(image: np.ndarray,
                        target_height: int,
                        target_width: int,
                        detection_box: Optional[Tuple[int, int, int, int]] = None,
                        seam_threshold: float = 30.0,
                        verbose: bool = True) -> Tuple[np.ndarray, str]:
    """
    Smart expansion using automatic method selection.
    
    This is a convenience function for collage_assembly.py to replace
    the manual if/else logic for choosing between seam carving and warping.
    
    Args:
        image: Input image to expand
        target_height, target_width: Target dimensions
        detection_box: Optional (x1, y1, x2, y2) of protected foreground
        seam_threshold: Energy threshold for seam carving decision
        verbose: Print decision info
        
    Returns:
        Tuple of (result_image, method_used)
        - result_image: Expanded image (or None if should use warp - caller handles)
        - method_used: 'scale', 'seam_carving', or 'warp'
    """
    retargeter = SmartRetargeter(seam_cost_threshold=seam_threshold)
    _, method, decision = retargeter.solve(image, target_width, target_height, 
                                          detection_box, verbose)
    
    h_src, w_src = image.shape[:2]
    
    # Execute based on decision
    if method == 'scale':
        # Simple scaling
        result = cv2.resize(image, (target_width, target_height), 
                          interpolation=cv2.INTER_LANCZOS4)
        return result, method
    
    elif method == 'seam_carving':
        # Create protection mask if detection box provided
        protect_mask = None
        if detection_box is not None:
            x1, y1, x2, y2 = detection_box
            protect_mask = np.zeros((h_src, w_src), dtype=np.float64)
            # Protect with padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w_src, x2 + padding)
            y2 = min(h_src, y2 + padding)
            protect_mask[y1:y2, x1:x2] = 255
        
        # Apply seam carving
        try:
            result = seam_carve(image, target_height, target_width, protect_mask)
            return result, method
        except Exception as e:
            if verbose:
                print(f"[SmartRetarget] Seam carving failed ({e}), falling back to scale")
            result = cv2.resize(image, (target_width, target_height), 
                              interpolation=cv2.INTER_LANCZOS4)
            return result, 'scale_fallback'
    
    else:  # method == 'warp'
        # Return None - caller should use content-aware warping
        return None, method


def seam_carve_expand_to_fill(image: np.ndarray, 
                               target_height: int, 
                               target_width: int,
                               detection_box: tuple,
                               debug_dir: str = None,
                               image_id: str = "") -> np.ndarray:
    """
    Use seam carving to expand image while protecting detection region.
    This produces smoother results than warp for filling empty areas.
    
    Args:
        image: Source image (RGBA or BGR)
        target_height: Target height
        target_width: Target width
        detection_box: (x1, y1, x2, y2) - region to protect
        debug_dir: Optional debug directory
        image_id: Image identifier for debug
        
    Returns:
        Expanded image with detection region preserved
    """
    h_src, w_src = image.shape[:2]
    det_x1, det_y1, det_x2, det_y2 = detection_box
    
    print(f"[SeamCarve Expand] Source: {w_src}x{h_src} -> Target: {target_width}x{target_height}")
    print(f"[SeamCarve Expand] Protecting detection: ({det_x1},{det_y1})-({det_x2},{det_y2})")
    
    # Create protection mask for detection region
    protect_mask = np.zeros((h_src, w_src), dtype=np.float64)
    padding = 10  # Small padding around detection
    px1 = max(0, int(det_x1) - padding)
    py1 = max(0, int(det_y1) - padding)
    px2 = min(w_src, int(det_x2) + padding)
    py2 = min(h_src, int(det_y2) + padding)
    protect_mask[py1:py2, px1:px2] = 255
    
    # Use seam carving to expand
    carver = SeamCarver(image, target_height, target_width, protect_mask)
    expanded = carver.seams_carving()
    
    print(f"[SeamCarve Expand] Result: {expanded.shape[1]}x{expanded.shape[0]}")
    
    # Debug visualization
    if debug_dir:
        from os.path import join
        vis = expanded.copy()
        if vis.shape[2] == 4:
            vis = cv2.cvtColor(vis, cv2.COLOR_RGBA2BGR)
        # Draw where detection should be (scaled proportionally)
        scale_x = target_width / w_src
        scale_y = target_height / h_src
        new_det_x1 = int(det_x1 * scale_x)
        new_det_y1 = int(det_y1 * scale_y)
        new_det_x2 = int(det_x2 * scale_x)
        new_det_y2 = int(det_y2 * scale_y)
        cv2.rectangle(vis, (new_det_x1, new_det_y1), (new_det_x2, new_det_y2), (0, 255, 0), 2)
        path = join(debug_dir, f"seam_carve_expand_{image_id}.png")
        cv2.imwrite(path, vis)
        print(f"[DEBUG] Seam carve expand: {path}")
    
    return expanded
