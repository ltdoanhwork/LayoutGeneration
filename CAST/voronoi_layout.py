#!/usr/bin/env python
# coding: utf-8
"""
Voronoi Layout V14 - Medial Axis Initialization

Key improvement:
- Added Medial Axis-based site initialization for better topology awareness
- Sites are placed along the skeleton of the shape, respecting its natural structure
- Endpoints (ears, limbs) get dedicated sites automatically

Previous fixes:
1. Jagged edges -> Reduced simplification factor.
2. Bad layout topology -> Used Distance Transform initialization.
3. Mismatched images -> Added Hungarian Algorithm for optimal assignment.
4. Gaps -> Added polygon buffering.
"""

import os
import sys
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon, box, LineString as ShapelyLineString
from shapely.affinity import scale as shp_scale, translate as shp_translate
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
from typing import Dict, List, Tuple

# Import shape decomposition for medial axis
try:
    import shape_decomposition as sd
    MEDIAL_AXIS_AVAILABLE = True
except ImportError:
    MEDIAL_AXIS_AVAILABLE = False
    print("[WARN] shape_decomposition not available, medial axis init disabled")

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    # Optimization Grid
    'resolution': 512,              # Grid size for gradient descent
    'num_iterations': 150,          # Increased for better convergence
    
    # Learning Rates
    'lr_sites': 0.01,               # Slightly increased
    'lr_weights': 0.02,
    'tau': 60.0,                    # Softmax temperature
    
    # Loss Weights (Simplified: 3 essential losses only)
    # Same weights as test files (proven stable)
    'w_capacity': 400.0,            # L_cap: capacity distribution
    'w_aspect': 600.0,              # L_asp: Match cell aspect ratio to bbox aspect ratio
    'w_overlap': 1500.0,            # Reserved for optional overlap objective
    'w_min_area': 1200.0,           # Penalize vanishing cells in soft ownership
    'w_repel': 30.0,                # Keep sites from collapsing into one point
    
    # REMOVED losses (redundant with Medial Axis init + improved losses):
    # 'w_cvt': 0.5,                 # L_cvt: Removed - redundant with good site init
    # 'w_min_area': 500.0,          # L_min: Removed - redundant with L_cap
    
    # Geometry Generation
    'render_res': 4096,             # Internal resolution for generating polygons
    'simplify': 0.0001,             # Polygon simplification epsilon (lower = more detailed)
    'poly_buffer': 2.0,             # Gap filling between cells (was 0.5, too small → white space)
    'repel_radius': 0.08,           # Soft repulsion radius (normalized)
    'weight_clip': 0.25,            # Clamp additive weights to keep all cells alive
    'dead_cell_boost': 0.2,         # Nudges weights of undersized cells to avoid dead hard-label cells
    
    # Cell Size Constraints - HARDENED but ADAPTIVE (scale with num cells)
    'min_cap_ratio': 0.025,         # Minimum cell = 2.5% of total area (capacity loss threshold)
    'min_avg_fraction_target': 0.75, # Target capacity floor as fraction of average cell area (avg=1/n)
    'min_avg_fraction_opt': 0.70,    # Optimization-time minimum area as fraction of average
    'capacity_uniform_blend': 0.55,  # Blend bbox-driven capacity toward uniform for near-average cell sizes
    'debug_every': 0,               # Save debug snapshot every N iters (0 = off)
    'debug_medial_axis': True,      # Save medial-axis + endpoint debug image
    'medial_ridge_threshold': 0.39, # Align with Colla-style ridge filtering
    
    # Site Initialization Method
    # 'distance_transform': Use distance transform to find deepest points (original)
    # 'medial_axis': Use medial axis skeleton for topology-aware initialization (NEW)
    # 'hybrid': Combine both: endpoints from medial axis + fill with distance transform
    'site_init_method': 'medial_axis',  # NEW: Default to medial axis for better structure
    
}
# =============================================================================
# CORE ENGINE
# =============================================================================

class VoronoiLayoutEngine:
    def __init__(self, mask_path: str, frame_infos: List[Dict], probabilities: List[float] = None, use_timeline_order: bool = False, output_dir: str = None, debug_every: int = 0):
        self.frame_infos = frame_infos
        self.probabilities = probabilities
        self.use_timeline_order = use_timeline_order
        self.output_dir = output_dir
        self.debug_every = max(0, int(debug_every)) if debug_every else 0
        self.n = len(frame_infos)
        self.mask_path = mask_path
        
        # 1. Load and Normalize Mask
        self._load_mask()
        
        # 2. Analyze Image Requirements
        self._analyze_images()
        
        # 3. Anchor Assignment: Initially cell[i] <-> image[i] (same index)
        # This will be updated after Hungarian matching
        self.anchor_img_idx = list(range(self.n))

        # Debug snapshots for site initialization/optimization analysis
        self.initial_sites_debug = None
        self.optimized_sites_debug = None

    def _load_mask(self):
        # Load mask (255 = foreground)
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Fallback if path is wrong, create a dummy box
            print(f"[WARN] Mask not found at {self.mask_path}, using dummy.")
            mask = np.zeros((500, 500), dtype=np.uint8)
            cv2.rectangle(mask, (50, 50), (450, 450), 255, -1)
        
        # Ensure binary
        _, self.mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Denoise mask to prevent jagged edges from noise
        kernel = np.ones((5,5), np.uint8)
        self.mask_binary = cv2.morphologyEx(self.mask_binary, cv2.MORPH_CLOSE, kernel)
        
        self.H, self.W = self.mask_binary.shape
        self.max_dim = max(self.H, self.W)
        
        # Normalized dimensions [0, 1]
        self.norm_h = self.H / self.max_dim
        self.norm_w = self.W / self.max_dim
        
        # Create shape polygon from mask contours (for Voronoi clipping)
        self.shape_poly = None
        try:
            contours, _ = cv2.findContours(self.mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 3:
                    # Convert to normalized coordinates
                    points = [(p[0][0] / self.max_dim, p[0][1] / self.max_dim) for p in largest]
                    self.shape_poly = Polygon(points)
                    if not self.shape_poly.is_valid:
                        self.shape_poly = self.shape_poly.buffer(0)
        except Exception as e:
            print(f"  [Voronoi] Failed to create shape_poly: {e}")
            self.shape_poly = None

    def _analyze_images(self):
        """Analyze input images to determine target capacities and aspect ratios."""
        caps = []
        aspects = []
        
        for i, info in enumerate(self.frame_infos):
            # Use IMAGE size for capacity (not bbox - bbox is only for crop centering)
            frame_size = info.get('frame_size', (640, 480))
            img_w, img_h = frame_size[0], frame_size[1]
            
            # Aspect ratio from BBOX if available (ensures cell matches OBJECT shape)
            if 'bbox' in info and info['bbox']:
                # SAS format: [x1, y1, x2, y2]
                bbox = info['bbox']
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bw > 10 and bh > 10:
                    aspect = bw / (bh + 1e-6)
                    # Clamp to avoid extreme cells
                    aspect = max(0.3, min(3.0, aspect))
                else:
                    aspect = img_w / (img_h + 1e-6)
            else:
                aspect = img_w / (img_h + 1e-6)

            aspects.append(aspect)
            
            # Capacity based on BBOX area (larger bbox = larger cell)
            # This ensures frames with bigger characters get more space
            if 'bbox' in info:
                bbox = info['bbox']
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                bbox_area = max(bw * bh, 100)  # Minimum area to avoid div by zero
            else:
                bbox_area = img_w * img_h * 0.64  # Fallback: 80% center crop
            
            base_cap = bbox_area ** 0.5  # sqrt for more balanced distribution
            
            # Probability boost: More moderate now for equal containment
            if self.probabilities:
                prob = self.probabilities[i]
                # Reduced boost: prob=1.0 gets 1.5x, prob=0.0 gets 0.5x
                size_boost = 0.5 + (1.0 * prob)
                base_cap *= size_boost
                
            caps.append(base_cap)
            
        # Normalize capacities so they sum to 1.0 (or total area, but optimizer uses relative)
        total_cap = sum(caps)
        norm_caps = [c / total_cap for c in caps]

        # Keep cell sizes closer to average while preserving some bbox-based variation.
        blend = float(np.clip(CONFIG.get('capacity_uniform_blend', 0.0), 0.0, 1.0))
        if self.n > 0 and blend > 0.0:
            avg_cap = 1.0 / self.n
            norm_caps = [(1.0 - blend) * c + blend * avg_cap for c in norm_caps]
        
        # Enforce minimum target cell size as a fraction of average area.
        min_cap = max(
            CONFIG.get('min_cap_ratio', 0.02),
            CONFIG.get('min_avg_fraction_target', 0.75) / max(self.n, 1),
        )
        norm_caps = [max(c, min_cap) for c in norm_caps]
        
        # Re-normalize after clamping
        total_clamped = sum(norm_caps)
        norm_caps = [c / total_clamped for c in norm_caps]
        
        self.target_caps = torch.tensor(norm_caps, device=DEVICE, dtype=torch.float32)
        self.target_aspects = torch.tensor(aspects, device=DEVICE, dtype=torch.float32)
        
        # All frames have equal priority (JSON filtering already done upstream)
        bbox_priorities = [1.0 for _ in self.frame_infos]
        self.bbox_priorities = torch.tensor(bbox_priorities, device=DEVICE, dtype=torch.float32)
        
        print(f"  [Voronoi] All {self.n} frames treated equally (priority=1.0)")

        # Anisotropy Matrices (Allow cells to stretch)
        self.aniso_matrices = torch.zeros((self.n, 2, 2), device=DEVICE)
        strength = 0.85 
        
        for i in range(self.n):
            asp = self.target_aspects[i].item()
            s = math.sqrt(max(0.25, min(4.0, asp))) # Clamp ratio
            sx = (1-strength) + strength * (1/s)
            sy = (1-strength) + strength * s
            
            tr = sx + sy
            self.aniso_matrices[i, 0, 0] = sx * (2/tr)
            self.aniso_matrices[i, 1, 1] = sy * (2/tr)

    def _poly_to_image_coords(self, poly: Polygon) -> Polygon:
        """Convert normalized Voronoi polygon to image pixel coords (top-left origin)."""
        if not hasattr(poly, 'exterior'):
            return poly

        # NOTE:
        # Polygons generated from mask contours are already in image-like coordinates
        # (OpenCV convention: y increases downward). So here we only SCALE to pixel space.
        # Flipping Y here would mirror geometry and break kept_ratio / feasibility.
        return shp_scale(poly, xfact=self.max_dim, yfact=self.max_dim, origin=(0, 0))

    def _build_site_projection_maps(self, mask_rs: np.ndarray):
        """Precompute nearest-foreground lookup to keep sites inside mask foreground."""
        fg_mask = mask_rs >= 127
        if not np.any(fg_mask):
            return None, None, None

        nearest_fg = ndimage.distance_transform_edt(
            ~fg_mask,
            return_distances=False,
            return_indices=True,
        )

        fg_mask_t = torch.tensor(fg_mask, device=DEVICE, dtype=torch.bool)
        nearest_y_t = torch.tensor(nearest_fg[0], device=DEVICE, dtype=torch.long)
        nearest_x_t = torch.tensor(nearest_fg[1], device=DEVICE, dtype=torch.long)
        return fg_mask_t, nearest_y_t, nearest_x_t

    def _project_sites_to_foreground_(
        self,
        sites: torch.Tensor,
        fg_mask_t: torch.Tensor,
        nearest_y_t: torch.Tensor,
        nearest_x_t: torch.Tensor,
        gw: int,
        gh: int,
    ):
        """In-place projection of sites onto nearest foreground pixel in mask grid."""
        if fg_mask_t is None or nearest_y_t is None or nearest_x_t is None:
            return

        if gw <= 1 or gh <= 1:
            return

        sx = torch.round((sites[:, 0] / max(self.norm_w, 1e-6)) * (gw - 1)).long().clamp(0, gw - 1)
        sy = torch.round((sites[:, 1] / max(self.norm_h, 1e-6)) * (gh - 1)).long().clamp(0, gh - 1)

        outside = ~fg_mask_t[sy, sx]
        if outside.any():
            oy = sy[outside]
            ox = sx[outside]
            ny = nearest_y_t[oy, ox]
            nx = nearest_x_t[oy, ox]

            sites[outside, 0] = (nx.float() / (gw - 1)) * self.norm_w
            sites[outside, 1] = (ny.float() / (gh - 1)) * self.norm_h

    def _save_medial_axis_debug(
        self,
        multilinestring,
        all_endpoints,
        accepted_endpoints,
        rejected_endpoints,
    ):
        """Save medial axis and endpoint selection diagnostics."""
        if not self.output_dir or not CONFIG.get('debug_medial_axis', True):
            return

        try:
            debug = cv2.cvtColor(self.mask_binary, cv2.COLOR_GRAY2BGR)
            h, w = self.mask_binary.shape

            lines = []
            if hasattr(multilinestring, 'geoms'):
                lines = list(multilinestring.geoms)
            elif hasattr(multilinestring, 'coords'):
                lines = [multilinestring]

            for ls in lines:
                coords = np.array(ls.coords, dtype=np.float32)
                if coords.shape[0] < 2:
                    continue
                pts = []
                for x, y in coords:
                    row, col = sd.xy2rowcol(float(x), float(y), h)
                    px = int(np.clip(col, 0, w - 1))
                    py = int(np.clip(row, 0, h - 1))
                    pts.append([px, py])
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(debug, [pts], False, (255, 220, 0), 1, lineType=cv2.LINE_AA)

            # Draw all endpoint candidates first (orange), then accepted/rejected overlays.
            for x, y in all_endpoints:
                cv2.circle(debug, (x, y), 4, (0, 165, 255), 1)
            for x, y in rejected_endpoints:
                cv2.circle(debug, (x, y), 4, (0, 0, 255), -1)
            for x, y in accepted_endpoints:
                cv2.circle(debug, (x, y), 5, (0, 255, 0), -1)

            cv2.putText(
                debug,
                f"candidates={len(all_endpoints)} accepted={len(accepted_endpoints)} rejected={len(rejected_endpoints)}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imwrite(
                os.path.join(self.output_dir, 'voronoi_debug_1b_medial_axis_endpoints.png'),
                debug,
            )
        except Exception as e:
            print(f"  [Voronoi] Could not save medial-axis debug: {e}")

    def _compute_cell_bbox_iou(self, polygons: List[Polygon], assignment = None) -> List[float]:
        """
        Compute actual IoU between each cell polygon and its assigned image's bbox.
        
        Returns list of IoU scores (0.0 to 1.0) for each image.
        """
        n = len(polygons)
        iou_scores = []
        
        for img_idx in range(n):
            # Get cell index (if assignment provided, use it; otherwise img_idx == cell_idx)
            cell_idx = assignment[img_idx] if assignment is not None else img_idx
            
            poly = polygons[cell_idx]
            if not hasattr(poly, 'exterior') or poly.is_empty:
                iou_scores.append(0.0)
                continue
            
            # Get bbox in normalized coords
            info = self.frame_infos[img_idx]
            img_w, img_h = info.get('frame_size', (640, 480))
            bbox = info.get('bbox', [0, 0, img_w, img_h])
            
            # Normalized bbox dimensions
            bw = (bbox[2] - bbox[0]) / img_w * self.norm_w
            bh = (bbox[3] - bbox[1]) / img_h * self.norm_h
            
            # Cell bounds
            px_min, py_min, px_max, py_max = poly.bounds
            cell_w = max(px_max - px_min, 1e-6)
            cell_h = max(py_max - py_min, 1e-6)
            
            # Scale bbox to cell size (cover mode)
            scale = max(cell_w / self.norm_w, cell_h / self.norm_h)
            scaled_bw = bw * scale
            scaled_bh = bh * scale
            
            # Create bbox polygon centered in cell
            cx, cy = poly.centroid.x, poly.centroid.y
            bbox_poly = box(
                cx - scaled_bw / 2, cy - scaled_bh / 2,
                cx + scaled_bw / 2, cy + scaled_bh / 2
            )
            
            # Compute IoU
            try:
                intersection = poly.intersection(bbox_poly)
                inter_area = intersection.area
                bbox_area = bbox_poly.area
                iou = inter_area / (bbox_area + 1e-6)
            except:
                iou = 0.0
            
            iou_scores.append(min(1.0, iou))
        
        return iou_scores

    def _init_sites_smart(self):
        """
        Initialize sites using Distance Transform.
        Places points at the 'deepest' parts of the mask to ensure centrality.
        
        LIMITATION: Does not know about shape topology (e.g., ears of Totoro).
        Sites may cluster in wide areas, ignoring narrow protrusions.
        """
        # Downscale mask for analysis
        h, w = self.mask_binary.shape
        scale = 256 / max(h, w)
        small_mask = cv2.resize(self.mask_binary, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        # Distance Transform: Value = distance to nearest zero pixel
        dist_map = cv2.distanceTransform(small_mask, cv2.DIST_L2, 5)
        
        sites = []
        for _ in range(self.n):
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
            
            # Convert back to normalized coords
            nx = (max_loc[0] / small_mask.shape[1]) * self.norm_w
            ny = (max_loc[1] / small_mask.shape[0]) * self.norm_h
            sites.append([nx, ny])
            
            # Suppress this area so next site is far away
            suppress_radius = int(math.sqrt(small_mask.shape[0]*small_mask.shape[1] / self.n) * 0.8)
            cv2.circle(dist_map, max_loc, suppress_radius, 0, -1)
            
        return torch.tensor(sites, device=DEVICE, dtype=torch.float32, requires_grad=True)
    
    def _init_sites_medial_axis(self):
        """
        HYBRID Site Initialization: Endpoints First + Distance Transform Fill
        
        Strategy:
        1. FIRST: Place sites at ALL medial axis endpoints (ears, limbs, protrusions)
           → Guarantees narrow regions get dedicated sites
        2. THEN: Fill remaining sites using Distance Transform
           → Distributes remaining sites in wide areas optimally
        
        This hybrid approach ensures:
        - Narrow protrusions (ears, tails, legs) ALWAYS get sites (endpoint priority)
        - Wide areas (body, head) get remaining sites (distance transform)
        - No clustering issues from arc-length sampling
        
        Example for Totoro (7 images):
        - Endpoints: 2 ears + 1 tail = 3 sites guaranteed
        - Distance Transform: 4 more sites in body/head area
        """
        if not MEDIAL_AXIS_AVAILABLE:
            print("  [Voronoi] Medial axis not available, falling back to distance transform")
            return self._init_sites_smart()
        
        try:
            h, w = self.mask_binary.shape
            medial_input = self.mask_binary > 127  # Shape = True, Background = False
            
            print(f"  [Voronoi] Building medial axis skeleton...")
            
            # Build medial axis of the SHAPE (interior skeleton)
            ma_result = sd.ridge_medial_axis(
                medial_input,
                ridge_threshold=CONFIG.get('medial_ridge_threshold', 0.39),
                small_threshold=5,
            )
            ma_group, distance_map = ma_result
            
            # Build multilinestring from medial axis
            mls_result = sd.build_medial_multilinestring(ma_group)
            multilinestring = mls_result[0] if isinstance(mls_result, tuple) else mls_result
            
            if multilinestring.is_empty:
                print("  [Voronoi] Medial axis empty, falling back to distance transform")
                return self._init_sites_smart()
            
            # Redistribute vertices for graph building
            gap = max(5, int(multilinestring.length / (self.n * 3)))
            final_vertices = sd.redistribute_vertices(multilinestring, gap)
            
            # Build graph from skeleton
            line_labels = mls_result[1] if isinstance(mls_result, tuple) and len(mls_result) > 1 else None
            G = None
            if line_labels is not None:
                try:
                    G = sd.build_medial_graph(final_vertices, line_labels, distance_map)
                except:
                    pass
            
            endpoint_sites = []
            endpoint_candidates = []
            accepted_endpoint_pixels = []
            rejected_endpoint_pixels = []
            
            # ===== STEP 1: Place sites at ALL endpoints =====
            if G is not None:
                try:
                    end_vertices = sd.find_end_vertices(G, exterior=False)
                    print(f"  [Voronoi] Found {len(end_vertices)} endpoints in skeleton")
                    
                    # Sort by distance from boundary (deeper = more important)
                    end_vertices_sorted = sorted(
                        end_vertices,
                        key=lambda v: G.nodes[v].get('distance', 0),
                        reverse=True
                    )
                    
                    # Add ALL endpoints as priority sites (up to n)
                    for v in end_vertices_sorted:
                        if len(endpoint_sites) >= self.n:
                            break
                        
                        x = float(G.nodes[v]['x'])
                        y = float(G.nodes[v]['y'])

                        # Skeleton graph coordinates are XY with bottom-left origin,
                        # but image masks are row/col with top-left origin.
                        row, col = sd.xy2rowcol(x, y, h)
                        check_x = int(np.clip(col, 0, w - 1))
                        check_y = int(np.clip(row, 0, h - 1))
                        endpoint_candidates.append((check_x, check_y))
                        
                        if self.mask_binary[check_y, check_x] > 127:
                            norm_x = (check_x / w) * self.norm_w
                            norm_y = (check_y / h) * self.norm_h
                            endpoint_sites.append([norm_x, norm_y])
                            accepted_endpoint_pixels.append((check_x, check_y))
                        else:
                            rejected_endpoint_pixels.append((check_x, check_y))
                    
                    print(f"  [Voronoi] ✓ Placed {len(endpoint_sites)} sites at endpoints")
                    
                except Exception as e:
                    print(f"  [Voronoi] Warning: Error finding endpoints: {e}")

            self._save_medial_axis_debug(
                multilinestring,
                endpoint_candidates,
                accepted_endpoint_pixels,
                rejected_endpoint_pixels,
            )
            
            # ===== STEP 2: Fill remaining with Distance Transform =====
            remaining_needed = self.n - len(endpoint_sites)
            
            if remaining_needed > 0:
                print(f"  [Voronoi] Filling {remaining_needed} more sites with distance transform...")
                
                all_sites = self._fill_sites_with_distance_transform(endpoint_sites)
                
                print(f"  [Voronoi] ✓ Total {len(all_sites)} sites (HYBRID: {len(endpoint_sites)} endpoints + {len(all_sites) - len(endpoint_sites)} DT)")
                
                return torch.tensor(all_sites[:self.n], device=DEVICE, dtype=torch.float32, requires_grad=True)
            else:
                print(f"  [Voronoi] ✓ Total {len(endpoint_sites)} sites (all from endpoints)")
                return torch.tensor(endpoint_sites[:self.n], device=DEVICE, dtype=torch.float32, requires_grad=True)
            
        except Exception as e:
            print(f"  [Voronoi] Medial axis failed ({e}), falling back to distance transform")
            import traceback
            traceback.print_exc()
            return self._init_sites_smart()
    
    def _fill_sites_with_distance_transform(self, existing_sites):
        """
        Scoring Method: Best position = DT_norm + (DistToSites_norm * alpha)
        
        This ensures sites are:
        1. Deep in the shape (high DT value)
        2. Far from existing sites (spread out evenly)
        
        Score = DT_normalized + (Distance_to_sites_normalized * 2.0)
        """
        h, w = self.mask_binary.shape
        scale = 256 / max(h, w)
        small_h, small_w = int(h * scale), int(w * scale)
        small_mask = cv2.resize(self.mask_binary, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        
        sites = list(existing_sites)
        n_needed = self.n - len(sites)
        
        if n_needed <= 0:
            return sites
        
        # 1. Compute normalized DT map (depth of shape)
        dt_map = cv2.distanceTransform(small_mask, cv2.DIST_L2, 5).astype(np.float32)
        max_dt = np.max(dt_map)
        dt_norm = dt_map / max_dt if max_dt > 0 else dt_map

        # Foreground occupancy map used for hard exclusion zones.
        # This prevents the greedy selector from repeatedly picking the same basin.
        available_mask = np.where(small_mask >= 127, 255, 0).astype(np.uint8)

        valid_px = max(1, int(np.count_nonzero(available_mask)))
        exclusion_radius = int(math.sqrt(valid_px / max(self.n, 1)) * 0.5)
        exclusion_radius = max(3, min(exclusion_radius, max(4, min(small_h, small_w) // 6)))
        
        # 2. Initialize sites mask for distance calculation
        sites_mask = np.ones((small_h, small_w), dtype=np.uint8) * 255
        
        # Mark existing sites on mask
        for sx, sy in existing_sites:
            px = int(np.clip((sx / self.norm_w) * small_w, 0, small_w - 1))
            py = int(np.clip((sy / self.norm_h) * small_h, 0, small_h - 1))
            cv2.circle(sites_mask, (px, py), exclusion_radius, 0, -1)  # Punch exclusion zone at site
            cv2.circle(available_mask, (px, py), exclusion_radius, 0, -1)
        
        # Compute initial distance to sites
        dist_to_sites = cv2.distanceTransform(sites_mask, cv2.DIST_L2, 5).astype(np.float32)
        
        # 3. Greedy selection using combined score
        alpha = 2.5  # Higher spread pressure than before
        
        for _ in range(n_needed):
            # Normalize distance to sites
            max_dist = np.max(dist_to_sites)
            dist_norm = dist_to_sites / max_dist if max_dist > 0 else dist_to_sites
            
            # Combined score: deep in shape + far from other sites
            combined_score = dt_norm + (dist_norm * alpha)
            
            # Mask out background + blocked neighborhoods
            combined_score[available_mask < 127] = -1
            
            # Find best position
            _, max_val, _, max_loc = cv2.minMaxLoc(combined_score)
            
            if max_val <= 0:
                break
            
            # Add new site (convert to normalized coords)
            nx = (max_loc[0] / small_w) * self.norm_w
            ny = (max_loc[1] / small_h) * self.norm_h
            sites.append([nx, ny])
            
            # Update distance map: punch larger exclusion around new site
            cv2.circle(sites_mask, max_loc, exclusion_radius, 0, -1)
            cv2.circle(available_mask, max_loc, exclusion_radius, 0, -1)
            dist_to_sites = cv2.distanceTransform(sites_mask, cv2.DIST_L2, 5).astype(np.float32)

        # If still short, continue with relaxed exclusion to guarantee enough sites.
        relaxed_radius = max(1, exclusion_radius // 3)
        while len(sites) < self.n:
            max_dist = np.max(dist_to_sites)
            dist_norm = dist_to_sites / max_dist if max_dist > 0 else dist_to_sites
            fallback_score = dt_norm + (dist_norm * alpha)
            fallback_score[small_mask < 127] = -1

            _, max_val, _, max_loc = cv2.minMaxLoc(fallback_score)
            if max_val <= 0:
                break

            nx = (max_loc[0] / small_w) * self.norm_w
            ny = (max_loc[1] / small_h) * self.norm_h
            sites.append([nx, ny])

            cv2.circle(sites_mask, max_loc, relaxed_radius, 0, -1)
            dist_to_sites = cv2.distanceTransform(sites_mask, cv2.DIST_L2, 5).astype(np.float32)
        
        return sites
    
    def _init_sites_grid(self):
        """Initialize sites in a grid pattern (Timeline friendly).
        
        This method spreads sites evenly from top to bottom in a grid pattern.
        This is optimal for Timeline Order because:
        1. Sites start at their approximate target Y positions
        2. Minimizes movement during optimization
        3. Natural top-to-bottom ordering aligns with temporal sequence
        """
        # Calculate number of rows/cols based on mask aspect ratio
        aspect = self.norm_w / self.norm_h
        n_cols = int(math.sqrt(self.n * aspect))
        n_cols = max(1, n_cols)
        n_rows = math.ceil(self.n / n_cols)
        
        sites = []
        # Create grid
        for r in range(n_rows):
            for c in range(n_cols):
                if len(sites) >= self.n:
                    break
                
                # Normalized coordinates (add 0.5 to get cell center)
                nx = (c + 0.5) * (self.norm_w / n_cols)
                ny = (r + 0.5) * (self.norm_h / n_rows)
                
                # Clamp to valid mask coordinates for checking
                check_x = int(np.clip(nx / self.norm_w * self.W, 0, self.W - 1))
                check_y = int(np.clip(ny / self.norm_h * self.H, 0, self.H - 1))
                
                sites.append([nx, ny])
        
        # If short (due to rounding), fill with center points
        while len(sites) < self.n:
            sites.append([self.norm_w / 2, self.norm_h / 2])
        
        print(f"  [Voronoi] Initialized {len(sites)} sites using GRID (Timeline Optimization)")
        return torch.tensor(sites, device=DEVICE, dtype=torch.float32, requires_grad=True)
    
    def _get_initial_sites(self):
        """
        Get initial sites based on configured method.
        
        NOTE: Timeline Order only affects ASSIGNMENT (match_images_spatial_order),
        NOT initialization. Medial axis init is always preferred because:
        1. Sites start inside the shape (Grid places many outside the mask → waste)
        2. Narrow regions (ears, tails) get dedicated sites automatically
        3. Spatial order assignment still ensures timeline flow after optimization
        """
        method = CONFIG.get('site_init_method', 'distance_transform')
        
        if method == 'medial_axis':
            return self._init_sites_medial_axis()
        elif method == 'hybrid':
            # Hybrid: Start with medial axis, fill with distance transform
            return self._init_sites_medial_axis()  # Already has fallback built-in
        else:
            return self._init_sites_smart()

    def _save_iteration_debug(self, iter_idx, mask_rs, sites, loss_dict):
        if not self.output_dir:
            return
        if self.debug_every <= 0 and iter_idx != -1:
            return

        # Save loss log (4 essential losses only)
        loss_path = os.path.join(self.output_dir, "voronoi_debug_iter_losses.csv")
        header = "iter,loss_total,loss_cap,loss_cvt,loss_asp,loss_ov\n"
        line = f"{iter_idx},{loss_dict['total']:.6f},{loss_dict['cap']:.6f},{loss_dict['cvt']:.6f},{loss_dict['asp']:.6f},{loss_dict['ov']:.6f}\n"
        
        # FIX: Reset file at iteration 0 to avoid duplicate entries from multiple runs
        if iter_idx == 0:
            with open(loss_path, "w") as f:
                f.write(header)
                f.write(line)
        else:
            with open(loss_path, "a") as f:
                f.write(line)

        # Save site visualization with FILLED Voronoi cells
        gh, gw = mask_rs.shape[:2]
        scale_x = gw / max(self.norm_w, 1e-6)
        scale_y = gh / max(self.norm_h, 1e-6)
        
        # Create color image for visualization
        debug_img = np.zeros((gh, gw, 3), dtype=np.uint8)
        
        # Color palette for cells (pastel colors)
        # BGR (OpenCV uses BGR) - Reversing RGB tuples from original code
        CELL_COLORS = [
            (186, 179, 255),  # Light pink (BGR)
            (201, 255, 186),  # Light green (BGR)
            (255, 225, 186),  # Light blue (BGR)
            (186, 255, 255),  # Light yellow (BGR)
            (255, 186, 223),  # Light purple (BGR)
            (255, 218, 185),  # Peach (BGR)
            (255, 255, 185),  # Cyan (BGR)
            (230, 185, 255),  # Rose (BGR)
            (200, 230, 200),  # Sage (BGR)
            (200, 200, 230),  # Dusty rose (BGR)
        ]
        
        # Generate Voronoi cells for visualization
        sites_np = sites.detach().cpu().numpy()
        
        # Compute Voronoi diagram
        cells_drawn = 0
        if len(sites_np) >= 3:
            try:
                from scipy.spatial import Voronoi as ScipyVoronoi
                from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
                
                vor = ScipyVoronoi(sites_np)
                
                # Create bounding box for clipping (use shape bounds)
                clip_box = shapely_box(0, 0, self.norm_w, self.norm_h)
                if self.shape_poly and self.shape_poly.is_valid:
                    clip_box = self.shape_poly
                
                # Draw each Voronoi cell (FILLED)
                for point_idx in range(len(sites_np)):
                    region_idx = vor.point_region[point_idx]
                    vertex_indices = vor.regions[region_idx]
                    
                    if -1 not in vertex_indices and len(vertex_indices) > 0:
                        try:
                            # Get vertices
                            poly_verts = [vor.vertices[i] for i in vertex_indices]
                            cell_poly = ShapelyPolygon(poly_verts)
                            
                            # Clip to bounds
                            clipped = cell_poly.intersection(clip_box)
                            
                            if not clipped.is_empty and hasattr(clipped, 'exterior'):
                                # Scale to image coords
                                pts = np.array([[int(x * scale_x), int(y * scale_y)] 
                                              for x, y in clipped.exterior.coords], dtype=np.int32)
                                
                                # FILL cell with color
                                color = CELL_COLORS[point_idx % len(CELL_COLORS)]
                                cv2.fillPoly(debug_img, [pts], color)
                                
                                # Draw cell outline (darker)
                                outline_color = tuple(max(0, c - 80) for c in color)
                                cv2.polylines(debug_img, [pts], True, outline_color, 2)
                                
                                cells_drawn += 1
                        except Exception as e:
                            pass
                
                # Draw sites on top of cells
                for point_idx, (sx, sy) in enumerate(sites_np):
                    px = int(sx * scale_x)
                    py = int(sy * scale_y)
                    # White circle with black outline
                    cv2.circle(debug_img, (px, py), 6, (0, 0, 0), -1)
                    cv2.circle(debug_img, (px, py), 4, (255, 255, 255), -1)
                    # Cell number
                    cv2.putText(debug_img, str(point_idx), (px + 8, py + 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(debug_img, str(point_idx), (px + 8, py + 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                              
            except Exception as e:
                # Fallback: draw mask with points
                debug_img = cv2.cvtColor(mask_rs, cv2.COLOR_GRAY2BGR)
                for i, (sx, sy) in enumerate(sites_np):
                    px = int(sx * scale_x)
                    py = int(sy * scale_y)
                    cv2.circle(debug_img, (px, py), 5, (0, 255, 0), -1)
                    cv2.putText(debug_img, str(i), (px + 6, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Apply mask (black outside shape)
        mask_3ch = cv2.cvtColor(mask_rs, cv2.COLOR_GRAY2BGR)
        debug_img = cv2.bitwise_and(debug_img, mask_3ch)
        
        # Add iteration number
        cv2.putText(debug_img, f"Iter {iter_idx}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Cells: {cells_drawn}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out_name = f"voronoi_debug_iter_{iter_idx:04d}.png"
        cv2.imwrite(os.path.join(self.output_dir, out_name), debug_img)

    def optimize(self):
        print(f"  [Voronoi] Optimizing layout for {self.n} cells...")
        
        # 1. Setup Grid
        res = CONFIG['resolution']
        gw = int(res * self.norm_w)
        gh = int(res * self.norm_h)
        
        yc = torch.linspace(0, self.norm_h, gh, device=DEVICE)
        xc = torch.linspace(0, self.norm_w, gw, device=DEVICE)
        yy, xx = torch.meshgrid(yc, xc, indexing='ij')
        grid_coords = torch.stack([xx, yy], dim=-1)
        
        # Prepare Mask
        mask_rs = cv2.resize(self.mask_binary, (gw, gh), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.tensor(mask_rs, device=DEVICE).float() / 255.0
        total_pixels = mask_tensor.sum()
        fg_mask_t, nearest_fg_y_t, nearest_fg_x_t = self._build_site_projection_maps(mask_rs)
        
        # 2. Init Sites using configured method (distance_transform or medial_axis)
        sites = self._get_initial_sites()
        with torch.no_grad():
            self._project_sites_to_foreground_(sites, fg_mask_t, nearest_fg_y_t, nearest_fg_x_t, gw, gh)
            self.initial_sites_debug = sites.detach().cpu().numpy().copy()
        weights = torch.zeros(self.n, device=DEVICE, requires_grad=True)
        
        optimizer = torch.optim.Adam([
            {'params': sites, 'lr': CONFIG['lr_sites']},
            {'params': weights, 'lr': CONFIG['lr_weights']}
        ])
        
        # 3. Loop
        n_iters = CONFIG['num_iterations']
        for i in range(n_iters):
            optimizer.zero_grad()
            
            # Anisotropic Distance
            diff = grid_coords.unsqueeze(2) - sites.view(1, 1, self.n, 2)
            # Flatten for efficient matmul
            diff_flat = diff.view(-1, self.n, 2, 1)
            A_flat = self.aniso_matrices.unsqueeze(0)
            
            # d^T * A * d
            term1 = torch.matmul(A_flat, diff_flat)
            d_sq = torch.matmul(diff_flat.transpose(-2, -1), term1).view(gh, gw, self.n)
            
            # Power Diagram
            d_final = d_sq - weights.view(1, 1, self.n)
            
            # Masking background (Infinity penalty)
            bg_penalty = (1.0 - mask_tensor.unsqueeze(-1)) * 1e6
            probs = F.softmax(-CONFIG['tau'] * (d_final + bg_penalty), dim=-1)
            
            # ========== Single-phase loss system ==========
            # L_cap: Capacity - distribute area proportional to target
            # L_cvt: Centroidal - sites at cell centroids (regularity)
            # L_asp: Aspect Ratio - match cell shape to bbox shape
            
            # Cell areas
            areas = (probs * mask_tensor.unsqueeze(-1)).sum(dim=(0,1))
            
            # Cell centroids
            coords_w = grid_coords.unsqueeze(2) * probs.unsqueeze(-1)
            centroids = coords_w.sum(dim=(0,1)) / (areas.unsqueeze(-1) + 1e-6)
            
            # ===== 1. L_cap: Capacity Loss =====
            loss_cap = ((areas - self.target_caps * total_pixels)**2).mean()
            
            # ===== 2. L_cvt: Centroidal Loss (regularity) =====
            loss_cvt = ((sites - centroids) ** 2).sum()
            
            # ===== 3. L_asp: Aspect Ratio Loss =====
            coords_x = grid_coords[..., 0].unsqueeze(-1)
            coords_y = grid_coords[..., 1].unsqueeze(-1)
            
            var_x = ((coords_x - centroids[:, 0].view(1, 1, -1))**2 * probs * mask_tensor.unsqueeze(-1)).sum(dim=(0,1)) / (areas + 1e-6)
            var_y = ((coords_y - centroids[:, 1].view(1, 1, -1))**2 * probs * mask_tensor.unsqueeze(-1)).sum(dim=(0,1)) / (areas + 1e-6)
            
            cell_aspects = torch.sqrt(var_x / (var_y + 1e-6))
            log_aspect_diff = torch.log(cell_aspects + 1e-6) - torch.log(self.target_aspects + 1e-6)
            loss_aspect = ((log_aspect_diff ** 2) * self.bbox_priorities).mean()

            # ===== 4. L_min_area: discourage near-zero cells =====
            area_ratio = areas / (total_pixels + 1e-6)
            min_ratio = max(
                CONFIG.get('min_cap_ratio', 0.025),
                CONFIG.get('min_avg_fraction_opt', 0.70) / max(self.n, 1),
            )
            loss_min_area = torch.relu(min_ratio - area_ratio).pow(2).mean()

            # ===== 5. L_repel: avoid site collapse =====
            if self.n > 1:
                pairwise = sites.unsqueeze(1) - sites.unsqueeze(0)
                dist = torch.sqrt((pairwise ** 2).sum(dim=-1) + 1e-9)
                repel_margin = CONFIG.get('repel_radius', 0.08)
                repel = torch.relu(repel_margin - dist)
                mask_offdiag = ~torch.eye(self.n, device=DEVICE, dtype=torch.bool)
                loss_repel = (repel[mask_offdiag] ** 2).mean()
            else:
                loss_repel = torch.tensor(0.0, device=DEVICE)

            # ===== Total Loss =====
            loss = (CONFIG['w_capacity'] * loss_cap + 
                    0.5 * loss_cvt +
                    CONFIG['w_aspect'] * loss_aspect +
                    CONFIG['w_min_area'] * loss_min_area +
                    CONFIG['w_repel'] * loss_repel)
            
            # Log loss magnitudes at key iterations
            if i == 0 or i == n_iters - 1 or (n_iters >= 100 and (i + 1) % 100 == 0):
                print(f"    [iter {i:4d}] cap={loss_cap.item():.6f} cvt={loss_cvt.item():.6f} "
                      f"asp={loss_aspect.item():.6f} min={loss_min_area.item():.6f} "
                      f"rep={loss_repel.item():.6f} | weighted: "
                      f"cap={CONFIG['w_capacity']*loss_cap.item():.1f} "
                      f"cvt={0.5*loss_cvt.item():.1f} "
                      f"asp={CONFIG['w_aspect']*loss_aspect.item():.1f} "
                      f"min={CONFIG['w_min_area']*loss_min_area.item():.1f} "
                      f"rep={CONFIG['w_repel']*loss_repel.item():.1f}")
            
            loss.backward()
            optimizer.step()
            
            # Clamp
            with torch.no_grad():
                sites[:, 0].clamp_(0, self.norm_w)
                sites[:, 1].clamp_(0, self.norm_h)

                # Keep weak cells alive in hard argmin ownership without inventing geometry.
                area_ratio_det = areas.detach() / (total_pixels.detach() + 1e-6)
                min_ratio_det = max(
                    CONFIG.get('min_cap_ratio', 0.025),
                    CONFIG.get('min_avg_fraction_opt', 0.70) / max(self.n, 1),
                )
                shortage = torch.relu(min_ratio_det - area_ratio_det)
                if torch.any(shortage > 0):
                    weights.add_(CONFIG.get('dead_cell_boost', 0.2) * shortage)

                weights.sub_(weights.mean())
                weights.clamp_(-CONFIG.get('weight_clip', 0.25), CONFIG.get('weight_clip', 0.25))
                self._project_sites_to_foreground_(sites, fg_mask_t, nearest_fg_y_t, nearest_fg_x_t, gw, gh)

            # Debug snapshot every N iterations
            if self.output_dir and self.debug_every > 0:
                if i % self.debug_every == 0 or i == n_iters - 1:
                    self._save_iteration_debug(
                        i,
                        mask_rs,
                        sites,
                        {
                            "total": loss.item(),
                            "cap": loss_cap.item(),
                            "cvt": loss_cvt.item(),
                            "asp": loss_aspect.item(),
                            "ov": 0.0
                        }
                    )

        with torch.no_grad():
            self.optimized_sites_debug = sites.detach().cpu().numpy().copy()

        return sites.detach(), weights.detach()

    def generate_polygons(self, sites, weights):
        """Generate high-quality vector polygons clipped to mask shape."""
        # High-res grid
        scale = CONFIG['render_res'] / CONFIG['resolution']
        rw = int(CONFIG['render_res'] * self.norm_w)
        rh = int(CONFIG['render_res'] * self.norm_h)
        
        xl = np.linspace(0, self.norm_w, rw, dtype=np.float32)
        yl = np.linspace(0, self.norm_h, rh, dtype=np.float32)
        gx, gy = np.meshgrid(xl, yl)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        
        # CPU calculation for large grid
        sites_np = sites.cpu().numpy()
        w_np = weights.cpu().numpy()
        A_np = self.aniso_matrices.cpu().numpy()
        
        labels = np.zeros(len(pts), dtype=np.int32)
        chunk = 200000
        
        for i in range(0, len(pts), chunk):
            end = min(i+chunk, len(pts))
            p = pts[i:end]
            diff = p[:, None, :] - sites_np[None, :, :]
            # d^T A d
            temp = np.einsum('nkj,kij->nki', diff, A_np)
            d_sq = np.sum(diff * temp, axis=2)
            labels[i:end] = np.argmin(d_sq - w_np[None, :], axis=1)
            
        label_map = labels.reshape(rh, rw)
        
        # Apply high-res mask and keep raw power-diagram ownership as-is.
        # Do not rebalance labels here; forcing label coverage creates artificial
        # bubble-like islands that are not true Voronoi/power cells.
        mask_hr = cv2.resize(self.mask_binary, (rw, rh), interpolation=cv2.INTER_NEAREST)
        label_map[mask_hr < 127] = -1

        # Rescue dead cells: if a label has zero support after optimization,
        # seed a tiny ownership island near its site so it does not disappear.
        valid_labels = label_map[label_map >= 0]
        if valid_labels.size > 0:
            counts = np.bincount(valid_labels, minlength=self.n)
        else:
            counts = np.zeros(self.n, dtype=np.int64)
        dead_labels = [i for i in range(self.n) if counts[i] == 0]

        if dead_labels:
            fg = (mask_hr >= 127)

            def _nearest_fg_pixel(start_x, start_y):
                if fg[start_y, start_x]:
                    return start_x, start_y
                max_r = max(rw, rh)
                r = 8
                while r <= max_r:
                    x1 = max(0, start_x - r)
                    x2 = min(rw, start_x + r + 1)
                    y1 = max(0, start_y - r)
                    y2 = min(rh, start_y + r + 1)
                    patch = fg[y1:y2, x1:x2]
                    ys, xs = np.where(patch)
                    if ys.size > 0:
                        xs_abs = xs + x1
                        ys_abs = ys + y1
                        d2 = (xs_abs - start_x) ** 2 + (ys_abs - start_y) ** 2
                        k = int(np.argmin(d2))
                        return int(xs_abs[k]), int(ys_abs[k])
                    r *= 2
                return None, None

            rescue_radius = max(3, int(round(min(rw, rh) * 0.003)))
            rescued = 0
            for label_id in dead_labels:
                sx = int(round((sites_np[label_id, 0] / max(self.norm_w, 1e-6)) * (rw - 1)))
                sy = int(round((sites_np[label_id, 1] / max(self.norm_h, 1e-6)) * (rh - 1)))
                sx = int(np.clip(sx, 0, rw - 1))
                sy = int(np.clip(sy, 0, rh - 1))

                tx, ty = _nearest_fg_pixel(sx, sy)
                if tx is None:
                    continue

                seed = np.zeros((rh, rw), dtype=np.uint8)
                cv2.circle(seed, (tx, ty), rescue_radius, 255, -1)
                seed_mask = (seed > 0) & fg
                if np.any(seed_mask):
                    label_map[seed_mask] = label_id
                    rescued += 1

            if rescued > 0:
                print(f"  [Voronoi] Rescued {rescued}/{len(dead_labels)} dead cells after optimization")
        
        polygons = []
        for i in range(self.n):
            mask_i = (label_map == i).astype(np.uint8)
            mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            mask_i_u8 = (mask_i * 255).astype(np.uint8)
            
            contours, _ = cv2.findContours(mask_i_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print(f"  [Voronoi] Cell {i} has empty support after optimization")
                polygons.append(Polygon())
                continue
            
            # Take largest component
            cnt = max(contours, key=cv2.contourArea)
            
            # Simplify (Approximate Polygon)
            # KEY FIX: Lower epsilon for smoother curves
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, CONFIG['simplify'] * peri, True)
            
            # Normalize to [0, max_dim]
            pts_norm = []
            for pt in approx:
                px, py = pt[0]
                nx = (px / rw) * self.norm_w
                ny = (py / rh) * self.norm_h
                pts_norm.append((nx, ny))
            
            if len(pts_norm) < 3:
                # Keep geometry from the real ownership mask instead of inventing shapes.
                hull = cv2.convexHull(cnt)
                pts_norm = []
                for pt in hull:
                    px, py = pt[0]
                    nx = (px / rw) * self.norm_w
                    ny = (py / rh) * self.norm_h
                    pts_norm.append((nx, ny))
                if len(pts_norm) < 3:
                    print(f"  [Voronoi] Cell {i} contour too small after simplification")
                    polygons.append(Polygon())
                    continue
                
            poly = Polygon(pts_norm)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                print(f"  [Voronoi] Cell {i} became empty after validity repair")
                polygons.append(Polygon())
                continue
            
            # Fix Gaps: Dilate
            buffer = CONFIG['poly_buffer'] / self.max_dim # Convert px to normalized
            poly = poly.buffer(buffer, join_style=2) 
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda g: g.area)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                print(f"  [Voronoi] Cell {i} became empty after buffering")
                polygons.append(Polygon())
                continue
            
            # CLIP TO MASK SHAPE - DISABLED: simplified mask_polygon loses narrow/concave
            # areas causing white gaps. Actual mask clipping is done by the rendering layer.
            # if mask_polygon is not None:
            #     try:
            #         clipped = poly.intersection(mask_polygon)
            #         if clipped.is_valid and not clipped.is_empty and clipped.area > 0.0001:
            #             if clipped.geom_type == 'MultiPolygon':
            #                 clipped = max(clipped.geoms, key=lambda g: g.area)
            #             if clipped.geom_type == 'Polygon':
            #                 poly = clipped
            #     except:
            #         pass
            
            polygons.append(poly)
            
        return polygons
    
    def _create_mask_polygon(self, mask_hr, rw, rh):
        """Create polygon from mask for clipping cells."""
        try:
            contours, _ = cv2.findContours(mask_hr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            cnt = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.003 * peri, True)  # Less aggressive simplification
            
            pts_norm = []
            for pt in approx:
                px, py = pt[0]
                nx = (px / rw) * self.norm_w
                ny = (py / rh) * self.norm_h
                pts_norm.append((nx, ny))
            
            if len(pts_norm) < 3:
                return None
                
            poly = Polygon(pts_norm)
            if not poly.is_valid:
                from shapely.validation import make_valid
                poly = make_valid(poly)
            
            return poly
        except:
            return None

    def match_images_spatial_order(self, polygons: List[Polygon]) -> List[int]:
        """
        Assign images to cells using COLUMN-MAJOR reading order.
        Order is: top-to-bottom within each column, then left-to-right across columns.
        
        Returns:
            assignment: List[int] where assignment[img_idx] = cell_idx
        """
        print("  [Spatial] Assigning images by reading order (COLUMN-MAJOR)...")

        # 1. Collect valid centroids; defer empty/invalid cells to the end
        centroids = []
        invalid_cells = []
        for cell_idx, poly in enumerate(polygons):
            if poly is None or not hasattr(poly, 'centroid') or poly.is_empty:
                invalid_cells.append(cell_idx)
                continue
            cx, cy = poly.centroid.x, poly.centroid.y
            if not np.isfinite(cx) or not np.isfinite(cy):
                invalid_cells.append(cell_idx)
                continue
            centroids.append((cell_idx, cx, cy))

        if not centroids:
            print("  [Spatial] Warning: all cells invalid; fallback to identity assignment")
            return list(range(self.n))

        # 2. Compute X-range and adaptive number of columns
        xs = [c[1] for c in centroids]
        x_min, x_max = min(xs), max(xs)
        x_range = max(x_max - x_min, 1e-6)
        num_cols = max(3, math.ceil(math.sqrt(len(centroids))))
        band_width = x_range / num_cols

        # 3. Assign each cell to a column band by X
        cols = [[] for _ in range(num_cols)]
        for cell_idx, cx, cy in centroids:
            col_idx = int((cx - x_min) / band_width)
            col_idx = min(num_cols - 1, max(0, col_idx))
            cols[col_idx].append((cell_idx, cx, cy))

        # 4. Within each column, sort top-to-bottom by Y
        ordered_cells = []
        for col_idx, col in enumerate(cols):
            col_sorted = sorted(col, key=lambda c: c[2])
            cell_ids = [c[0] for c in col_sorted]
            ordered_cells.extend(cell_ids)
            if cell_ids:
                print(f"    Col {col_idx}: cells {cell_ids}")

        if invalid_cells:
            print(f"  [Spatial] Appending invalid cells at end: {invalid_cells}")
            ordered_cells.extend(invalid_cells)

        if len(ordered_cells) < self.n:
            missing = [idx for idx in range(self.n) if idx not in ordered_cells]
            ordered_cells.extend(missing)

        # 5. Create assignment: image[i] -> ordered_cells[i]
        assignment = [ordered_cells[i] for i in range(self.n)]

        print(f"  [Spatial] Assignment complete: {num_cols} columns, reading order preserved")
        print(f"  [Spatial] Top-left cells: {ordered_cells[:min(3, len(ordered_cells))]}")
        
        return assignment

# =============================================================================
# INTERFACE
# =============================================================================

class WeightedVoronoiLayout:
    def __init__(self, polygon):
        self.polygon = polygon
        self.bounds = polygon.bounds

    def create_layout(self, frame_infos: List[Dict], probabilities: List[float] = None, 
                      use_timeline_order: bool = True, num_iterations: int = 600, 
                      debug_every: int = 0, debug_dir: str = None, verbose: bool = True,
                      mask_path: str = None):
        """
        Create a Voronoi layout for the given images within the polygon.
        
        Args:
            frame_infos: List of dicts with image info (bbox, path, etc.)
            probabilities: Optional list of importance scores (0-1) for each image
            use_timeline_order: If True (DEFAULT), preserve frame timeline order by:
                - Sorting cells by spatial position (reading order: top-left to bottom-right)
                - Assigning image[i] to sorted_cell[i]
                - This ensures frame 0 appears in top-left, frame N in bottom-right
            num_iterations: Number of optimization iterations
            debug_dir: Directory to save debug visualizations
            verbose: Print progress
            mask_path: Path to shape mask image (use directly instead of rendering polygon)
            
        Returns:
            Dict with layout results
        """
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        if verbose:
            print(f"Creating layout for {len(frame_infos)} images in {self.polygon.bounds}")
            if probabilities:
                print(f"  Using probabilities for {'Sizing + ' if use_timeline_order else ''}Optimization")
            if use_timeline_order:
                print(f"  Using TIMELINE ORDER (Reading flow: Top-Left -> Bottom-Right)")

        # 1. Setup Canvas (will be overridden below if mask_path is provided)
        min_x, min_y, max_x, max_y = self.bounds
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        
        # 2. Load mask directly if provided (avoids polygon→mask losing shape holes)
        # NOTE: polygon.bounds are in flipped-Y coordinate space (generate_canvas_polygon
        # calls cv2.flip before contour extraction), so they must NOT be used as direct
        # image-space crop indices — doing so crops the wrong region (e.g. only feet).
        # Instead, use the full mask and reset origin to (0,0).
        mask = None
        if mask_path and os.path.isfile(mask_path):
            full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if full_mask is not None:
                _, full_mask = cv2.threshold(full_mask, 127, 255, cv2.THRESH_BINARY)
                mask = full_mask  # use full mask directly, no polygon-bounds crop
                fm_h, fm_w = full_mask.shape[:2]
                # Override canvas to full mask dimensions; set origin to (0,0) so
                # the final cell-coordinate shift (x+min_x, y+min_y) is a no-op and
                # the resulting cells are already in image pixel space (Y=0 at top).
                min_x, min_y = 0, 0
                max_x, max_y = fm_w, fm_h
                width, height = fm_w, fm_h
                if verbose:
                    fg_pct = np.count_nonzero(mask) / mask.size * 100
                    print(f"  [Voronoi] Using mask directly from {os.path.basename(mask_path)} ({fg_pct:.1f}% foreground)")
        
        # Fallback: Render polygon to mask
        if mask is None:
            mask = np.zeros((height, width), dtype=np.uint8)
            if self.polygon.geom_type == 'MultiPolygon':
                polys = list(self.polygon.geoms)
            else:
                polys = [self.polygon]
            for p in polys:
                coords = np.array(p.exterior.coords)
                coords[:, 0] -= min_x
                coords[:, 1] = (max_y - min_y) - (coords[:, 1] - min_y)
                coords = coords.astype(np.int32)
                cv2.fillPoly(mask, [coords], 255)
            
        temp_path = os.path.join(debug_dir or '.', '_voronoi_temp.png')
        cv2.imwrite(temp_path, mask)
        
        # DEBUG: Stage 1 - Input mask
        if debug_dir:
            debug_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(debug_dir, 'voronoi_debug_1_mask.png'), debug_mask)
        
        # 3. Run Voronoi Engine
        CONFIG['num_iterations'] = num_iterations
        engine = VoronoiLayoutEngine(
            temp_path,
            frame_infos,
            probabilities,
            use_timeline_order=use_timeline_order,
            output_dir=debug_dir,
            debug_every=debug_every
        )

        # === PHASE 1: Initial Optimization ===
        if verbose:
            print(f"  [Phase 1] Initial optimization ({num_iterations} iterations)...")
        sites, weights = engine.optimize()
        raw_polys = engine.generate_polygons(sites, weights)
        
        assignment_idx = engine.match_images_spatial_order(raw_polys)
        
        # Compute IoU scores
        final_iou = engine._compute_cell_bbox_iou(raw_polys, assignment_idx)
        avg_iou = sum(final_iou) / len(final_iou) if final_iou else 0
        if verbose:
            print(f"  [Phase 1] Avg bbox coverage: {avg_iou:.1%}")

        def _gradient_bgr(order_idx, total_items):
            """Publication-friendly CG gradient (deep blue -> teal -> amber -> orange)."""
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

            # Avoid the darkest/lightest extremes and slightly smooth distribution.
            t = 0.08 + 0.84 * t
            t = t ** 0.92

            pos = t * (len(anchors_rgb) - 1)
            lo = int(np.floor(pos))
            hi = min(lo + 1, len(anchors_rgb) - 1)
            a = pos - lo
            rgb = (1.0 - a) * anchors_rgb[lo] + a * anchors_rgb[hi]

            # Slightly compress saturation to keep colors paper-friendly.
            luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            rgb = 0.88 * rgb + 0.12 * luma

            return (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        
        def _save_sites_debug_image(file_name, sites_arr):
            if not debug_dir or sites_arr is None:
                return

            debug_sites = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            scale = engine.max_dim

            for i, (sx, sy) in enumerate(sites_arr):
                px = int(sx * scale)
                py = int(sy * scale)

                color = (255, 180, 0)

                cv2.circle(debug_sites, (px, py), 8, color, -1)
                cv2.putText(
                    debug_sites,
                    str(i),
                    (px + 10, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            cv2.imwrite(os.path.join(debug_dir, file_name), debug_sites)

        def _save_cells_debug_image(file_name, cells, sites_arr=None, assignment_idx=None):
            if not debug_dir or cells is None:
                return

            debug_cells = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            label_draw_items = []
            border_draw_items = []

            scale = engine.max_dim

            cell_to_timeline = None
            if assignment_idx is not None:
                cell_to_timeline = {cell_id: img_id for img_id, cell_id in enumerate(assignment_idx)}

            for i, cell in enumerate(cells):
                if not hasattr(cell, 'exterior') or cell.is_empty:
                    continue

                pts = np.array(cell.exterior.coords) * scale
                if pts.shape[0] < 3:
                    continue
                pts = pts.astype(np.int32)

                if cell_to_timeline:
                    timeline_id = cell_to_timeline.get(i, -1)
                    if timeline_id >= 0:
                        color = _gradient_bgr(timeline_id, len(assignment_idx))
                        label_main = str(timeline_id + 1)
                    else:
                        color = _gradient_bgr(i, len(cells))
                        label_main = str(i + 1)
                else:
                    color = _gradient_bgr(i, len(cells))
                    label_main = str(i + 1)

                cv2.fillPoly(debug_cells, [pts], color)

                cx, cy = int(cell.centroid.x * scale), int(cell.centroid.y * scale)
                border_draw_items.append(pts)
                label_draw_items.append((label_main, cx, cy))

            # Fill neutral cracks/holes left by rasterization or polygon clipping
            # so debug cell renders match the contiguous panel visualization.
            in_shape = mask > 127
            is_black = np.all(debug_cells < 10, axis=2)
            is_white = np.all(debug_cells > 245, axis=2)
            is_hole = in_shape & (is_black | is_white)
            seed_mask = in_shape & (~is_hole)
            if np.any(is_hole) and np.any(seed_mask):
                src = np.where(seed_mask, 0, 1).astype(np.uint8)
                _, labels = cv2.distanceTransformWithLabels(
                    src,
                    cv2.DIST_L2,
                    5,
                    labelType=cv2.DIST_LABEL_PIXEL,
                )
                seed_y, seed_x = np.where(seed_mask)
                nearest_idx = labels[is_hole] - 1
                nearest_idx = np.clip(nearest_idx, 0, len(seed_y) - 1)
                debug_cells[is_hole] = debug_cells[seed_y[nearest_idx], seed_x[nearest_idx]]

            for pts in border_draw_items:
                cv2.polylines(debug_cells, [pts], True, (255, 255, 255), 1)

            for label_main, cx, cy in label_draw_items:
                (tw, th), _ = cv2.getTextSize(label_main, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                tx, ty = int(cx - tw / 2), int(cy + th / 2)
                cv2.putText(debug_cells, label_main, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(debug_cells, label_main, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

            if sites_arr is not None:
                for i, (sx, sy) in enumerate(sites_arr):
                    px = int(sx * scale)
                    py = int(sy * scale)
                    cv2.circle(debug_cells, (px, py), 3, (255, 255, 255), -1)

            cv2.imwrite(os.path.join(debug_dir, file_name), debug_cells)
        
        # DEBUG: Stage 2 - Sites before optimization (initialization)
        if debug_dir:
            _save_sites_debug_image(
                'voronoi_debug_2_sites_before_opt.png',
                engine.initial_sites_debug,
            )

        # DEBUG: Stage 3 - Sites after optimization
        if debug_dir:
            _save_sites_debug_image(
                'voronoi_debug_3_sites_after_opt.png',
                engine.optimized_sites_debug,
            )
            # Keep legacy filename for compatibility with existing workflows.
            _save_sites_debug_image(
                'voronoi_debug_2_sites.png',
                engine.optimized_sites_debug,
            )

        # DEBUG: Voronoi cells before and after optimization
        initial_polys = None
        initial_assignment_idx = None
        if engine.initial_sites_debug is not None:
            try:
                init_sites = torch.tensor(engine.initial_sites_debug, device=DEVICE, dtype=torch.float32)
                init_weights = torch.zeros(len(engine.initial_sites_debug), device=DEVICE, dtype=torch.float32)
                initial_polys = engine.generate_polygons(init_sites, init_weights)
                # Keep labels in timeline space for consistency across all debug images.
                if initial_polys is not None:
                    initial_assignment_idx = engine.match_images_spatial_order(initial_polys)
            except Exception as e:
                if verbose:
                    print(f"  [DEBUG] Could not generate pre-opt Voronoi cells: {e}")

        if debug_dir:
            _save_cells_debug_image(
                'voronoi_debug_2_cells_before_opt.png',
                initial_polys,
                sites_arr=engine.initial_sites_debug,
                assignment_idx=initial_assignment_idx,
            )
            _save_cells_debug_image(
                'voronoi_debug_3_cells_after_opt.png',
                raw_polys,
                sites_arr=engine.optimized_sites_debug,
                assignment_idx=assignment_idx,
            )
            # Keep legacy filename for compatibility with existing workflows.
            _save_cells_debug_image(
                'voronoi_debug_3_cells.png',
                raw_polys,
                sites_arr=engine.optimized_sites_debug,
                assignment_idx=assignment_idx,
            )
        
        # Final IoU report
        final_iou = engine._compute_cell_bbox_iou(raw_polys, assignment_idx)
        if debug_dir:
            iou_report = {
                "avg": float(avg_iou),
                "per_image": [{"img": int(i), "iou": float(iou), "cell": int(assignment_idx[i])} 
                             for i, iou in enumerate(final_iou)]
            }
            import json
            with open(os.path.join(debug_dir, 'bbox_iou_report.json'), 'w') as f:
                json.dump(iou_report, f, indent=2)
        
        # EXTENDED DEBUG: Visualizations
        if debug_dir:
            pass  # Additional debug viz can be added here
        
        # DEBUG: Stage 4 - Assignment visualization
        if debug_dir:
            debug_assign = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            scale = engine.max_dim
            for img_id, cell_id in enumerate(assignment_idx):
                cell = raw_polys[cell_id]
                if hasattr(cell, 'exterior') and not cell.is_empty:
                    pts = np.array(cell.exterior.coords) * scale
                    if pts.shape[0] < 3:
                        continue
                    pts = pts.astype(np.int32)
                    # Color follows timeline order with a smooth gradient.
                    color = _gradient_bgr(img_id, len(assignment_idx))
                    cv2.fillPoly(debug_assign, [pts], color)
                    cx, cy = int(cell.centroid.x * scale), int(cell.centroid.y * scale)
                    label = str(img_id + 1)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    tx, ty = int(cx - tw / 2), int(cy + th / 2)
                    cv2.putText(debug_assign, label, (tx, ty),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(debug_assign, label, (tx, ty),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(debug_dir, 'voronoi_debug_4_assignment.png'), debug_assign)
        
        # DEBUG: Stage 5 - Analysis (cell sizes, center dominance, assignment costs)
        if debug_dir:
            scale = engine.max_dim
            
            # Calculate mask centroid
            mask_cx = width / 2
            mask_cy = height / 2
            
            # Collect cell statistics
            cell_stats = []
            for i, cell in enumerate(raw_polys):
                if hasattr(cell, 'exterior') and not cell.is_empty:
                    area = cell.area * (scale ** 2)  # Convert to pixel area
                    cx, cy = cell.centroid.x * scale, cell.centroid.y * scale
                    dist_to_center = np.sqrt((cx - mask_cx)**2 + (cy - mask_cy)**2)
                    
                    minx, miny, maxx, maxy = cell.bounds
                    cell_w = (maxx - minx) * scale
                    cell_h = (maxy - miny) * scale
                    aspect = cell_w / (cell_h + 1e-6)
                    
                    cell_stats.append({
                        'cell_id': i,
                        'area': area,
                        'aspect': aspect,
                        'dist_to_center': dist_to_center,
                        'cx': cx,
                        'cy': cy
                    })
            
            # Find which image is assigned to each cell
            cell_to_image = {}
            for img_id, cell_id in enumerate(assignment_idx):
                cell_to_image[cell_id] = img_id
            
            # Calculate assignment costs (for explanation)
            t_aspects = engine.target_aspects.cpu().numpy()
            
            # Create analysis visualization  
            debug_analysis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Sort by area for heat coloring
            sorted_by_area = sorted(cell_stats, key=lambda x: x['area'], reverse=True)
            max_area = sorted_by_area[0]['area'] if sorted_by_area else 1
            
            for stat in cell_stats:
                cell_id = stat['cell_id']
                cell = raw_polys[cell_id]
                
                if hasattr(cell, 'exterior') and not cell.is_empty:
                    pts = np.array(cell.exterior.coords) * scale
                    if pts.shape[0] < 3:
                        continue
                    pts = pts.astype(np.int32)
                    
                    # Heat color: larger area = more red, smaller = more blue
                    heat = stat['area'] / max_area
                    color = (int(255 * (1 - heat)), 0, int(255 * heat))  # Blue to Red
                    
                    cv2.fillPoly(debug_analysis, [pts], color)
                    cv2.polylines(debug_analysis, [pts], True, (255, 255, 255), 1)
                    
                    # Label with area
                    cx, cy = int(stat['cx']), int(stat['cy'])
                    area_k = stat['area'] / 1000
                    cv2.putText(debug_analysis, f"{area_k:.1f}k", (cx-20, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Mark center of mask
            cv2.circle(debug_analysis, (int(mask_cx), int(mask_cy)), 10, (0, 255, 0), 3)
            cv2.putText(debug_analysis, "CENTER", (int(mask_cx)-30, int(mask_cy)-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(os.path.join(debug_dir, 'voronoi_debug_5_analysis.png'), debug_analysis)
            
            # Write text summary
            with open(os.path.join(debug_dir, 'voronoi_debug_summary.txt'), 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("VORONOI LAYOUT ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Total cells: {len(cell_stats)}\n")
                f.write(f"Mask center: ({mask_cx:.0f}, {mask_cy:.0f})\n\n")
                
                # Top 5 largest cells
                f.write("TOP 5 LARGEST CELLS:\n")
                f.write("-" * 40 + "\n")
                for rank, stat in enumerate(sorted_by_area[:5]):
                    img_id = cell_to_image.get(stat['cell_id'], -1)
                    img_name = frame_infos[img_id]['filename'] if img_id >= 0 else "N/A"
                    f.write(f"  {rank+1}. Cell {stat['cell_id']}: Area={stat['area']:.0f}px, "
                           f"Assigned to img{img_id} ({img_name})\n")
                
                f.write("\n")
                
                # Top 5 most central cells
                sorted_by_center = sorted(cell_stats, key=lambda x: x['dist_to_center'])
                f.write("TOP 5 MOST CENTRAL CELLS:\n")
                f.write("-" * 40 + "\n")
                for rank, stat in enumerate(sorted_by_center[:5]):
                    img_id = cell_to_image.get(stat['cell_id'], -1)
                    img_name = frame_infos[img_id]['filename'] if img_id >= 0 else "N/A"
                    f.write(f"  {rank+1}. Cell {stat['cell_id']}: Dist={stat['dist_to_center']:.0f}px, "
                           f"Assigned to img{img_id} ({img_name})\n")
                
                f.write("\n")
                
                # Assignment reasoning
                f.write("ASSIGNMENT REASONING (aspect ratio matching):\n")
                f.write("-" * 40 + "\n")
                for img_id, cell_id in enumerate(assignment_idx):
                    img_aspect = t_aspects[img_id]
                    cell_stat = next((s for s in cell_stats if s['cell_id'] == cell_id), None)
                    if cell_stat:
                        cell_aspect = cell_stat['aspect']
                        cost = abs(np.log(img_aspect) - np.log(cell_aspect))
                        f.write(f"  img{img_id} -> cell{cell_id}: "
                               f"img_aspect={img_aspect:.2f}, cell_aspect={cell_aspect:.2f}, "
                               f"cost={cost:.3f}\n")
                
                f.write("\n" + "=" * 60 + "\n")
        
        # 5. Final Transform
        final_cells = []
        final_assignments = {}
        
        scale_val = engine.max_dim
        
        # Re-order cells based on assignment: Cell 0 is for Image 0
        for img_id, cell_id in enumerate(assignment_idx):
            poly = raw_polys[cell_id]
            if poly is None or poly.is_empty or not hasattr(poly, 'exterior'):
                final_cells.append(Polygon())
                final_assignments[img_id] = img_id
                continue
            
            # Scale & Translate back to original coords
            poly = shp_scale(poly, xfact=scale_val, yfact=scale_val, origin=(0,0))
            
            # Shift
            coords = list(poly.exterior.coords)
            new_coords = []
            for x, y in coords:
                new_coords.append((x + min_x, y + min_y))
                
            final_cells.append(Polygon(new_coords))
            final_assignments[img_id] = img_id # Direct map because we sorted cells
        
        return {
            'success': True,
            'cells': final_cells,
            'assignments': final_assignments,
            'dims': (width, height)
        }

def convert_voronoi_to_slicing_format(layout_result):
    cells = layout_result.get('cells', [])
    assignments = layout_result.get('assignments', {})
    
    parts_dict = {}
    mapping = {}
    
    for i, cell in enumerate(cells):
        if hasattr(cell, 'exterior') and not cell.is_empty:
            coords = list(cell.exterior.coords)
            if len(coords) >= 3:
                parts_dict[i] = [[float(c[0]), float(c[1])] for c in coords]
            else:
                parts_dict[i] = []
        else:
            parts_dict[i] = []
            
        mapping[i] = i
        
    return parts_dict, mapping, []

# =============================================================================
# CLI
# =============================================================================

def render_collage(layout_result, images_pixel, output_size):
    """
    Paste images into Voronoi cells.
    
    Args:
        layout_result: Result from create_layout
        images_pixel: List of numpy arrays (images)
        output_size: (width, height) of output canvas
    
    Returns:
        Canvas with images pasted into cells
    """
    W, H = output_size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    
    cells = layout_result['cells']
    assignments = layout_result['assignments']  # cell_idx -> img_idx (reordered)
    
    for cell_idx, cell in enumerate(cells):
        if cell.is_empty:
            continue
            
        # Find which image is assigned to this cell
        img_idx = cell_idx  # After reordering in create_layout, cell_idx = img_idx
        
        if img_idx >= len(images_pixel):
            continue
            
        img_source = images_pixel[img_idx]
        if img_source is None:
            continue

        # Get cell bounding box
        minx, miny, maxx, maxy = cell.bounds
        minx, miny = int(minx), int(miny)
        maxx, maxy = int(maxx + 1), int(maxy + 1)
        
        target_w = maxx - minx
        target_h = maxy - miny
        
        if target_w <= 0 or target_h <= 0:
            continue

        # Resize image using "Aspect Fill" (Center Crop)
        h_src, w_src = img_source.shape[:2]
        scale = max(target_w / w_src, target_h / h_src)
        
        new_w = int(w_src * scale)
        new_h = int(h_src * scale)
        
        resized_img = cv2.resize(img_source, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center crop
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        
        cropped_img = resized_img[start_y:start_y+target_h, start_x:start_x+target_w]
        
        # Ensure exact size
        if cropped_img.shape[:2] != (target_h, target_w):
            cropped_img = cv2.resize(cropped_img, (target_w, target_h))

        # Create local polygon mask
        local_poly_pts = []
        for x, y in cell.exterior.coords:
            local_poly_pts.append([int(x - minx), int(y - miny)])
            
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(local_poly_pts)], 255)
        
        # Paste to canvas with mask
        roi = canvas[miny:miny+target_h, minx:minx+target_w]
        
        img_masked = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
        mask_inv = cv2.bitwise_not(mask)
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        dst = cv2.add(roi_bg, img_masked)
        
        canvas[miny:miny+target_h, minx:minx+target_w] = dst
        
    return canvas


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python voronoi_layout.py <mask_path> <img_folder> <out_dir>")
        sys.exit(1)
        
    mask_p = sys.argv[1]
    img_f = sys.argv[2]
    out_d = sys.argv[3]
    os.makedirs(out_d, exist_ok=True)
    
    # Load images
    images_info = []
    images_pixel = []
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    file_list = sorted([f for f in os.listdir(img_f) if f.lower().endswith(valid_extensions)])
    
    print(f"Loading {len(file_list)} images...")
    
    for f in file_list:
        path = os.path.join(img_f, f)
        im = cv2.imread(path)
        if im is not None:
            h, w = im.shape[:2]
            images_info.append({'filename': f, 'frame_size': (w, h), 'bbox': [0,0,w,h]})
            images_pixel.append(im)
    
    if not images_info:
        print("No valid images found!")
        sys.exit(1)
    
    print(f"Running layout optimization for {len(images_info)} images...")
    
    # Load mask
    mask = cv2.imread(mask_p, 0)
    if mask is None:
        print("Mask not found, creating default box.")
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.rectangle(mask, (50,50), (950,950), 255, -1)
        
    h, w = mask.shape
    poly = box(0, 0, w, h)
    
    layout = WeightedVoronoiLayout(poly)
    res = layout.create_layout(images_info, num_iterations=800, debug_dir=out_d)
    
    # Render final collage
    print("Rendering final collage...")
    final_image = render_collage(res, images_pixel, (w, h))
    
    # Save result
    output_path = os.path.join(out_d, "final_collage.jpg")
    cv2.imwrite(output_path, final_image)
    
    # Version with borders
    for cell in res['cells']:
        pts = np.array(cell.exterior.coords).astype(np.int32)
        cv2.polylines(final_image, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
        
    cv2.imwrite(os.path.join(out_d, "final_collage_bordered.jpg"), final_image)
    
    print(f"Done! Saved to {output_path}")