#!/usr/bin/env python
# coding: utf-8
"""
Anchor-First Voronoi Layout: Existence Before Allocation

This module implements an anchor-first layout optimization for shape-aware
video collage generation.  Cell existence is ensured through two complementary
phases that are both described and attributed in the paper:

**Phase 1 — Optimization-time anchor guarantee**
Each keyframe is assigned a guaranteed disjoint core region inside the target
shape before gradient descent begins.  Cores are topology-aware, spatially
separated, and provide minimum support so that every cell has a non-zero region
throughout optimization.  The free shell region is then competed for via a
weighted anisotropic power diagram with three losses:

  L_cap_res  — residual capacity matching (free-region capacity only)
  L_cvt_norm — normalized centroidal regularity (area-weighted site-centroid)
  L_fea      — one-sided feasibility penalty (normalized deficit below 70%
               of each cell's target share; gradient = 0 when area >= floor)

The optimization and polygon-extraction steps both use the same anisotropic
power diagram (d^T A d) with mean-normalized matrices so that uniform-aspect
inputs remain isotropic.

Final extraction reads the optimized anisotropic power diagram directly
under the foreground mask, without rescue/repair passes.
"""

import os
import sys
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import deque
from shapely.geometry import Polygon, box
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
    # ── 10 core user-facing parameters ───────────────────────────────────────
    'resolution': 512,              # Optimization grid size
    'num_iterations': 400,          # Gradient-descent steps (synced with run.py)
    'lr_sites': 0.01,               # Site learning rate
    'lr_weights': 0.02,             # Weight learning rate
    'tau': 60.0,                    # Softmax temperature for power-diagram
    'w_capacity': 50.0,             # L_cap_res loss weight
    'w_cvt_norm': 35.0,             # L_cvt_norm loss weight
    'w_fea': 28.0,                  # L_fea loss weight
    'warmup_ratio': 0.20,           # Stage-1 (site warmup) fraction of total iters
    'core_alpha': 0.30,             # Core quota = core_alpha × avg_cell_area

    # ── Ablation switches (set by run.py, not user-facing hyperparams) ───────
    'enable_loss_cap': True,        # L_cap_res on/off
    'ablation_mode': None,          # 'wo_cap' | 'wo_cvt' | 'wo_fea' | None

    # ── Derived / hardcoded internals (do not tune) ───────────────────────────
    # Anchor / core
    'core_target_beta': 0.85,
    'core_component_max_fill': 0.72,
    # shell radii — derived at runtime from core_alpha (see _generate_anchor_maps)
    'shell_radius_avg_factor': 2.20,
    'shell_radius_target_factor': 1.10,
    'shell_radius_core_factor': 1.80,
    'anchor_strength': 2.0,
    # Capacity ramp — derived from warmup_ratio at runtime
    'capacity_ramp_power': 1.5,
    'stage2_weight_lr_scale': 0.35,
    'trust_radius_factor': 3.0,
    # Geometry
    'render_res': 4096,
    'simplify': 0.0001,
    'poly_buffer': 2.0,
    # Cell size floor — derived at runtime from n
    'min_avg_fraction_target': 0.75,
    'min_avg_fraction_opt': 0.70,
    'capacity_uniform_blend': 0.55,
    # Debug
    'debug_every': 0,
    'debug_medial_axis': True,
    'medial_ridge_threshold': 0.39,
    'site_init_method': 'medial_axis',
}

def _apply_env_overrides() -> None:
    """Override CONFIG values from environment variables.

    Intended for batch tuning without editing source.
    """
    env_map = {
        'CAST_VORONOI_W_CAPACITY': ('w_capacity', float),
        'CAST_VORONOI_W_CVT_NORM': ('w_cvt_norm', float),
        'CAST_VORONOI_W_FEA': ('w_fea', float),
        'CAST_VORONOI_LR_SITES': ('lr_sites', float),
        'CAST_VORONOI_LR_WEIGHTS': ('lr_weights', float),
        'CAST_VORONOI_NUM_ITERATIONS': ('num_iterations', int),
        'CAST_VORONOI_TAU': ('tau', float),
        'CAST_VORONOI_WARMUP_RATIO': ('warmup_ratio', float),
        'CAST_VORONOI_CORE_ALPHA': ('core_alpha', float),
    }

    for env_key, (cfg_key, cast_fn) in env_map.items():
        raw = os.getenv(env_key)
        if raw is None or raw == '':
            continue
        try:
            CONFIG[cfg_key] = cast_fn(raw)
            print(f"[CONFIG] {cfg_key} <- {env_key}={raw}")
        except ValueError:
            print(f"[WARN] Invalid {env_key}={raw}; using default {cfg_key}={CONFIG[cfg_key]}")


_apply_env_overrides()
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
        # min_cap_ratio derived from n: 0.5/n ensures a reasonable floor
        min_cap = max(
            0.5 / max(self.n, 1),
            CONFIG.get('min_avg_fraction_target', 0.75) / max(self.n, 1),
        )
        norm_caps = [max(c, min_cap) for c in norm_caps]
        
        # Re-normalize after clamping
        total_clamped = sum(norm_caps)
        norm_caps = [c / total_clamped for c in norm_caps]
        
        self.target_caps = torch.tensor(norm_caps, device=DEVICE, dtype=torch.float32)
        self.target_aspects = torch.tensor(aspects, device=DEVICE, dtype=torch.float32)
        
        # All frames have equal priority in optimization weights (separate from
        # target capacity sizing that may still be influenced by bbox/probability).
        bbox_priorities = [1.0 for _ in self.frame_infos]
        self.bbox_priorities = torch.tensor(bbox_priorities, device=DEVICE, dtype=torch.float32)

        if self.probabilities and any(abs(float(p) - 1.0) > 1e-6 for p in self.probabilities):
            pmin = float(min(self.probabilities))
            pmax = float(max(self.probabilities))
            print(
                f"  [Voronoi] Capacity sizing uses probabilities (range={pmin:.2f}-{pmax:.2f}); "
                f"optimization priority remains uniform"
            )
        else:
            print(
                f"  [Voronoi] Capacity sizing from bbox geometry; optimization priority uniform "
                f"across {self.n} frames"
            )

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
                cv2.polylines(debug, [pts], False, (0, 140, 255), 3, lineType=cv2.LINE_AA)

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

            # Export RGBA so the outside-black background is transparent.
            alpha = np.where(self.mask_binary > 0, 255, 0).astype(np.uint8)
            debug_rgba = np.dstack([debug, alpha])
            cv2.imwrite(
                os.path.join(self.output_dir, 'voronoi_debug_1b_medial_axis_endpoints.png'),
                debug_rgba,
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
            
            # ===== STEP 1: Place sites at endpoints (with spatial spread) =====
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
                    
                    # Minimum pixel distance between accepted endpoints.
                    # Prevents clustering when many endpoints are in one region
                    # (e.g. head of a human silhouette).  Skipped endpoints are
                    # replaced by DT-filled sites which have better spatial spread.
                    fg_area = max(1, int(np.count_nonzero(self.mask_binary > 127)))
                    _min_ep_dist = math.sqrt(fg_area / max(self.n, 1)) * 0.65
                    # Cap: accept at most ~50% from endpoints so DT can fill gaps
                    _max_ep = max(2, int(math.ceil(self.n * 0.5)))
                    
                    # Add endpoints as priority sites (with spread filter)
                    for v in end_vertices_sorted:
                        if len(endpoint_sites) >= min(self.n, _max_ep):
                            break
                        
                        x = float(G.nodes[v]['x'])
                        y = float(G.nodes[v]['y'])

                        # Skeleton graph coordinates are unified in image space
                        # (top-left origin), same as mask row/col.
                        row, col = sd.xy2rowcol(x, y, h)
                        check_x = int(np.clip(col, 0, w - 1))
                        check_y = int(np.clip(row, 0, h - 1))
                        endpoint_candidates.append((check_x, check_y))
                        
                        if self.mask_binary[check_y, check_x] > 127:
                            # Spatial spread check: reject if too close to an
                            # already-accepted endpoint.
                            too_close = False
                            for ax, ay in accepted_endpoint_pixels:
                                if math.hypot(check_x - ax, check_y - ay) < _min_ep_dist:
                                    too_close = True
                                    break
                            if too_close:
                                rejected_endpoint_pixels.append((check_x, check_y))
                                continue
                            
                            norm_x = (check_x / w) * self.norm_w
                            norm_y = (check_y / h) * self.norm_h
                            endpoint_sites.append([norm_x, norm_y])
                            accepted_endpoint_pixels.append((check_x, check_y))
                        else:
                            rejected_endpoint_pixels.append((check_x, check_y))
                    
                    print(f"  [Voronoi] ✓ Placed {len(endpoint_sites)} sites at endpoints "
                          f"(spread filter: min_dist={_min_ep_dist:.0f}px, max_ep={_max_ep})")
                    
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
        
        NOTE: Timeline order no longer constrains ASSIGNMENT; matching is free-form,
        NOT initialization. Medial axis init is always preferred because:
        1. Sites start inside the shape (Grid places many outside the mask → waste)
        2. Narrow regions (ears, tails) get dedicated sites automatically
        3. Free image-cell matching can swap frames across anchors after optimization
        """
        method = CONFIG.get('site_init_method', 'distance_transform')
        
        if method == 'medial_axis':
            return self._init_sites_medial_axis()
        elif method == 'hybrid':
            # Hybrid: Start with medial axis, fill with distance transform
            return self._init_sites_medial_axis()  # Already has fallback built-in
        else:
            return self._init_sites_smart()
    
    def _generate_anchor_maps(self, sites, gh, gw, mask_tensor, target_areas):
        """
        Build automatic Core + Local-Shell priors.

        Core:
            - quota is auto-scaled from current foreground geometry
            - connected region growing from each seed inside the valid mask
        Shell:
            - local competition support around each core
            - residual capacity will be optimized only on this shell support

        Args:
            sites: torch.Tensor [n, 2], normalized seed sites
            gh, gw: optimization-grid height/width
            mask_tensor: torch.Tensor [gh, gw], foreground mask in {0,1}
            target_areas: torch.Tensor [n], per-cell target areas in grid pixels

        Returns:
            core_maps_disjoint: torch.Tensor [gh, gw, n], hard disjoint cores
            core_union: torch.Tensor [gh, gw], binary union of all cores
            shell_support: torch.Tensor [gh, gw, n], allowed shell competition support
            anchor_radius_norm: float, equivalent mean core radius in normalized units
            core_quota_px_t: torch.Tensor [n], target core quotas in pixels
            core_area_px_t: torch.Tensor [n], realized core areas in pixels
            core_owner_np: np.ndarray [gh, gw], owner id per core pixel or -1
        """
        fg_mask = (mask_tensor.detach().cpu().numpy() > 0.5)
        total_fg_px = int(np.count_nonzero(fg_mask))
        avg_cell_area_px = float(total_fg_px) / max(self.n, 1)

        target_np = target_areas.detach().cpu().numpy().astype(np.float32)

        alpha = float(np.clip(CONFIG.get('core_alpha', 0.30), 0.05, 0.95))
        beta = float(np.clip(CONFIG.get('core_target_beta', 0.85), 0.20, 1.20))
        # core_floor_ratio derived from core_alpha: floor = 0.5 * alpha
        core_floor_ratio = float(np.clip(0.5 * alpha, 0.0, 1.0))

        base_quota_px = alpha * avg_cell_area_px
        core_quota_px = np.minimum(np.full(self.n, base_quota_px, dtype=np.float32), beta * target_np)
        core_floor_px = max(1.0, core_floor_ratio * base_quota_px)
        core_quota_px = np.maximum(core_quota_px, core_floor_px)
        core_quota_px = np.clip(core_quota_px, 1.0, None)

        # Seed projection to nearest valid foreground pixel.
        seed_x = np.round((sites[:, 0].detach().cpu().numpy() / max(self.norm_w, 1e-6)) * (gw - 1)).astype(np.int32)
        seed_y = np.round((sites[:, 1].detach().cpu().numpy() / max(self.norm_h, 1e-6)) * (gh - 1)).astype(np.int32)
        seed_x = np.clip(seed_x, 0, gw - 1)
        seed_y = np.clip(seed_y, 0, gh - 1)

        if np.any(~fg_mask):
            nearest_fg = ndimage.distance_transform_edt(
                ~fg_mask,
                return_distances=False,
                return_indices=True,
            )
            outside = ~fg_mask[seed_y, seed_x]
            if np.any(outside):
                oy = seed_y[outside]
                ox = seed_x[outside]
                seed_y[outside] = nearest_fg[0, oy, ox]
                seed_x[outside] = nearest_fg[1, oy, ox]

        seed_pixels = [(int(seed_y[i]), int(seed_x[i])) for i in range(self.n)]

        # Component-aware quota scaling to keep local shell room in each topology component.
        cc_count, cc_labels = cv2.connectedComponents(fg_mask.astype(np.uint8), connectivity=8)
        seed_cc = np.array([cc_labels[py, px] for py, px in seed_pixels], dtype=np.int32)
        comp_fill_cap = float(np.clip(CONFIG.get('core_component_max_fill', 0.72), 0.35, 0.95))

        for cc_id in np.unique(seed_cc):
            if cc_id <= 0:
                continue
            idx = np.where(seed_cc == cc_id)[0]
            if len(idx) == 0:
                continue
            comp_area = float(np.count_nonzero(cc_labels == cc_id))
            if comp_area <= 0:
                continue
            comp_quota_cap = comp_fill_cap * comp_area
            cur_sum = float(core_quota_px[idx].sum())
            if cur_sum > comp_quota_cap and cur_sum > 1e-6:
                core_quota_px[idx] *= (comp_quota_cap / cur_sum)

        core_quota_int = np.maximum(1, np.floor(core_quota_px).astype(np.int32))

        # Connected multi-source region growing (disjoint cores).
        owner = np.full((gh, gw), -1, dtype=np.int32)
        core_counts = np.zeros(self.n, dtype=np.int32)
        queues = [deque() for _ in range(self.n)]
        visited = [np.zeros((gh, gw), dtype=np.uint8) for _ in range(self.n)]
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for i, (py, px) in enumerate(seed_pixels):
            queues[i].append((py, px))
            visited[i][py, px] = 1

        max_steps = max(1, total_fg_px * 6)
        steps = 0
        made_progress = True
        while made_progress and steps < max_steps:
            steps += 1
            made_progress = False
            for i in range(self.n):
                if core_counts[i] >= core_quota_int[i]:
                    continue
                q = queues[i]
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in neighbors:
                        ny = cy + dy
                        nx = cx + dx
                        if ny < 0 or ny >= gh or nx < 0 or nx >= gw:
                            continue
                        if not fg_mask[ny, nx] or visited[i][ny, nx]:
                            continue
                        visited[i][ny, nx] = 1
                        q.append((ny, nx))

                    if fg_mask[cy, cx] and owner[cy, cx] < 0:
                        owner[cy, cx] = i
                        core_counts[i] += 1
                        made_progress = True
                        break

        core_union_np = (owner >= 0) & fg_mask
        core_maps_np = np.zeros((gh, gw, self.n), dtype=np.float32)
        if np.any(core_union_np):
            ys, xs = np.where(core_union_np)
            core_maps_np[ys, xs, owner[ys, xs]] = 1.0

        # Local shell support masks (topology-aware + auto-scaled by geometry).
        free_np = fg_mask & (~core_union_np)
        avg_radius_px = math.sqrt(max(avg_cell_area_px, 1.0) / math.pi)
        target_radius_px = np.sqrt(np.maximum(target_np, 1.0) / math.pi)
        core_radius_px = np.sqrt(np.maximum(core_quota_px, 1.0) / math.pi)
        r_avg = float(np.clip(CONFIG.get('shell_radius_avg_factor', 2.2), 1.1, 5.0))
        r_target = float(np.clip(CONFIG.get('shell_radius_target_factor', 1.1), 0.2, 4.0))
        r_core = float(np.clip(CONFIG.get('shell_radius_core_factor', 1.8), 1.0, 4.0))
        shell_radius_px = r_avg * avg_radius_px + r_target * target_radius_px
        shell_radius_px = np.maximum(shell_radius_px, r_core * core_radius_px)

        yy, xx = np.indices((gh, gw), dtype=np.float32)
        shell_support_np = np.zeros((gh, gw, self.n), dtype=np.float32)
        for i in range(self.n):
            same_cc = (cc_labels == seed_cc[i])
            core_i = (owner == i)
            if np.any(core_i):
                dist_i = ndimage.distance_transform_edt(~core_i)
            else:
                py, px = seed_pixels[i]
                dist_i = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
            allow_i = (dist_i <= shell_radius_px[i]) & same_cc & free_np
            shell_support_np[:, :, i] = allow_i.astype(np.float32)

        # Ensure every free pixel belongs to at least one local shell support.
        any_allow = np.max(shell_support_np, axis=-1) > 0.5
        uncovered = free_np & (~any_allow)
        if np.any(uncovered):
            seed_yf = np.array([p[0] for p in seed_pixels], dtype=np.float32)
            seed_xf = np.array([p[1] for p in seed_pixels], dtype=np.float32)
            comp_vals = np.unique(cc_labels[uncovered])
            for cc_id in comp_vals:
                if cc_id <= 0:
                    continue
                pix_mask = uncovered & (cc_labels == cc_id)
                py, px = np.where(pix_mask)
                if py.size == 0:
                    continue
                cells = np.where(seed_cc == cc_id)[0]
                if cells.size == 0:
                    cells = np.arange(self.n, dtype=np.int32)
                chunk = 4096
                for st in range(0, py.size, chunk):
                    ed = min(st + chunk, py.size)
                    pyc = py[st:ed].astype(np.float32)
                    pxc = px[st:ed].astype(np.float32)
                    dy = pyc[:, None] - seed_yf[cells][None, :]
                    dx = pxc[:, None] - seed_xf[cells][None, :]
                    nearest = cells[np.argmin(dy * dy + dx * dx, axis=1)]
                    shell_support_np[py[st:ed], px[st:ed], nearest] = 1.0

        core_maps_disjoint = torch.tensor(core_maps_np, device=DEVICE, dtype=torch.float32)
        core_union = torch.tensor(core_union_np.astype(np.float32), device=DEVICE, dtype=torch.float32)
        shell_support = torch.tensor(shell_support_np, device=DEVICE, dtype=torch.float32)
        core_quota_px_t = torch.tensor(core_quota_px, device=DEVICE, dtype=torch.float32)
        core_area_px_t = core_maps_disjoint.sum(dim=(0, 1))

        area_per_px_norm = (self.norm_w * self.norm_h) / max(gw * gh, 1)
        mean_core_area_norm = float(np.mean(np.maximum(core_counts, 1)) * area_per_px_norm)
        anchor_radius_norm = math.sqrt(max(mean_core_area_norm, 1e-9) / math.pi)

        print(
            f"  [Core] quota(avg={base_quota_px:.1f}px, floor={core_floor_px:.1f}px) "
            f"realized={core_area_px_t.mean().item():.1f}px "
            f"min/max={core_area_px_t.min().item():.1f}/{core_area_px_t.max().item():.1f}"
        )

        return (
            core_maps_disjoint,
            core_union,
            shell_support,
            anchor_radius_norm,
            core_quota_px_t,
            core_area_px_t,
            owner,
        )

    def _save_iteration_debug(self, iter_idx, mask_rs, sites, loss_dict, label_map_np=None):
        """Save per-iteration debug snapshot.

        Parameters
        ----------
        label_map_np : np.ndarray | None
            argmin(d_final, dim=-1) from the optimization loop at the optimization
            grid resolution.  When provided, the exact power-diagram ownership is
            visualized (including boundary cells).  Falls back to a site-only
            overlay when None.
        """
        if not self.output_dir:
            return
        if self.debug_every <= 0 and iter_idx != -1:
            return

        # Save loss log (3 losses: cap_res, cvt_norm, fea)
        loss_path = os.path.join(self.output_dir, "voronoi_debug_iter_losses.csv")
        header = "iter,loss_total,loss_cap_res,loss_cvt_norm,loss_fea\n"
        line = (f"{iter_idx},{loss_dict['total']:.6f},{loss_dict['cap_res']:.6f},"
                f"{loss_dict['cvt_norm']:.6f},{loss_dict['fea']:.6f}\n")
        if iter_idx == 0:
            with open(loss_path, "w") as f:
                f.write(header)
                f.write(line)
        else:
            with open(loss_path, "a") as f:
                f.write(line)

        gh, gw = mask_rs.shape[:2]
        scale_x = gw / max(self.norm_w, 1e-6)
        scale_y = gh / max(self.norm_h, 1e-6)
        sites_np = sites.detach().cpu().numpy()

        # Harmonious soft palette (BGR) for cleaner visual grouping.
        CELL_COLORS = [
            (255, 195, 220), (180, 235, 185), (220, 205, 255), (180, 220, 255),
            (170, 235, 235), (255, 210, 170), (205, 230, 140), (235, 190, 245),
            (255, 205, 145), (190, 210, 255),
        ]

        debug_img = np.zeros((gh, gw, 3), dtype=np.uint8)

        if label_map_np is not None:
            # Resize label map to debug image resolution (nearest-neighbor to preserve cell IDs)
            lm = cv2.resize(
                label_map_np.astype(np.int32), (gw, gh), interpolation=cv2.INTER_NEAREST
            )
            # Colorize each cell directly from the actual power-diagram assignment.
            # This shows exactly what the optimizer uses — including every boundary cell.
            for cell_idx in range(self.n):
                cell_mask = (lm == cell_idx)
                if cell_mask.any():
                    debug_img[cell_mask] = CELL_COLORS[cell_idx % len(CELL_COLORS)]
            # Draw stronger black boundaries for better readability.
            for cell_idx in range(self.n):
                cell_mask_u8 = ((lm == cell_idx).astype(np.uint8) * 255)
                contours, _ = cv2.findContours(cell_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_img, contours, -1, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        else:
            # Fallback: draw mask with site dots only
            debug_img = cv2.cvtColor(mask_rs, cv2.COLOR_GRAY2BGR)

        # Draw site markers on top
        for idx, (sx, sy) in enumerate(sites_np):
            px = int(sx * scale_x)
            py = int(sy * scale_y)
            cv2.circle(debug_img, (px, py), 6, (0, 0, 0), -1)
            cv2.circle(debug_img, (px, py), 4, (255, 255, 255), -1)
            cv2.putText(debug_img, str(idx), (px + 8, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(debug_img, str(idx), (px + 8, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Mask out background
        mask_3ch = cv2.cvtColor(mask_rs, cv2.COLOR_GRAY2BGR)
        debug_img = cv2.bitwise_and(debug_img, mask_3ch)

        # Emphasize outer silhouette to keep shape edge clearly visible.
        mask_u8 = (mask_rs > 0).astype(np.uint8) * 255
        outer_contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, outer_contours, -1, (0, 0, 0), 3, lineType=cv2.LINE_AA)

        # Mild unsharp mask to improve perceived crispness.
        blur = cv2.GaussianBlur(debug_img, (0, 0), 1.0)
        debug_img = cv2.addWeighted(debug_img, 1.32, blur, -0.32, 0)

        # Save RGBA so outside-mask black background is fully transparent.
        alpha = np.where(mask_rs > 0, 255, 0).astype(np.uint8)
        debug_img = np.dstack([debug_img, alpha])

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
        
        target_areas = self.target_caps * total_pixels

        # Stage A: automatic Core + Local-Shell priors (self-scaling by geometry)
        with torch.no_grad():
            (
                anchor_maps_disjoint,
                anchor_union,
                shell_support,
                anchor_radius,
                core_quota_px,
                core_areas,
                core_owner_map,
            ) = self._generate_anchor_maps(
                sites.detach(),
                gh,
                gw,
                mask_tensor,
                target_areas,
            )
            anchor_centers = sites.detach().clone()

        self.anchor_maps = anchor_maps_disjoint
        self.anchor_radius = anchor_radius
        self.anchor_centers = anchor_centers
        self.anchor_owner_map = core_owner_map.astype(np.int32)
        self.anchor_union_map = (core_owner_map >= 0).astype(np.uint8)
        self.anchor_maps_disjoint = anchor_maps_disjoint.detach().clone()
        self.core_quota_px = core_quota_px.detach().clone()
        self.core_area_px = core_areas.detach().clone()

        free_mask = mask_tensor * (1.0 - anchor_union)
        free_pixels = free_mask.sum()

        # Residual shell targets after guaranteed core support.
        target_residual = torch.clamp(target_areas - core_areas, min=0.0)
        shell_support = shell_support * free_mask.unsqueeze(-1)

        # ── Normalized anisotropic matrices for optimization ─────────────────
        # Normalize so mean(A_i) = I → when all frames share the same aspect ratio
        # the net effect is purely isotropic (no degenerate diagonal tessellation).
        A_raw = self.aniso_matrices  # (n, 2, 2) on DEVICE
        mean_A = A_raw.mean(dim=0)   # (2, 2)
        try:
            mean_A_inv = torch.linalg.inv(mean_A)
        except Exception:
            mean_A_inv = torch.eye(2, device=DEVICE)
        A_norm = torch.einsum('nij,jk->nik', A_raw, mean_A_inv)  # (n, 2, 2)
        # Keep the exact optimization-time normalized metric for extraction.
        self.A_norm = A_norm.detach().cpu().numpy()

        # Two-stage schedule setup
        n_iters = CONFIG['num_iterations']
        core_fraction = float(anchor_union.sum().item() / (total_pixels.item() + 1e-6))
        warmup_ratio = float(CONFIG.get('warmup_ratio', 0.50))
        # capacity_ramp_ratio derived from warmup_ratio (not a separate hyperparameter)
        ramp_ratio = warmup_ratio + 0.10
        # More guaranteed core support -> gentler and later shell-capacity activation.
        warmup_ratio_eff = min(0.92, warmup_ratio + 0.25 * core_fraction)
        ramp_ratio_eff = min(0.95, ramp_ratio + 0.35 * core_fraction)

        warmup_iters = int(n_iters * warmup_ratio_eff)
        warmup_iters = max(0, min(warmup_iters, max(n_iters - 1, 0)))
        ramp_iters = int(n_iters * ramp_ratio_eff)
        ramp_iters = max(1, ramp_iters)
        stage2_weight_lr = CONFIG['lr_weights'] * CONFIG.get('stage2_weight_lr_scale', 0.60)

        # Trust-region radius around anchor centers (motion constraint, not a loss)
        trust_radius = max(
            anchor_radius * CONFIG.get('trust_radius_factor', 4.0),
            max(self.norm_w / max(gw, 1), self.norm_h / max(gh, 1)) * 2.0,
        )

        reserved_pct = float(anchor_union.sum().item() / (total_pixels.item() + 1e-6) * 100.0)
        print(
            f"  [Schedule] Stage1 warmup={warmup_iters} iters, "
            f"Stage2 ramp={ramp_iters} iters, stage2 weight lr={stage2_weight_lr:.5f}, "
            f"core={core_fraction*100.0:.1f}%"
        )
        print(
            f"  [Residual] reserved={reserved_pct:.2f}% free={(100.0-reserved_pct):.2f}%, "
            f"trust_radius={trust_radius:.4f}"
        )
        
        # NaN coverage check: every free foreground pixel must have at least one
        # shell-support cell. _generate_anchor_maps fills uncovered pixels, but we
        # verify here so any regression surfaces immediately.
        with torch.no_grad():
            uncovered = (shell_support.max(dim=-1).values < 0.5) & (free_mask > 0.5)
            if uncovered.any():
                print(f"  [WARN] {uncovered.sum().item():.0f} uncovered shell pixels — "
                      f"anchor map generation may have a bug")

        optimizer = torch.optim.Adam([
            {'params': sites, 'lr': CONFIG['lr_sites']},
            {'params': weights, 'lr': CONFIG['lr_weights']}
        ])

        # Per-cell raw ownership floor for collapse detection in hard power assignment.
        collapse_threshold = torch.maximum(
            0.02 * self.target_caps.to(DEVICE),
            torch.full_like(self.target_caps.to(DEVICE), 0.005 / max(float(self.n), 1.0)),
        )

        # 3. Two-stage optimization loop
        early_stop_triggered = False
        for i in range(n_iters):
            optimizer.zero_grad()

            # Snapshot current state so we can rollback if this iteration collapses a cell.
            prev_sites = sites.detach().clone()
            prev_weights = weights.detach().clone()

            # Smooth schedule (no hard switch)
            progress = (i + 1) / float(max(n_iters, 1))
            stage = "S1" if progress < warmup_ratio_eff else "S2"

            weight_lr_min = 0.15 * stage2_weight_lr
            weight_lr_max = stage2_weight_lr
            weight_lr_progress = min(1.0, progress / max(warmup_ratio_eff, 1e-6))
            weight_lr = weight_lr_min + (weight_lr_max - weight_lr_min) * weight_lr_progress
            optimizer.param_groups[1]['lr'] = weight_lr

            cap_min = 0.0
            cap_progress = min(1.0, progress / float(max(ramp_ratio_eff, 1e-6)))
            cap_factor = cap_min + (1.0 - cap_min) * (
                cap_progress ** float(CONFIG.get('capacity_ramp_power', 1.5))
            )
            
            # Anisotropic power diagram — consistent with generate_polygons().
            # A_norm is pre-normalized so mean(A_i)=I, avoiding degenerate tessellation
            # when all frames share the same aspect ratio.
            diff = grid_coords.unsqueeze(2) - sites.view(1, 1, self.n, 2)  # (gh,gw,n,2)
            temp = torch.einsum('hwni,nij->hwnj', diff, A_norm)             # (gh,gw,n,2)
            d_sq = (diff * temp).sum(dim=-1)                                # (gh,gw,n)

            # Power Diagram
            d_final = d_sq - weights.view(1, 1, self.n)

            # NaN-safe softmax via logsumexp trick (no nan_to_num needed).
            # Shell competes only inside local shell support masks.
            logits_shell = -CONFIG['tau'] * d_final
            logits_shell = logits_shell.masked_fill(shell_support < 0.5, -1e9)
            # Shift by row-max for numerical stability before exp
            logits_shell = logits_shell - logits_shell.max(dim=-1, keepdim=True).values
            probs_shell = torch.exp(logits_shell) * shell_support
            shell_denom = probs_shell.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            probs_shell = (probs_shell / shell_denom) * free_mask.unsqueeze(-1)
            probs = probs_shell + anchor_maps_disjoint
            
            # ========== Feasibility-first residual-capacity system ==========
            # L_cap_res : residual capacity matching on free region only
            # L_cvt_norm: centroidal regularity
            # L_fea     : one-sided feasibility penalty (normalized)
            
            # Cell areas
            areas = (probs * mask_tensor.unsqueeze(-1)).sum(dim=(0, 1))
            free_areas = probs_shell.sum(dim=(0, 1))
            effective_areas = free_areas + core_areas
            
            # Cell centroids
            coords_w = grid_coords.unsqueeze(2) * probs.unsqueeze(-1)
            centroids = coords_w.sum(dim=(0,1)) / (areas.unsqueeze(-1) + 1e-6)
            
            # ===== 1. L_cap_res: Residual Capacity Loss on local shell =====
            # Normalize by total foreground pixels so this term stays stable
            # across different N and shape sizes (avoids runaway gradients).
            residual_delta = (free_areas - target_residual) / (total_pixels + 1e-6)
            loss_cap_res = (residual_delta ** 2).mean()
            
            # ===== 2. L_cvt_norm: Normalized Centroidal Loss (regularity) =====
            # Weighted by inverse sqrt(area_ratio) to focus on small/risky cells
            area_ratio = effective_areas / (total_pixels + 1e-6)
            area_ratio_safe = torch.clamp(area_ratio, min=0.04)
            loss_cvt_norm = (((sites - centroids) ** 2).sum(dim=1) / torch.sqrt(area_ratio_safe)).mean()
            
            # ===== 3. L_fea: One-sided Feasibility Penalty =====
            # Penalizes cells whose area falls below 70% of their target share.
            # Deficit is normalized by target_ratio so gradient magnitude is
            # scale-invariant w.r.t. cell count.  One-sided: gradient is
            # exactly 0 for cells at or above the floor, so large cells are
            # never forced to shrink (which would collapse cells in narrow shapes).
            target_ratio = self.target_caps.to(DEVICE)  # [n], sums to 1
            fea_floor = 0.82 * target_ratio              # 82% of each cell's target
            deficit = torch.clamp(fea_floor - area_ratio, min=0.0)
            fea_boost = 1.5 - 0.5 * progress
            loss_fea = fea_boost * (((deficit / (target_ratio + 1e-8)) ** 2).sum())

            # Extra collapse guard on shell/free ownership.
            free_ratio = free_areas / (total_pixels + 1e-6)
            min_free_ratio = 0.20 * target_ratio
            collapse_deficit = torch.clamp(min_free_ratio - free_ratio, min=0.0)
            loss_collapse = ((collapse_deficit / (target_ratio + 1e-8)) ** 2).sum()

            # ===== Total Loss (with ablation support) =====
            # 3-loss model: cap_res + cvt_norm + fea
            enable_cap = CONFIG.get('enable_loss_cap', True)
            enable_cvt = True
            enable_fea = True
            
            # Override based on ablation_mode (3-loss ablations only)
            ablation_mode = CONFIG.get('ablation_mode', None)
            if ablation_mode == 'wo_cap':
                enable_cap = False
            elif ablation_mode == 'wo_cvt':
                enable_cvt = False
            elif ablation_mode == 'wo_fea':
                enable_fea = False
            
            # Compute total loss
            loss = torch.tensor(0.0, device=DEVICE)
            if enable_cap:
                loss += CONFIG['w_capacity'] * cap_factor * loss_cap_res
            if enable_cvt:
                loss += CONFIG['w_cvt_norm'] * loss_cvt_norm
            if enable_fea:
                loss += CONFIG['w_fea'] * loss_fea

            loss += 40.0 * loss_collapse
            
            # Log loss magnitudes at key iterations
            if i == 0 or i == n_iters - 1 or (n_iters >= 100 and (i + 1) % 100 == 0):
                cap_w = CONFIG['w_capacity'] * cap_factor * loss_cap_res.item() if enable_cap else 0.0
                cvt_w = CONFIG['w_cvt_norm'] * loss_cvt_norm.item() if enable_cvt else 0.0
                fea_w = CONFIG['w_fea'] * loss_fea.item() if enable_fea else 0.0
                print(f"    [iter {i:4d}][{stage}] cap_res={loss_cap_res.item():.6f} cvt_norm={loss_cvt_norm.item():.6f} "
                      f"fea={loss_fea.item():.6f} | weighted: "
                      f"cap={cap_w:.1f} cvt={cvt_w:.1f} fea={fea_w:.1f} "
                      f"cap_factor={cap_factor:.2f} w_lr={weight_lr:.5f} free={free_pixels.item():.0f} "
                      f"[ENABLED: cap={enable_cap} cvt={enable_cvt} fea={enable_fea}]")
            
            loss.backward()
            optimizer.step()
            
            # Projected gradient descent + trust-region projection around anchors
            with torch.no_grad():
                sites[:, 0].clamp_(0, self.norm_w)
                sites[:, 1].clamp_(0, self.norm_h)
                self._project_sites_to_foreground_(sites, fg_mask_t, nearest_fg_y_t, nearest_fg_x_t, gw, gh)

            # Hard per-iteration collapse guard: if one update collapses any cell,
            # immediately rollback to previous iteration and stop.
            with torch.no_grad():
                diff_post = grid_coords.unsqueeze(2) - sites.view(1, 1, self.n, 2)
                temp_post = torch.einsum('hwni,nij->hwnj', diff_post, A_norm)
                d_sq_post = (diff_post * temp_post).sum(dim=-1)
                d_final_post = d_sq_post - weights.view(1, 1, self.n)
                hard_assign_post = d_final_post.argmin(dim=-1)
                hard_mask = mask_tensor > 0.5

                collapsed_cell = None
                collapsed_ratio = None
                for cell_id in range(self.n):
                    raw_ownership = ((hard_assign_post == cell_id) & hard_mask).sum().float()
                    raw_ratio = raw_ownership / (total_pixels + 1e-6)
                    if raw_ratio < collapse_threshold[cell_id]:
                        collapsed_cell = cell_id
                        collapsed_ratio = raw_ratio
                        break

                if collapsed_cell is not None:
                    sites.copy_(prev_sites)
                    weights.copy_(prev_weights)
                    early_stop_triggered = True
                    safe_iter = max(i - 1, -1)
                    print(
                        f"    [EARLY-STOP] iter {i}: cell {collapsed_cell} fell below threshold "
                        f"(raw ownership {collapsed_ratio.item():.4f} < {collapse_threshold[collapsed_cell].item():.4f})."
                    )
                    print(
                        f"    [EARLY-STOP] Rollback to previous state (iter {safe_iter}) and stop optimization."
                    )
                    break

                # Trust-region motion constraint (not a loss)
                delta = sites - anchor_centers
                dist = torch.norm(delta, dim=1, keepdim=True)
                outside = dist > trust_radius
                if outside.any():
                    dist_sel = dist[outside].view(-1, 1)
                    scaled = delta[outside.squeeze(1)] / (dist_sel + 1e-6)
                    sites[outside.squeeze(1)] = anchor_centers[outside.squeeze(1)] + scaled * trust_radius

                sites[:, 0].clamp_(0, self.norm_w)
                sites[:, 1].clamp_(0, self.norm_h)
                self._project_sites_to_foreground_(sites, fg_mask_t, nearest_fg_y_t, nearest_fg_x_t, gw, gh)

            # Debug snapshot every N iterations
            if self.output_dir and self.debug_every > 0:
                if i % self.debug_every == 0 or i == n_iters - 1:
                    # Pass the actual power-diagram label map so the debug image
                    # shows exactly what the optimizer uses (aniso power diagram),
                    # not an approximation from scipy.spatial.Voronoi.
                    debug_lm = d_final.argmin(dim=-1).detach().cpu().numpy()
                    self._save_iteration_debug(
                        i,
                        mask_rs,
                        sites,
                        {
                            "total": loss.item(),
                            "cap_res": loss_cap_res.item(),
                            "cvt_norm": loss_cvt_norm.item(),
                            "fea": loss_fea.item()
                        },
                        label_map_np=debug_lm,
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
        if hasattr(self, 'A_norm'):
            A_np = self.A_norm
        else:
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
        
        # Apply high-res mask.
        mask_hr = cv2.resize(self.mask_binary, (rw, rh), interpolation=cv2.INTER_NEAREST)
        label_map[mask_hr < 127] = -1

        # Enforce render-time core guarantee on the same render grid.
        core_masks_hr = [None] * self.n
        if hasattr(self, 'anchor_maps_disjoint'):
            core_maps = self.anchor_maps_disjoint.detach().cpu().numpy()  # [gh, gw, n]
            for i in range(self.n):
                core_i = core_maps[:, :, i]
                core_hr = cv2.resize(core_i.astype(np.float32), (rw, rh), interpolation=cv2.INTER_NEAREST)
                core_mask = (core_hr > 0.5) & (mask_hr >= 127)
                core_masks_hr[i] = core_mask
                label_map[core_mask] = i

        # Build full-fidelity mask polygon for clipping (all contours, holes preserved).
        full_mask_poly = self._build_full_mask_polygon(mask_hr, rw, rh)

        polygons = []
        unbuffered_polygons = []
        invalid_cells = []  # Track invalid cells explicitly

        anchor_centers_np = None
        if hasattr(self, 'anchor_centers'):
            if torch.is_tensor(self.anchor_centers):
                anchor_centers_np = self.anchor_centers.detach().cpu().numpy()
            else:
                anchor_centers_np = np.asarray(self.anchor_centers)

        for i in range(self.n):
            # Anchor center in render-resolution pixels (for component selection).
            if anchor_centers_np is not None and i < len(anchor_centers_np):
                ax_n = float(anchor_centers_np[i, 0])
                ay_n = float(anchor_centers_np[i, 1])
            else:
                ax_n = float(sites_np[i, 0])
                ay_n = float(sites_np[i, 1])

            ax_px = int(np.clip((ax_n / max(self.norm_w, 1e-6)) * (rw - 1), 0, rw - 1))
            ay_px = int(np.clip((ay_n / max(self.norm_h, 1e-6)) * (rh - 1), 0, rh - 1))

            # Connected-component selection on hard render-time ownership.
            cell_mask = ((label_map == i) & (mask_hr >= 127)).astype(np.uint8)
            if cell_mask.sum() == 0:
                print(f"  [Voronoi] Cell {i} has empty support after optimization")
                invalid_cells.append(i)
                polygons.append(Polygon())
                unbuffered_polygons.append(Polygon())
                continue

            num_cc, cc_map = cv2.connectedComponents(cell_mask, connectivity=8)
            if num_cc <= 1:
                mask_main = cell_mask
            else:
                main_cc = 0

                # Priority 1: component containing anchor.
                cc_anchor = int(cc_map[ay_px, ax_px]) if (0 <= ay_px < rh and 0 <= ax_px < rw) else 0
                if cc_anchor > 0:
                    main_cc = cc_anchor

                # Priority 2: component with max overlap with core.
                core_mask_i = core_masks_hr[i]
                if main_cc == 0 and core_mask_i is not None and np.any(core_mask_i):
                    overlap_labels = cc_map[core_mask_i]
                    overlap_labels = overlap_labels[overlap_labels > 0]
                    if overlap_labels.size > 0:
                        binc = np.bincount(overlap_labels.astype(np.int32))
                        if binc.size > 1:
                            main_cc = int(np.argmax(binc))

                # Priority 3: largest component.
                if main_cc == 0:
                    cc_sizes = np.bincount(cc_map.ravel())
                    if cc_sizes.size > 1:
                        cc_sizes[0] = 0
                        main_cc = int(np.argmax(cc_sizes))

                if main_cc <= 0:
                    print(f"  [Voronoi] Cell {i} has no connected component after selection")
                    invalid_cells.append(i)
                    polygons.append(Polygon())
                    unbuffered_polygons.append(Polygon())
                    continue

                mask_main = (cc_map == main_cc).astype(np.uint8)

                # If selected component is too small, widen by core first (no bridge heuristics).
                if core_mask_i is not None and np.any(core_mask_i):
                    main_area = int(mask_main.sum())
                    core_area = int(core_mask_i.sum())
                    if main_area < max(16, int(0.75 * core_area)):
                        expanded = ((mask_main > 0) | core_mask_i).astype(np.uint8)
                        num_cc2, cc_map2 = cv2.connectedComponents(expanded, connectivity=8)
                        if num_cc2 > 1:
                            main_cc2 = 0
                            cc_anchor2 = int(cc_map2[ay_px, ax_px]) if (0 <= ay_px < rh and 0 <= ax_px < rw) else 0
                            if cc_anchor2 > 0:
                                main_cc2 = cc_anchor2
                            if main_cc2 == 0:
                                overlap_labels2 = cc_map2[core_mask_i]
                                overlap_labels2 = overlap_labels2[overlap_labels2 > 0]
                                if overlap_labels2.size > 0:
                                    binc2 = np.bincount(overlap_labels2.astype(np.int32))
                                    if binc2.size > 1:
                                        main_cc2 = int(np.argmax(binc2))
                            if main_cc2 == 0:
                                cc_sizes2 = np.bincount(cc_map2.ravel())
                                if cc_sizes2.size > 1:
                                    cc_sizes2[0] = 0
                                    main_cc2 = int(np.argmax(cc_sizes2))
                            if main_cc2 > 0:
                                mask_main = (cc_map2 == main_cc2).astype(np.uint8)
                            else:
                                mask_main = expanded
                        else:
                            mask_main = expanded

            mask_main = cv2.morphologyEx(mask_main, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours((mask_main * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"  [Voronoi] Cell {i} has no valid contour after CC selection")
                invalid_cells.append(i)
                polygons.append(Polygon())
                unbuffered_polygons.append(Polygon())
                continue

            cnt = max(contours, key=cv2.contourArea)
            pts_norm = []
            for pt in cnt:
                px, py = pt[0]
                nx = (px / rw) * self.norm_w
                ny = (py / rh) * self.norm_h
                pts_norm.append((nx, ny))

            if len(pts_norm) < 3:
                print(f"  [Voronoi] Cell {i} contour is degenerate after CC selection")
                invalid_cells.append(i)
                polygons.append(Polygon())
                unbuffered_polygons.append(Polygon())
                continue

            poly = Polygon(pts_norm)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly.is_empty:
                print(f"  [Voronoi] Cell {i} became empty after validity repair")
                invalid_cells.append(i)
                polygons.append(Polygon())
                unbuffered_polygons.append(Polygon())
                continue
            
            # Save unbuffered polygon for evaluation (no overlap by definition)
            unbuffered_polygons.append(poly)
            
            # Fix Gaps: Dilate
            buffer = CONFIG['poly_buffer'] / self.max_dim # Convert px to normalized
            poly = poly.buffer(buffer, join_style=2) 
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda g: g.area)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                print(f"  [Voronoi] Cell {i} became empty after buffering")
                invalid_cells.append(i)
                polygons.append(Polygon())
                continue
            
            # Clip to full mask shape so polygons never extend outside the canvas shape.
            # full_mask_poly is built from ALL contours via unary_union, so narrow arms
            # and concave regions are preserved correctly.
            if full_mask_poly is not None:
                try:
                    clipped = poly.intersection(full_mask_poly)
                    if clipped.is_valid and not clipped.is_empty and clipped.area > 1e-10:
                        if clipped.geom_type == 'MultiPolygon':
                            clipped = max(clipped.geoms, key=lambda g: g.area)
                        if clipped.geom_type == 'Polygon':
                            poly = clipped
                except Exception:
                    pass  # clipping failed: keep unclipped poly

            polygons.append(poly)
        
        # Report invalid cells explicitly
        if invalid_cells:
            print(f"  [Voronoi] WARNING: {len(invalid_cells)}/{self.n} cells are invalid: {invalid_cells}")
            print(f"  [Voronoi] This reflects the true behavior of the anchor-first model.")
        
        return polygons, unbuffered_polygons, invalid_cells

    def _compute_raw_power_label_map(self, sites, weights):
        """Compute raw power-diagram ownership on render grid (mask-clipped only).

        This intentionally skips all stabilization/post-processing steps used by
        generate_polygons (core overwrite, CC selection, morphology, buffering).
        """
        rw = int(CONFIG['render_res'] * self.norm_w)
        rh = int(CONFIG['render_res'] * self.norm_h)

        xl = np.linspace(0, self.norm_w, rw, dtype=np.float32)
        yl = np.linspace(0, self.norm_h, rh, dtype=np.float32)
        gx, gy = np.meshgrid(xl, yl)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        sites_np = sites.detach().cpu().numpy()
        w_np = weights.detach().cpu().numpy()
        if hasattr(self, 'A_norm'):
            A_np = self.A_norm
        else:
            A_np = self.aniso_matrices.detach().cpu().numpy()

        labels = np.zeros(len(pts), dtype=np.int32)
        chunk = 200000
        for i in range(0, len(pts), chunk):
            end = min(i + chunk, len(pts))
            p = pts[i:end]
            diff = p[:, None, :] - sites_np[None, :, :]
            temp = np.einsum('nkj,kij->nki', diff, A_np)
            d_sq = np.sum(diff * temp, axis=2)
            labels[i:end] = np.argmin(d_sq - w_np[None, :], axis=1)

        label_map = labels.reshape(rh, rw)
        mask_hr = cv2.resize(self.mask_binary, (rw, rh), interpolation=cv2.INTER_NEAREST)
        label_map[mask_hr < 127] = -1
        return label_map, rw, rh

    def _extract_raw_power_polygons(self, label_map, rw, rh):
        """Extract polygons directly from raw label-map ownership."""
        polygons = []
        for i in range(self.n):
            cell_mask = ((label_map == i).astype(np.uint8) * 255)
            if np.count_nonzero(cell_mask) == 0:
                polygons.append(Polygon())
                continue

            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                polygons.append(Polygon())
                continue

            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) < 3:
                polygons.append(Polygon())
                continue

            pts_norm = []
            for pt in cnt:
                px, py = pt[0]
                nx = (px / rw) * self.norm_w
                ny = (py / rh) * self.norm_h
                pts_norm.append((nx, ny))

            if len(pts_norm) < 3:
                polygons.append(Polygon())
                continue

            poly = Polygon(pts_norm)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly.is_empty:
                polygons.append(Polygon())
                continue

            polygons.append(poly)
        return polygons

    def generate_raw_power_polygons(self, sites, weights):
        """Public debug helper: raw power-diagram polygons (mask-clipped only)."""
        label_map, rw, rh = self._compute_raw_power_label_map(sites, weights)
        return self._extract_raw_power_polygons(label_map, rw, rh)
    
    def _build_full_mask_polygon(self, mask_hr, rw, rh):
        """Build a full-fidelity mask polygon from ALL contours (including holes).

        Uses unary_union over all exterior contours so narrow arms, concave
        areas and disjoint components are preserved correctly.  This enables
        safe polygon clipping without losing any mask geometry.
        """
        from shapely.ops import unary_union
        try:
            # RETR_CCOMP: two-level hierarchy (outer + hole)
            contours, hierarchy = cv2.findContours(
                mask_hr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
            )
            if not contours or hierarchy is None:
                return None

            hierarchy = hierarchy[0]  # (n, 4): [next, prev, child, parent]
            polys = []
            for idx, (cnt, hier) in enumerate(zip(contours, hierarchy)):
                if hier[3] != -1:
                    continue  # skip holes (handled via buffer(0) on outer)
                if len(cnt) < 3:
                    continue
                pts_norm = [
                    ((pt[0][0] / rw) * self.norm_w,
                     (pt[0][1] / rh) * self.norm_h)
                    for pt in cnt
                ]
                if len(pts_norm) < 3:
                    continue
                p = Polygon(pts_norm)
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty and p.area > 1e-10:
                    polys.append(p)

            if not polys:
                return None
            result = unary_union(polys)
            if not result.is_valid:
                result = result.buffer(0)
            return result if not result.is_empty else None
        except Exception as e:
            print(f"  [Voronoi] _build_full_mask_polygon failed: {e}")
            return None

    def evaluate_layout_metrics(self, polygons: List[Polygon], assignment_idx: List[int]) -> Dict:
        """
        Evaluate layout quality using 3 metrics corresponding to the 3 main losses.
        
        Metrics:
        1. AD (Area Deviation): How well cell areas match target areas (corresponds to L_cap)
        2. ARD (Aspect Ratio Distortion): How well cell shapes match target aspects (corresponds to L_asp)
        3. AIE (Anchor IoU Error): How well cells match anchor rectangles (corresponds to L_ov)
        
        Args:
            polygons: List of final cell polygons (unbuffered, in normalized coords)
            assignment_idx: List mapping image_idx -> cell_idx
            
        Returns:
            Dict with aggregate and per-cell metrics
        """
        print(f"  [Metrics] Evaluating layout quality...")
        
        # Total mask area in normalized coords
        total_area_norm = (self.mask_binary > 127).sum() / (self.max_dim ** 2)
        
        # Storage for per-cell metrics
        per_cell_ad = []
        per_cell_ard = []
        per_cell_aie = []
        invalid_cells = []
        
        for img_idx in range(self.n):
            cell_idx = assignment_idx[img_idx]
            poly = polygons[cell_idx]
            
            # Check if cell is valid
            if poly is None or poly.is_empty or not hasattr(poly, 'exterior'):
                invalid_cells.append(img_idx)
                per_cell_ad.append(float('inf'))
                per_cell_ard.append(float('inf'))
                per_cell_aie.append(1.0)  # Max error
                continue
            
            # Get target values for this image
            target_cap = self.target_caps[img_idx].item()
            target_aspect = self.target_aspects[img_idx].item()
            target_area_norm = target_cap * total_area_norm
            
            # ===== 1. AD: Area Deviation =====
            cell_area_norm = poly.area
            ad_value = abs(cell_area_norm - target_area_norm)
            per_cell_ad.append(ad_value)
            
            # ===== 2. ARD: Aspect Ratio Distortion =====
            # Use minimum oriented bounding rectangle (OBB) to correctly handle
            # rotated cells (AABB overestimates aspect ratio for diagonal cells).
            try:
                obb = poly.minimum_rotated_rectangle
                obb_coords = np.array(obb.exterior.coords)
                edge0 = obb_coords[1] - obb_coords[0]
                edge1 = obb_coords[2] - obb_coords[1]
                side_a = float(np.linalg.norm(edge0))
                side_b = float(np.linalg.norm(edge1))
                cell_w = max(side_a, side_b)
                cell_h = min(side_a, side_b)

                if cell_h < 1e-6:
                    ard_value = 0.0
                else:
                    cell_aspect = cell_w / cell_h
                    # Log-space difference
                    ard_value = abs(np.log(cell_aspect + 1e-6) - np.log(target_aspect + 1e-6))

                per_cell_ard.append(ard_value)
            except Exception:
                per_cell_ard.append(float('inf'))
            
            # ===== 3. AIE: Anchor IoU Error =====
            # Build virtual anchor rectangle using same logic as L_ov
            try:
                cx, cy = poly.centroid.x, poly.centroid.y
                
                # Anchor rectangle dimensions (matching L_ov construction)
                half_h = 0.5 * math.sqrt(target_area_norm / (target_aspect + 1e-6))
                half_w = 0.5 * math.sqrt(target_area_norm * target_aspect)
                
                # Create anchor box centered at cell centroid
                anchor_box = box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
                
                # Compute IoU
                intersection = poly.intersection(anchor_box)
                union = poly.union(anchor_box)
                
                if union.area < 1e-9:
                    iou = 0.0
                else:
                    iou = intersection.area / union.area
                
                aie_value = 1.0 - iou
                per_cell_aie.append(aie_value)
            except Exception as e:
                per_cell_aie.append(1.0)
        
        # Filter out invalid cells for aggregate metrics
        valid_ad = [v for v in per_cell_ad if not math.isinf(v)]
        valid_ard = [v for v in per_cell_ard if not math.isinf(v)]
        valid_aie = [v for v in per_cell_aie if v < 1.0]
        
        # Compute aggregate metrics
        ad_mean = np.mean(valid_ad) if valid_ad else float('inf')
        nad_mean = ad_mean / total_area_norm if total_area_norm > 0 else float('inf')
        ard_mean = np.mean(valid_ard) if valid_ard else float('inf')
        aie_mean = np.mean(valid_aie) if valid_aie else 1.0
        
        metrics = {
            'AD': float(ad_mean),
            'nAD': float(nad_mean),
            'ARD': float(ard_mean),
            'AIE': float(aie_mean),
            'num_cells': self.n,
            'num_invalid': len(invalid_cells),
            'invalid_cells': invalid_cells,
            'per_cell': {
                'AD': [float(v) for v in per_cell_ad],
                'ARD': [float(v) for v in per_cell_ard],
                'AIE': [float(v) for v in per_cell_aie]
            }
        }
        
        print(f"  [Metrics] AD={ad_mean:.6f}, nAD={nad_mean:.6f}, ARD={ard_mean:.6f}, AIE={aie_mean:.6f}")
        print(f"  [Metrics] Valid cells: {len(valid_ad)}/{self.n}, Invalid: {len(invalid_cells)}")
        
        return metrics

    def _cell_shape_features(self, poly: Polygon):
        if poly is None or poly.is_empty or not hasattr(poly, 'exterior'):
            return None

        try:
            obb = poly.minimum_rotated_rectangle
            obb_coords = np.array(obb.exterior.coords)
            edge0 = obb_coords[1] - obb_coords[0]
            edge1 = obb_coords[2] - obb_coords[1]
            side_a = float(np.linalg.norm(edge0))
            side_b = float(np.linalg.norm(edge1))
            cell_w = max(side_a, side_b)
            cell_h = max(min(side_a, side_b), 1e-6)
            aspect = cell_w / cell_h
        except Exception:
            px_min, py_min, px_max, py_max = poly.bounds
            aspect = max(px_max - px_min, 1e-6) / max(py_max - py_min, 1e-6)

        return max(poly.area, 1e-9), max(aspect, 1e-6)

    def _bbox_iou_for_assignment(self, img_idx: int, poly: Polygon) -> float:
        if poly is None or poly.is_empty or not hasattr(poly, 'exterior'):
            return 0.0

        info = self.frame_infos[img_idx]
        img_w, img_h = info.get('frame_size', (640, 480))
        bbox = info.get('bbox', [0, 0, img_w, img_h])

        bw = (bbox[2] - bbox[0]) / img_w * self.norm_w
        bh = (bbox[3] - bbox[1]) / img_h * self.norm_h

        px_min, py_min, px_max, py_max = poly.bounds
        cell_w = max(px_max - px_min, 1e-6)
        cell_h = max(py_max - py_min, 1e-6)

        scale = max(cell_w / self.norm_w, cell_h / self.norm_h)
        scaled_bw = bw * scale
        scaled_bh = bh * scale

        cx, cy = poly.centroid.x, poly.centroid.y
        bbox_poly = box(
            cx - scaled_bw / 2, cy - scaled_bh / 2,
            cx + scaled_bw / 2, cy + scaled_bh / 2
        )

        try:
            intersection = poly.intersection(bbox_poly)
            union = poly.union(bbox_poly)
            if union.area <= 1e-9:
                return 0.0
            return float(intersection.area / union.area)
        except Exception:
            return 0.0

    def match_images_free_assignment(self, polygons: List[Polygon]) -> List[int]:
        """
        Assign images to cells with no timeline/reading-order constraint.

        The Hungarian solve lets any image use any cell, based on how well the
        cell matches that image's target area, aspect ratio, and bbox retention.
        
        Returns:
            assignment: List[int] where assignment[img_idx] = cell_idx
        """
        print("  [Assign] Matching images to cells freely (no timeline/order constraint)...")

        if not polygons:
            print("  [Assign] Warning: no cells available; fallback to identity assignment")
            return list(range(self.n))

        total_area_norm = (self.mask_binary > 127).sum() / (self.max_dim ** 2)
        cost = np.full((self.n, len(polygons)), 1e6, dtype=np.float64)

        for img_idx in range(self.n):
            target_area = max(self.target_caps[img_idx].item() * total_area_norm, 1e-9)
            target_aspect = max(self.target_aspects[img_idx].item(), 1e-6)

            for cell_idx, poly in enumerate(polygons):
                features = self._cell_shape_features(poly)
                if features is None:
                    continue

                cell_area, cell_aspect = features
                area_cost = abs(math.log(cell_area / target_area))
                aspect_cost = abs(math.log(cell_aspect / target_aspect))
                bbox_cost = 1.0 - self._bbox_iou_for_assignment(img_idx, poly)

                cost[img_idx, cell_idx] = (
                    1.0 * area_cost +
                    1.0 * aspect_cost +
                    0.75 * bbox_cost
                )

        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = list(range(self.n))
        for img_idx, cell_idx in zip(row_ind, col_ind):
            if img_idx < self.n:
                assignment[img_idx] = int(cell_idx)

        avg_cost = float(np.mean([cost[i, assignment[i]] for i in range(self.n)]))
        print(f"  [Assign] Free assignment complete: avg_cost={avg_cost:.4f}")
        print(f"  [Assign] First assignments image->cell: {assignment[:min(8, len(assignment))]}")
        
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
            use_timeline_order: Kept for API compatibility; Voronoi assignment is free-form
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
                print(f"  Image-cell assignment is FREE (no timeline/reading-order constraint)")

        # 1. Setup Canvas (will be overridden below if mask_path is provided)
        min_x, min_y, max_x, max_y = self.bounds
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        
        # 2. Load mask directly if provided (avoids polygon→mask losing shape holes)
        # Coordinates are unified in image space (top-left origin).
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
                coords[:, 1] -= min_y
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
        processed_polys, unbuffered_polys, invalid_cells = engine.generate_polygons(sites, weights)
        
        # Generate TRUE raw power diagram polygons (no post-processing) for comparison/collage
        raw_power_polys = engine.generate_raw_power_polygons(sites, weights)
        
        # Log invalid cells
        if invalid_cells:
            print(f"  [Voronoi] Layout has {len(invalid_cells)}/{len(processed_polys)} invalid cells")
        
        # Use processed polygons for spatial assignment (more stable)
        assignment_idx = engine.match_images_free_assignment(processed_polys)
        
        # Compute IoU scores using processed polygons
        final_iou = engine._compute_cell_bbox_iou(processed_polys, assignment_idx)
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

        def _save_cells_debug_image(file_name, cells, sites_arr=None, assignment_idx=None, fill_holes=True):
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
            if fill_holes:
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
        initial_raw_power_polys = None
        initial_raw_assignment_idx = None
        after_raw_power_polys = None
        after_raw_assignment_idx = None

        def _count_non_empty_cells(cells):
            if cells is None:
                return 0, []
            empty_ids = []
            for idx, cell in enumerate(cells):
                if (not hasattr(cell, 'is_empty')) or cell.is_empty:
                    empty_ids.append(idx)
            return len(cells) - len(empty_ids), empty_ids

        if engine.initial_sites_debug is not None:
            try:
                init_sites = torch.tensor(engine.initial_sites_debug, device=DEVICE, dtype=torch.float32)
                init_weights = torch.zeros(len(engine.initial_sites_debug), device=DEVICE, dtype=torch.float32)
                initial_polys, _, _ = engine.generate_polygons(init_sites, init_weights)
                # Keep labels in timeline space for consistency across all debug images.
                if initial_polys is not None:
                    initial_assignment_idx = engine.match_images_free_assignment(initial_polys)

                # Raw power-diagram debug (before optimization)
                initial_raw_power_polys = engine.generate_raw_power_polygons(init_sites, init_weights)
                if initial_raw_power_polys is not None:
                    initial_raw_assignment_idx = engine.match_images_free_assignment(initial_raw_power_polys)

                if verbose:
                    pre_proc_alive, pre_proc_empty = _count_non_empty_cells(initial_polys)
                    pre_raw_alive, pre_raw_empty = _count_non_empty_cells(initial_raw_power_polys)
                    print(
                        f"  [RawPower][before_opt] alive={pre_raw_alive}/{engine.n}, empty={pre_raw_empty}"
                    )
                    print(
                        f"  [Processed][before_opt] alive={pre_proc_alive}/{engine.n}, empty={pre_proc_empty}"
                    )
            except Exception as e:
                if verbose:
                    print(f"  [DEBUG] Could not generate pre-opt Voronoi cells: {e}")

        # Keep initial cell centroids in the output for backward compatibility.
        # Order-based evaluation ignores them for free-assignment runs.
        initial_site_centroids_norm = []
        if initial_polys is not None and initial_assignment_idx is not None:
            for img_idx in range(engine.n):
                cell_idx = initial_assignment_idx[img_idx]
                poly = initial_polys[cell_idx] if cell_idx < len(initial_polys) else None
                if poly is not None and hasattr(poly, 'centroid') and not poly.is_empty:
                    cx_01 = float(poly.centroid.x / max(engine.norm_w, 1e-9))
                    cy_01 = float(poly.centroid.y / max(engine.norm_h, 1e-9))
                    initial_site_centroids_norm.append([cx_01, cy_01])
                else:
                    initial_site_centroids_norm.append([0.5, 0.5])
            if verbose:
                print(f"  [Debug] Saved {len(initial_site_centroids_norm)} initial cell centroids")

        # Raw power-diagram debug (after optimization)
        try:
            after_raw_power_polys = engine.generate_raw_power_polygons(sites, weights)
            if after_raw_power_polys is not None:
                after_raw_assignment_idx = engine.match_images_free_assignment(after_raw_power_polys)

            if verbose:
                post_raw_alive, post_raw_empty = _count_non_empty_cells(after_raw_power_polys)
                post_proc_alive, post_proc_empty = _count_non_empty_cells(processed_polys)
                print(
                    f"  [RawPower][after_opt] alive={post_raw_alive}/{engine.n}, empty={post_raw_empty}"
                )
                print(
                    f"  [Processed][after_opt] alive={post_proc_alive}/{engine.n}, empty={post_proc_empty}"
                )
        except Exception as e:
            if verbose:
                print(f"  [DEBUG] Could not generate post-opt raw power cells: {e}")

        if debug_dir:
            _save_cells_debug_image(
                'voronoi_debug_2_cells_before_opt.png',
                initial_raw_power_polys,
                sites_arr=engine.initial_sites_debug,
                assignment_idx=initial_raw_assignment_idx,
                fill_holes=False,
            )
            _save_cells_debug_image(
                'voronoi_debug_3_cells_after_opt.png',
                after_raw_power_polys,
                sites_arr=engine.optimized_sites_debug,
                assignment_idx=after_raw_assignment_idx,
                fill_holes=False,
            )
            # Processed/stabilized variants (core guarantee + post-processing pipeline).
            _save_cells_debug_image(
                'voronoi_debug_2_cells_before_opt_processed.png',
                initial_polys,
                sites_arr=engine.initial_sites_debug,
                assignment_idx=initial_assignment_idx,
            )
            _save_cells_debug_image(
                'voronoi_debug_3_cells_after_opt_processed.png',
                processed_polys,
                sites_arr=engine.optimized_sites_debug,
                assignment_idx=assignment_idx,
            )
            # Keep legacy filename for compatibility with existing workflows.
            _save_cells_debug_image(
                'voronoi_debug_3_cells.png',
                processed_polys,
                sites_arr=engine.optimized_sites_debug,
                assignment_idx=assignment_idx,
            )
            _save_cells_debug_image(
                'voronoi_debug_2_cells_before_opt_raw_power.png',
                initial_raw_power_polys,
                sites_arr=engine.initial_sites_debug,
                assignment_idx=initial_raw_assignment_idx,
                fill_holes=False,
            )
            _save_cells_debug_image(
                'voronoi_debug_3_cells_after_opt_raw_power.png',
                after_raw_power_polys,
                sites_arr=engine.optimized_sites_debug,
                assignment_idx=after_raw_assignment_idx,
                fill_holes=False,
            )
        
        # Final IoU report
        final_iou = engine._compute_cell_bbox_iou(processed_polys, assignment_idx)
        if debug_dir:
            iou_report = {
                "avg": float(avg_iou),
                "per_image": [{"img": int(i), "iou": float(iou), "cell": int(assignment_idx[i])} 
                             for i, iou in enumerate(final_iou)]
            }
            import json
            with open(os.path.join(debug_dir, 'bbox_iou_report.json'), 'w') as f:
                json.dump(iou_report, f, indent=2)
        
        # ===== EVALUATE LAYOUT METRICS (AD, ARD, AIE) =====
        # Use unbuffered polygons for evaluation to avoid overlap artifacts
        layout_metrics = engine.evaluate_layout_metrics(unbuffered_polys, assignment_idx)
        
        # Save metrics to JSON and CSV
        if debug_dir:
            # Get ablation mode for filename
            ablation_mode = CONFIG.get('ablation_mode', 'full')
            if ablation_mode is None:
                ablation_mode = 'full'
            
            # JSON output (detailed)
            metrics_json = {
                'ablation_mode': ablation_mode,
                'num_cells': layout_metrics['num_cells'],
                'num_invalid': layout_metrics['num_invalid'],
                'invalid_cells': layout_metrics['invalid_cells'],
                'aggregate_metrics': {
                    'AD': layout_metrics['AD'],
                    'nAD': layout_metrics['nAD'],
                    'ARD': layout_metrics['ARD'],
                    'AIE': layout_metrics['AIE']
                },
                'per_cell_metrics': layout_metrics['per_cell']
            }
            
            with open(os.path.join(debug_dir, 'layout_metrics.json'), 'w') as f:
                json.dump(metrics_json, f, indent=2)
            
            # CSV output (aggregate)
            import csv
            csv_path = os.path.join(debug_dir, 'layout_metrics.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ablation_mode', 'num_cells', 'num_invalid', 'AD', 'nAD', 'ARD', 'AIE'])
                writer.writerow([
                    ablation_mode,
                    layout_metrics['num_cells'],
                    layout_metrics['num_invalid'],
                    f"{layout_metrics['AD']:.6f}",
                    f"{layout_metrics['nAD']:.6f}",
                    f"{layout_metrics['ARD']:.6f}",
                    f"{layout_metrics['AIE']:.6f}"
                ])
            
            # CSV output (per-cell)
            csv_per_cell_path = os.path.join(debug_dir, 'layout_metrics_per_cell.csv')
            with open(csv_per_cell_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['cell_idx', 'img_idx', 'AD', 'ARD', 'AIE'])
                for img_idx in range(layout_metrics['num_cells']):
                    cell_idx = assignment_idx[img_idx]
                    writer.writerow([
                        cell_idx,
                        img_idx,
                        f"{layout_metrics['per_cell']['AD'][img_idx]:.6f}",
                        f"{layout_metrics['per_cell']['ARD'][img_idx]:.6f}",
                        f"{layout_metrics['per_cell']['AIE'][img_idx]:.6f}"
                    ])
            
            if verbose:
                print(f"  [Metrics] Saved to:")
                print(f"    - layout_metrics.json")
                print(f"    - layout_metrics.csv")
                print(f"    - layout_metrics_per_cell.csv")
        
        # EXTENDED DEBUG: Visualizations
        if debug_dir:
            pass  # Additional debug viz can be added here
        
        # DEBUG: Stage 4 - Assignment visualization
        if debug_dir:
            debug_assign = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            scale = engine.max_dim
            for img_id, cell_id in enumerate(assignment_idx):
                cell = processed_polys[cell_id]
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
            for i, cell in enumerate(processed_polys):
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
                cell = processed_polys[cell_id]
                
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
        final_cells_unbuffered = []
        final_assignments = {}
        
        scale_val = engine.max_dim
        
        # Re-order cells based on assignment: Cell 0 is for Image 0
        # Keep raw power-diagram output for collage geometry.
        if verbose:
            print("  [Output] Final collage geometry source: RAW power Voronoi polygons")
        for img_id, cell_id in enumerate(assignment_idx):
            poly = raw_power_polys[cell_id] if cell_id < len(raw_power_polys) else Polygon()
            ubpoly = unbuffered_polys[cell_id] if cell_id < len(unbuffered_polys) else Polygon()
            
            if poly is None or poly.is_empty or not hasattr(poly, 'exterior'):
                final_cells.append(Polygon())
                final_cells_unbuffered.append(Polygon())
                final_assignments[img_id] = img_id
                continue
            
            # Scale & Translate back to original coords (buffered)
            poly = shp_scale(poly, xfact=scale_val, yfact=scale_val, origin=(0,0))
            coords = list(poly.exterior.coords)
            new_coords = [(x + min_x, y + min_y) for x, y in coords]
            final_cells.append(Polygon(new_coords))
            
            # Scale & Translate unbuffered polygon too
            if ubpoly is not None and not ubpoly.is_empty and hasattr(ubpoly, 'exterior'):
                ubpoly = shp_scale(ubpoly, xfact=scale_val, yfact=scale_val, origin=(0,0))
                ub_coords = list(ubpoly.exterior.coords)
                ub_new_coords = [(x + min_x, y + min_y) for x, y in ub_coords]
                final_cells_unbuffered.append(Polygon(ub_new_coords))
            else:
                final_cells_unbuffered.append(Polygon())
            
            final_assignments[img_id] = img_id # Direct map because we sorted cells
        
        return {
            'success': True,
            'cells': final_cells,
            'cells_unbuffered': final_cells_unbuffered,
            'assignments': final_assignments,
            'dims': (width, height),
            'initial_site_centroids': initial_site_centroids_norm,
        }

def convert_voronoi_to_slicing_format(layout_result):
    cells = layout_result.get('cells', [])
    cells_unbuffered = layout_result.get('cells_unbuffered', [])
    assignments = layout_result.get('assignments', {})
    
    parts_dict = {}
    unbuffered_dict = {}
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
        
        # Unbuffered coords for evaluation
        if i < len(cells_unbuffered):
            ubcell = cells_unbuffered[i]
            if hasattr(ubcell, 'exterior') and not ubcell.is_empty:
                ub_coords = list(ubcell.exterior.coords)
                if len(ub_coords) >= 3:
                    unbuffered_dict[i] = [[float(c[0]), float(c[1])] for c in ub_coords]
                else:
                    unbuffered_dict[i] = []
            else:
                unbuffered_dict[i] = []
        else:
            unbuffered_dict[i] = []
            
        mapping[i] = i
        
    return parts_dict, unbuffered_dict, mapping, []

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
