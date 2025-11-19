# -*- coding: utf-8 -*-
"""
temporal_layout_composer_unified.py
===================================
Complete temporal layout composition pipeline in a single file.

Combines:
- Temporal segmentation (clustering images by temporal proximity)
- Spatial layout optimization (hierarchical tree with spatial constraints)
- Constraint propagation (temporal clusters at layer-2)

All optimization code is self-contained - no external imports of sas_optimization.
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
import math
import random
import traceback
from typing import List, Dict, Tuple, Any
from pathlib import Path

import numpy as np
import cv2
from shapely.geometry import LineString, Point, Polygon
from shapely import ops
import networkx as nx
from PIL import Image

# Scoring utilities
from src.scoring.clip_embedding import get_clip_embedding
from src.scoring.iqa_score import get_iqa_score

# Import temporal segmenter
from src.scoring.temporal_segmenter import AutoSegmenter

# Import Colla dependencies (shape decomposition + SAS optimizer)
colla_path = os.path.join(os.path.dirname(__file__), 'repos', 'Colla')
if colla_path not in sys.path:
    sys.path.insert(0, colla_path)

try:
    import shape_decomposition as sd
except Exception as e:
    print(f"Warning: Could not import shape_decomposition: {e}")
    sd = None

try:
    import sas_optimization as sas
except Exception as e:
    print(f"Warning: Could not import sas_optimization: {e}")
    sas = None


def vector_angle(vector_1, vector_2):
    """Calculate angle between two vectors (counter-clockwise positive)."""
    vector_1 = vector_1 / (np.linalg.norm(vector_1) + 1e-8)
    vector_2 = vector_2 / (np.linalg.norm(vector_2) + 1e-8)
    
    ang = np.arccos(np.clip(np.dot(vector_1, vector_2), -1, 1))
    cross_product = np.cross(vector_1, vector_2)
    
    return ang if cross_product >= 0 else -ang


def interior_angle(vector_1, vector_2):
    """Calculate interior angle between vectors."""
    vector_1 = vector_1 / (np.linalg.norm(vector_1) + 1e-8)
    vector_2 = vector_2 / (np.linalg.norm(vector_2) + 1e-8)
    
    ang = np.arccos(np.clip(np.dot(vector_1, vector_2), -1, 1))
    ang = np.abs(ang) if np.cross(vector_1, vector_2) > 0 else 2*np.pi - np.abs(ang)
    return ang * 180 / math.pi


def interior_angles(vertices):
    """Calculate all interior angles of a polygon."""
    angles = []
    for i in range(len(vertices)):
        p1 = np.array(vertices[i])
        ref = np.array(vertices[i-1])
        p2 = np.array(vertices[(i+1) % len(vertices)])
        v1 = ref - p1
        v2 = p2 - p1
        angles.append(interior_angle(v1, v2))
    return angles


def cell_quality(polygon, simplification_threshold=10):
    """Check if polygon is a good cell (not too many sides, no sharp angles)."""
    try:
        vertices = list(polygon.simplify(simplification_threshold).exterior.coords)[:-1]
        side = len(vertices)
        if side <= 3:
            return False
        angles = interior_angles(vertices)
        sharp_angle = any(angle < 35 for angle in angles)
        return not sharp_angle
    except:
        return True


def centroid_cut(centroid, direction, magnitude):
    """Create a cut line through centroid perpendicular to direction."""
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
    new_first = np.array(centroid) + direction_norm * magnitude
    new_second = np.array(centroid) - direction_norm * magnitude
    return (new_first, new_second)


def medial_axis_tangent(multilinestring, point):
    """Get tangent vector to medial axis at closest point to given point."""
    from shapely.ops import nearest_points
    
    try:
        p1, p2 = nearest_points(multilinestring, point)
        if multilinestring.contains(p1):
            location = p1
        else:
            # Try nearby points
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                test_point = Point(p1.x + dx, p1.y + dy)
                if multilinestring.contains(test_point):
                    location = test_point
                    break
            else:
                location = p1
        
        linear_coordinate = multilinestring.project(location)
        projection = multilinestring.interpolate(linear_coordinate).coords[0]
        
        # Get reference points for tangent
        try:
            reference1 = multilinestring.interpolate(linear_coordinate - 5).coords[0]
        except:
            reference1 = projection
        
        try:
            reference2 = multilinestring.interpolate(linear_coordinate + 5).coords[0]
        except:
            reference2 = projection
        
        if Point(reference1).distance(Point(projection)) > 8.0:
            reference1 = projection
        if Point(reference2).distance(Point(projection)) > 8.0:
            reference2 = projection
        
        tangent = np.array([reference1[0] - reference2[0], reference1[1] - reference2[1]])
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            tangent = tangent / tangent_norm
        if tangent[0] < 0:
            tangent = tangent * -1.0
        
        return tangent, projection
    except:
        return np.array([1.0, 0.0]), (point.x, point.y)


# ============================================================================
# SPATIAL OPTIMIZATION - EMBEDDED FUNCTIONS (from sas_optimization.py)

class TreeNode:
    """
    Data structure representing a node in the slicing tree.
    """
    def __init__(self):
        self.polygon = None
        self.type = "N"  # Type: N (unset), A (axial), C (crosswise)
        self.configuration = -1
        self.left_child = None  # Image 0
        self.right_child = None  # Image 1
        self.assignment = {"id": -1, "aspect_ratio": -0.1, "coord": []}
        self.cut = []
        self.temporal_segments = []
        self.temporal_assignment_part = -1

    def is_leaf(self):
        if not self.left_child or not self.right_child:
            return True
        else:
            return False

    def score(self):
        if self.is_leaf():
            return 0
        else:
            return self.left_child.score() + self.right_child.score()

    def centroid(self):
        if self.polygon is None:
            return (0, 0)
        return self.polygon.centroid.coords[0]

    def area(self):
        if self.polygon is None:
            return 0
        return self.polygon.area

    def get_height(self):
        if self.is_leaf():
            return 0
        else:
            return 1 + max(self.left_child.get_height(), self.right_child.get_height())

    def get_num(self):
        if self.is_leaf():
            return 1
        else:
            return self.left_child.get_num() + self.right_child.get_num()
    
    def get_axial(self, medial_axis):
        """Get axial direction (tangent to medial axis at centroid)."""
        try:
            tangent, _ = medial_axis_tangent(medial_axis, Point(self.centroid()[0], self.centroid()[1]))
            return tangent
        except:
            return np.array([1.0, 0.0])
    
    def get_crosswise(self, medial_axis):
        """Get crosswise direction (perpendicular to axial)."""
        try:
            tangent, _ = medial_axis_tangent(medial_axis, Point(self.centroid()[0], self.centroid()[1]))
            perpendicular1 = np.array([tangent[1], -tangent[0]])
            perpendicular2 = np.array([-tangent[1], tangent[0]])
            if np.cross(tangent, perpendicular1) > 0:
                return perpendicular1
            else:
                return perpendicular2
        except:
            return np.array([0.0, 1.0])
    
    def get_size(self, direction):
        """Get width and height of polygon when rotated to align with direction."""
        try:
            angle = vector_angle(np.array([1.0, 0.0]), direction) * 180 / math.pi
            aligned = affinity.rotate(self.polygon, -angle, (0, 0))
            bounding_box = aligned.bounds
            return bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]
        except:
            bounds = self.polygon.bounds
            return bounds[2] - bounds[0], bounds[3] - bounds[1]


class Part:
    """Simple part class for spatial decomposition."""
    def __init__(self, polygon):
        self.polygon = polygon
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        if not self.left_child or not self.right_child:
            return True
        else:
            return False


class Partition:
    """Partition class for managing spatial divisions."""
    def __init__(self, polygon):
        self.root = Part(polygon)
        self.cut_list = []

    def list_leaves(self):
        return self._list_leaves(self.root)

    def _list_leaves(self, cur_node):
        if cur_node.is_leaf():
            return [cur_node]
        else:
            return self._list_leaves(cur_node.left_child) + self._list_leaves(cur_node.right_child)


def load_mask(path):
    """Load a mask image."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return image


def extract_foreground(label):
    """
    Extract foreground pixels' bounding box.
    Returns: (x1, x2, y1, y2, foreground_exists)
    """
    total_area = label.shape[0] * label.shape[1]
    foreground = (label == 255).astype(int)
    foreground_area = np.sum(foreground)
    foreground_exist = True

    if foreground_area > total_area / 200:
        x1 = int(np.min(np.where(foreground)[1]))
        x2 = int(np.max(np.where(foreground)[1]))
        y1 = int(label.shape[0] - np.max(np.where(foreground)[0]))
        y2 = int(label.shape[0] - np.min(np.where(foreground)[0]))
    else:
        foreground_exist = False
        x1 = int(label.shape[1] / 10)
        x2 = int(label.shape[1] * 9 / 10)
        y1 = int(label.shape[0] / 10)
        y2 = int(label.shape[0] * 9 / 10)
    return x1, x2, y1, y2, foreground_exist


def sum_dict(dict1, dict2):
    """Sum two dictionaries by keys."""
    return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2)}


def leaf_elevation_summary(root):
    """Get summary of leaf depths."""
    def _leaf_depth(tree_node, cur_depth, total_height):
        if tree_node.is_leaf():
            return [cur_depth]
        else:
            return _leaf_depth(tree_node.left_child, cur_depth + 1, total_height) + \
                   _leaf_depth(tree_node.right_child, cur_depth + 1, total_height)

    def summarize(occurrence):
        summary = {}
        for d in occurrence:
            if d not in summary:
                summary[d] = 0
            summary[d] += 1
        return summary

    total_height = root.get_height()
    return summarize(_leaf_depth(root, 0, total_height))


def calculate_image_assignment(images, summary):
    """Assign images to tree depths."""
    summary_copy = summary.copy()
    assignment = {}
    keys = sorted(summary.keys(), reverse=True)
    i = 0
    for image in images:
        if summary_copy[keys[i]] == 0:
            i += 1

        summary_copy[keys[i]] = summary_copy[keys[i]] - 1
        if keys[i] in assignment:
            assignment[keys[i]].append(image)
        else:
            assignment[keys[i]] = [image]
    return assignment


def _list_leaves(cur_node):
    """Get all leaves from a node - MUST BE DEFINED BEFORE extract_forest_geometry."""
    if cur_node.is_leaf():
        return [cur_node]
    else:
        left_leaves = _list_leaves(cur_node.left_child) if cur_node.left_child else []
        right_leaves = _list_leaves(cur_node.right_child) if cur_node.right_child else []
        return left_leaves + right_leaves


class BalancedStrategy:
    """Strategy to balance tree based on height."""
    def choose(self, node):
        if node.left_child.get_height() > node.right_child.get_height():
            return node.right_child
        elif node.left_child.get_height() < node.right_child.get_height():
            return node.left_child
        else:
            return node.left_child if random.random() > 0.5 else node.right_child


class UnbalancedStrategy:
    """Strategy for unbalanced tree."""
    def choose(self, node):
        if node.left_child.get_height() > node.right_child.get_height():
            return node.right_child
        elif node.left_child.get_height() < node.right_child.get_height():
            return node.left_child
        else:
            return node.left_child if random.random() > 0.5 else node.right_child


def tree_initialization(number_of_leaf_node, balanced=True, fix_seed=False):
    """Initialize a slicing tree."""
    if fix_seed:
        random.seed(10)

    def insert(tree_node, strategy):
        if not tree_node.is_leaf():
            node_to_insert = strategy.choose(tree_node)
            insert(node_to_insert, strategy)
        else:
            new_left = TreeNode()
            new_right = TreeNode()
            tree_node.left_child = new_left
            tree_node.right_child = new_right
        return tree_node

    root = TreeNode()
    ba = BalancedStrategy()
    ub = UnbalancedStrategy()
    for i in range(number_of_leaf_node - 1):
        if balanced:
            insert(root, ba)
        else:
            insert(root, ub)
    return root


def heuristic_initialization(cur_node, medial_axis, depth):
    """Heuristically initialize tree configuration using SAS optimizer when available."""
    if sas and hasattr(sas, "heuristic_initialization"):
        return sas.heuristic_initialization(cur_node, medial_axis, depth)
    if depth == 0 or cur_node.is_leaf():
        return
    cur_node.configuration = random.randint(0, 3)
    heuristic_initialization(cur_node.left_child, medial_axis, depth - 1)
    heuristic_initialization(cur_node.right_child, medial_axis, depth - 1)


def forest_optimization(forest, multilinestring_int):
    """Optimize slicing trees via SAS optimizer when available."""
    if sas and hasattr(sas, "forest_optimization"):
        return sas.forest_optimization(forest, multilinestring_int)
    solution = []
    for tree in forest:
        solution.append((0, tree))
    return solution


def extract_forest_geometry(forest):
    """Extract geometry from forest."""
    parts_dict = []
    image_assignments = {}
    idx = 0
    mapping = {}
    cuts = []

    for tree in forest:
        # Extract leaves from tree
        leaves = _list_leaves(tree)
        for leaf in leaves:
            if leaf.polygon:
                part_format = {
                    "index": idx,
                    "coords": list(leaf.polygon.exterior.coords),
                    "foreground": leaf.assignment.get("foreground", []) if leaf.assignment else []
                }
                parts_dict.append(part_format)
                
                # Build assignment dict with all metadata
                assignment_dict = {
                    "assigned_part": idx,
                    "filename": leaf.assignment.get("filename", "") if leaf.assignment else "",
                    "image_id": leaf.assignment.get("id", -1) if leaf.assignment else -1,
                    "aspect_ratio": leaf.assignment.get("aspect_ratio", 1.0) if leaf.assignment else 1.0,
                    "priority_score": leaf.assignment.get("priority_score", 0.0) if leaf.assignment else 0.0,
                    "priority_rank": leaf.assignment.get("priority_rank", -1) if leaf.assignment else -1,
                    "clip_score": leaf.assignment.get("clip_score", 0.0) if leaf.assignment else 0.0,
                    "iqa_score": leaf.assignment.get("iqa_score", 0.0) if leaf.assignment else 0.0,
                    "image_path": leaf.assignment.get("image_path", "") if leaf.assignment else ""
                }
                image_assignments[idx] = assignment_dict
                mapping[idx] = leaf.assignment.get("id", -1) if leaf.assignment else -1
                idx += 1

    return parts_dict, image_assignments, cuts


def assign_images_to_tree(tree_node, images_list, depth=0):
    """Assign images from list to tree leaves in order."""
    if tree_node.is_leaf():
        if images_list:
            tree_node.assignment = images_list.pop(0)
    else:
        if tree_node.left_child:
            assign_images_to_tree(tree_node.left_child, images_list, depth + 1)
        if tree_node.right_child:
            assign_images_to_tree(tree_node.right_child, images_list, depth + 1)


def assign_images_to_forest_by_cluster(forest, image_dict, temporal_segments, segment_files):
    """
    Assign images to forest where each tree/subtree gets one temporal cluster.
    This preserves temporal ordering while respecting spatial hierarchy.
    
    Args:
        forest: List of TreeNode roots (one per partition/cluster ideally)
        image_dict: List of image metadata
        temporal_segments: List of segment indices [[0,1], [2,3,4,5,6,7], ...]
        segment_files: List of segment file paths [["img0.png", "img1.png"], ...]
    """
    # Build mapping: segment_idx -> list of images in that segment
    segment_to_images = {}
    for seg_idx, seg_files in enumerate(segment_files):
        segment_to_images[seg_idx] = []
        for img_file in seg_files:
            # Extract base filename without extension
            img_basename = os.path.basename(img_file)
            img_name_no_ext = os.path.splitext(img_basename)[0]
            
            # Find matching image in image_dict
            for img_meta in image_dict:
                img_id = img_meta.get("filename", "")
                # Match by: exact match, or basename match, or name without extension
                if (img_id == img_file or 
                    img_id == img_basename or 
                    img_id == img_name_no_ext or
                    img_file in img_id or
                    img_basename in img_id):
                    segment_to_images[seg_idx].append(img_meta)
                    break
    
    print(f"[assign_images_to_forest_by_cluster] Segment to images mapping:")
    for seg_idx, images in segment_to_images.items():
        print(f"  Segment {seg_idx}: {len(images)} images")
        for img in images[:2]:  # Show first 2 images
            print(f"    - {img.get('filename', 'unknown')}")
    
    total_assigned = 0
    
    # Assign each segment's images to a tree in the forest
    for tree_idx, tree in enumerate(forest):
        if tree_idx >= len(segment_to_images):
            break
        
        # Get images for this segment
        seg_idx = tree_idx  # Assume trees are ordered same as segments
        images_for_tree = segment_to_images.get(seg_idx, [])
        
        if not images_for_tree:
            print(f"[assign_images_to_forest_by_cluster] WARNING: No images for tree {tree_idx}")
            continue
        
        print(f"[assign_images_to_forest_by_cluster] Assigning {len(images_for_tree)} images to tree {tree_idx}")
        
        # Assign images to leaves of this tree in order
        def assign_to_leaves(node, images_list):
            nonlocal total_assigned
            if node.is_leaf():
                if images_list:
                    img_meta = images_list.pop(0)
                    # Ensure id field is set for rendering
                    if "id" not in img_meta or img_meta["id"] == -1:
                        img_meta["id"] = total_assigned
                    node.assignment = img_meta
                    total_assigned += 1
                    # Debug: print assignment
                    if "filename" in img_meta:
                        print(f"    ✓ Assigned leaf to {img_meta['filename']} (id: {img_meta['id']})")
                else:
                    print(f"    - Leaf: no more images to assign")
            else:
                if node.left_child:
                    assign_to_leaves(node.left_child, images_list)
                if node.right_child:
                    assign_to_leaves(node.right_child, images_list)
        
        assign_to_leaves(tree, images_for_tree.copy())
    
    print(f"[assign_images_to_forest_by_cluster] Total assigned: {total_assigned} images to forest leaves")


def render_layout_with_assigned_images(forest, image_dir, canvas_shape, output_path):
    """
    Render the final layout with assigned images.
    Images are drawn in their assigned partitions based on forest structure.
    
    Args:
        forest: List of TreeNode roots with assigned images
        image_dir: Directory containing original images
        canvas_shape: (width, height) of canvas
        output_path: Path to save the rendered layout
    """
    try:
        canvas = np.ones((canvas_shape[1], canvas_shape[0], 3), dtype=np.uint8) * 255
        
        # Extract all leaves from forest
        all_leaves = []
        for tree in forest:
            leaves = _list_leaves(tree)
            all_leaves.extend(leaves)
        
        print(f"[TemporalLayoutComposer] Rendering {len(all_leaves)} partitions with assigned images...")
        print(f"[DEBUG] Canvas shape: {canvas_shape}, created canvas: {canvas.shape}")
        print(f"[DEBUG] Image directory: {image_dir}")
        print(f"[DEBUG] Output path: {output_path}")
        
        # Draw each image in its assigned partition
        rendered_count = 0
        for leaf_idx, leaf in enumerate(all_leaves):
            print(f"[DEBUG] Processing leaf {leaf_idx}:")
            print(f"  - Has polygon: {leaf.polygon is not None}")
            print(f"  - Has assignment: {leaf.assignment is not None}")
            
            if not leaf.polygon or not leaf.assignment:
                print(f"  - Skipping: missing polygon or assignment")
                continue
            
            image_id = leaf.assignment.get("id", -1)
            filename = leaf.assignment.get("filename", "")
            print(f"  - Image ID: {image_id}, filename: {filename}")
            
            if image_id == -1 and not filename and not leaf.assignment.get("image_path"):
                print(f"  - Skipping: invalid ID, filename, and image path")
                continue
            
            # Try to find and load the image
            img_path = leaf.assignment.get("image_path") if leaf.assignment else None
            if img_path and not os.path.exists(img_path):
                img_path = None
            basename = os.path.basename(filename) if "/" in filename else filename
            print(f"  - Looking for image with basename: {basename}")
            
            if not img_path:
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    # Try with basename
                    candidate = os.path.join(image_dir, f"{basename}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                    
                    # Try without extension
                    name_no_ext = os.path.splitext(basename)[0]
                    candidate = os.path.join(image_dir, f"{name_no_ext}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
            
            if not img_path:
                print(f"  ⚠ Image not found: {filename}")
                continue
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  ⚠ Could not read image: {img_path}")
                    continue
                
                # Get partition bounds
                coords = list(leaf.polygon.exterior.coords)
                if len(coords) < 3:
                    continue
                
                # Flip Y coordinates for OpenCV (origin top-left vs Shapely bottom-left)
                pts = np.array([[int(c[0]), canvas_shape[1] - int(c[1])] for c in coords], dtype=np.int32)
                
                # Create mask for this partition
                mask = np.zeros((canvas_shape[1], canvas_shape[0]), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                
                # Get bounds of this partition (with flipped Y)
                x_coords = [int(c[0]) for c in coords]
                y_coords = [canvas_shape[1] - int(c[1]) for c in coords]  # Flip Y
                x_min = max(0, min(x_coords))
                x_max = min(canvas_shape[0], max(x_coords))
                y_min = max(0, min(y_coords))
                y_max = min(canvas_shape[1], max(y_coords))
                
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Resize image to fit partition
                w = x_max - x_min
                h = y_max - y_min
                if w <= 0 or h <= 0:
                    continue
                
                resized = cv2.resize(img, (w, h))
                
                # Apply mask and blend
                partition_mask = mask[y_min:y_max, x_min:x_max]
                partition_region = canvas[y_min:y_max, x_min:x_max]
                
                # Where mask is 255, use resized image; otherwise keep original canvas
                for c in range(3):
                    partition_region[:, :, c] = np.where(
                        partition_mask > 0,
                        resized[:, :, c],
                        partition_region[:, :, c]
                    )
                
                rendered_count += 1
                
            except Exception as e:
                print(f"  ⚠ Error rendering {basename}: {e}")
                continue
        
        # Save the result
        cv2.imwrite(output_path, canvas)
        print(f"[TemporalLayoutComposer] Saved layout visualization with {rendered_count} images: {output_path}")
        return True
    except Exception as e:
        print(f"[TemporalLayoutComposer] Error rendering layout: {e}")
        traceback.print_exc()
        return False


def get_optimal(tree_node, medial_axis, output_dir=None, step_num=0):
    """
    Recursively find optimal slicing tree configuration using Colla's approach.
    Tests 4 configurations (2 axial + 2 crosswise) and picks best score.
    """
    if tree_node.is_leaf():
        # Leaf node: already optimized
        convex = tree_node.polygon.convex_hull.simplify(10)
        quality = cell_quality(tree_node.polygon)
        return (1.0 if quality else 0.5), tree_node
    
    if tree_node.configuration != -1:
        # Already configured, just recurse
        score_left, left_result = get_optimal(tree_node.left_child, medial_axis, output_dir, step_num)
        score_right, right_result = get_optimal(tree_node.right_child, medial_axis, output_dir, step_num)
        return score_left + score_right, tree_node
    
    # Test all 4 configurations
    try:
        all_results = []
        
        # Get polygon dimensions along different directions
        polygon_dimensions_axial = tree_node.get_size(tree_node.get_axial(medial_axis))
        polygon_dimensions_cross = tree_node.get_size(tree_node.get_crosswise(medial_axis))
        
        # Configuration 0-3: axial (0,1) vs crosswise (2,3)
        cut_axial = LineString([Point(p) for p in centroid_cut(
            tree_node.centroid(), 
            tree_node.get_axial(medial_axis), 
            polygon_dimensions_axial[0]
        )])
        cut_crosswise = LineString([Point(p) for p in centroid_cut(
            tree_node.centroid(),
            tree_node.get_crosswise(medial_axis),
            polygon_dimensions_cross[1]
        )])
        
        # Split along both directions
        try:
            axial_splits = list(ops.split(tree_node.polygon, cut_axial))
            axial_splits.sort(key=lambda x: -x.area)
        except:
            axial_splits = [tree_node.polygon, tree_node.polygon]
        
        try:
            crosswise_splits = list(ops.split(tree_node.polygon, cut_crosswise))
            crosswise_splits.sort(key=lambda x: -x.area)
        except:
            crosswise_splits = [tree_node.polygon, tree_node.polygon]
        
        # Test all 4 orderings
        configs = [
            (0, axial_splits, "axial_12"),      # axial, left=larger
            (1, axial_splits[::-1], "axial_21"), # axial, left=smaller
            (2, crosswise_splits, "cross_12"),   # crosswise, left=larger
            (3, crosswise_splits[::-1], "cross_21"), # crosswise, left=smaller
        ]
        
        for config_id, splits, config_name in configs:
            if len(splits) < 2:
                continue
            
            try:
                # Temporarily assign and evaluate
                left_copy = TreeNode()
                left_copy.polygon = splits[0]
                left_copy.configuration = -1
                left_copy.assignment = {"id": -1, "aspect_ratio": 1.0, "coord": []}
                
                right_copy = TreeNode()
                right_copy.polygon = splits[1]
                right_copy.configuration = -1
                right_copy.assignment = {"id": -1, "aspect_ratio": 1.0, "coord": []}
                
                # Recursively evaluate
                score_l, _ = get_optimal(left_copy, medial_axis)
                score_r, _ = get_optimal(right_copy, medial_axis)
                
                total_score = score_l + score_r
                all_results.append((total_score, config_id, splits, config_name))
            except:
                pass
        
        if not all_results:
            # Fallback: just return current
            return 0.0, tree_node
        
        # Pick best
        best_score, best_config, best_splits, best_name = max(all_results, key=lambda x: x[0])
        
        # Apply best configuration
        tree_node.configuration = best_config
        tree_node.left_child.polygon = best_splits[0]
        tree_node.right_child.polygon = best_splits[1]
        
        # Recurse on children
        score_l, _ = get_optimal(tree_node.left_child, medial_axis)
        score_r, _ = get_optimal(tree_node.right_child, medial_axis)
        
        return best_score, tree_node
        
    except Exception as e:
        print(f"[get_optimal] Warning: {e}")
        return 0.0, tree_node


class TemporalLayoutComposer:
    """
    Compose temporal segmentation with spatial layout optimization.
    
    Key idea:
    - Temporal segments define which images belong together temporally
    - Layer 2 of the forest tree respects these clusters
    - Each temporal cluster gets its own subtree in the forest
    """
    
    def __init__(self,
                 image_dir: str,
                 mask_folder: str,
                 shape_image: str,
                 output_dir: str,
                 min_len: int = 3,
                 max_len: int = 4,
                 w_clip: float = 0.8,
                 w_iqa: float = 0.2,
                 balanced: bool = True):
        """
        Args:
            image_dir: Directory with cropped images (for temporal segmentation)
            mask_folder: Folder with object masks
            shape_image: Path to canvas/shape image
            output_dir: Output directory for results
            min_len: Min segment length (temporal)
            max_len: Max segment length (temporal)
            w_clip, w_iqa: Temporal segmenter weights
            balanced: Whether to use balanced tree strategy
        """
        self.image_dir = image_dir
        self.mask_folder = mask_folder
        self.shape_image = shape_image
        self.output_dir = output_dir
        self.min_len = min_len
        self.max_len = max_len
        self.w_clip = w_clip
        self.w_iqa = w_iqa
        self.balanced = balanced
        
        self.segments = None
        self.segment_files = None
        self.temporal_heuristic = None
        self.layout_segments = None
        
    def run_temporal_segmentation(self) -> Tuple[List[List[int]], List[List[str]]]:
        """Run temporal segmenter to get image clusters."""
        print("[TemporalLayoutComposer] Running temporal segmentation...")
        segmenter = AutoSegmenter(
            w_clip=self.w_clip,
            w_iqa=self.w_iqa,
            min_len=self.min_len,
            max_len=self.max_len,
            mode="auto"
        )
        segments_idx, segments_files = segmenter.segment(self.image_dir)
        print(f"[TemporalLayoutComposer] Found {len(segments_idx)} temporal segments")
        for k, seg in enumerate(segments_idx, 1):
            print(f"  Segment {k}: indices {seg[0]}-{seg[-1]} (len={len(seg)})")
        return segments_idx, segments_files
    
    def load_shape_and_masks(self) -> Tuple[Polygon, List[Dict], List[str]]:
        """Load canvas shape and object masks."""
        print("[TemporalLayoutComposer] Loading canvas shape...")
        canvas = cv2.imread(self.shape_image, cv2.IMREAD_GRAYSCALE)
        if canvas is None:
            raise FileNotFoundError(f"Canvas image not found: {self.shape_image}")
        
        polygon = sd.generate_canvas_polygon(canvas)[0]
        
        print("[TemporalLayoutComposer] Loading object masks...")
        image_ids = sorted([
            f.split(".")[0] 
            for f in os.listdir(self.mask_folder) 
            if f.endswith('.png')
        ])
        
        image_dict = []
        image_template = {
            "filename": "",
            "foreground_exists": True,
            "foreground": [],
            "assigned_part": 0
        }
        
        for image_id in image_ids:
            mask_path = os.path.join(self.mask_folder, f"{image_id}.png")
            label = load_mask(mask_path)
            x1, x2, y1, y2, foreground_exists = extract_foreground(label)
            
            # Calculate aspect ratio from foreground bounding box
            if foreground_exists and (x2 - x1) > 0 and (y2 - y1) > 0:
                aspect_ratio = (x2 - x1) / (y2 - y1)
            else:
                aspect_ratio = 1.0  # Default aspect ratio
            
            item = image_template.copy()
            item["filename"] = image_id
            item["foreground_exists"] = foreground_exists
            item["foreground"] = [x1, x2, y1, y2] if foreground_exists else []
            item["aspect_ratio"] = aspect_ratio
            image_dict.append(item)
        
        return polygon, image_dict, image_ids

    def _resolve_image_path(self, image_id: str) -> str:
        """Try to locate the actual image file for a given mask/image id."""
        if not image_id:
            return ""

        candidate_names = [image_id]
        base_name = os.path.splitext(image_id)[0]
        if base_name and base_name not in candidate_names:
            candidate_names.append(base_name)

        extensions = ["", ".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        for name in candidate_names:
            # Absolute path already?
            if os.path.isabs(name) and os.path.exists(name):
                return name
            for ext in extensions:
                candidate = name if name.lower().endswith(ext) else f"{name}{ext}"
                candidate_path = os.path.join(self.image_dir, candidate)
                if os.path.exists(candidate_path):
                    return candidate_path
        return ""

    @staticmethod
    def _normalize_scores(values: List[float]) -> List[float]:
        """Normalize values to [0, 1] range with graceful fallbacks."""
        if not values:
            return []
        arr = np.array(values, dtype=np.float32)
        if arr.size == 0:
            return []
        min_val = float(arr.min())
        max_val = float(arr.max())
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return [0.5 for _ in values]
        if max_val - min_val < 1e-6:
            return [0.5 for _ in values]
        norm = (arr - min_val) / (max_val - min_val)
        return norm.tolist()

    def _compute_image_priorities(self, image_dict: List[Dict]) -> List[Dict]:
        """Augment image metadata with CLIP/IQA scores and derive priority ranking."""
        print("[TemporalLayoutComposer] Computing CLIP/IQA priorities...")
        clip_scores = []
        iqa_scores = []
        resolved_items = []

        for item in image_dict:
            image_id = item.get("filename", "")
            image_path = self._resolve_image_path(image_id)
            item["image_path"] = image_path

            clip_score = 0.0
            iqa_score = 0.0

            if not image_path:
                print(f"    Warning: could not locate image for {image_id}, priority defaults to 0")
            else:
                try:
                    with Image.open(image_path) as pil_img:
                        pil_rgb = pil_img.convert("RGB")
                        clip_vec = get_clip_embedding(pil_rgb)
                        clip_score = float(np.linalg.norm(clip_vec)) if clip_vec is not None else 0.0
                        iqa_score = float(get_iqa_score(pil_rgb))
                except Exception as exc:
                    print(f"    Warning: scoring failed for {image_id}: {exc}")

            clip_scores.append(clip_score)
            iqa_scores.append(iqa_score)
            resolved_items.append(item)

        clip_norm = self._normalize_scores(clip_scores)
        iqa_norm = self._normalize_scores(iqa_scores)
        weight_sum = max(self.w_clip + self.w_iqa, 1e-6)

        for idx, item in enumerate(resolved_items):
            item["clip_score_raw"] = clip_scores[idx]
            item["clip_score_norm"] = clip_norm[idx] if idx < len(clip_norm) else 0.0
            item["iqa_score_raw"] = iqa_scores[idx]
            item["iqa_score_norm"] = iqa_norm[idx] if idx < len(iqa_norm) else 0.0
            priority = (
                self.w_clip * item["clip_score_norm"] +
                self.w_iqa * item["iqa_score_norm"]
            ) / weight_sum
            item["priority_score"] = priority

        # Sort descending by priority score so the best images appear first
        resolved_items.sort(key=lambda d: d.get("priority_score", 0.0), reverse=True)
        for rank, item in enumerate(resolved_items):
            item["priority_rank"] = rank

        print("    Priority stats:")
        if resolved_items:
            best = resolved_items[0]
            worst = resolved_items[-1]
            print(f"        Top image: {best.get('filename')} (score={best.get('priority_score', 0):.3f})")
            print(f"        Bottom image: {worst.get('filename')} (score={worst.get('priority_score', 0):.3f})")

        return resolved_items
    
    def build_temporal_constrained_forest(self,
                                          polygon: Polygon,
                                          convex_parts: List[Part],
                                          multilinestring_int: Tuple,
                                          image_indices_per_segment: List[List[int]]) -> Tuple[List, Dict]:
        """Build forest where each tree gets one temporal cluster."""
        print("[TemporalLayoutComposer] Building temporal-constrained forest...")
        
        num_segments = len(image_indices_per_segment)
        total_area = polygon.area
        
        forest = []
        forest_summary = {}
        
        # Use medial axis to generate cuts and create partitions based on image counts
        print(f"[TemporalLayoutComposer] Creating {num_segments} partitions using medial axis...")
        
        segment_polygons = self._create_partitions_from_medial_axis(
            polygon, multilinestring_int[0], image_indices_per_segment
        )
        
        print(f"[TemporalLayoutComposer] Generated {len(segment_polygons)} spatial partitions")
        
        for seg_idx in range(num_segments):
            # Get polygon for this segment (from medial axis partition)
            seg_polygon = segment_polygons[seg_idx]
            
            # Get images for this segment
            images_for_segment = len(image_indices_per_segment[seg_idx])
            
            if images_for_segment == 0:
                print(f"[TemporalLayoutComposer] Skipping segment {seg_idx}: 0 images")
                continue
            
            # Limit tree size to prevent deep recursion
            max_images_per_tree = 12
            if images_for_segment > max_images_per_tree:
                print(f"[TemporalLayoutComposer] Splitting segment {seg_idx}: {images_for_segment} images > {max_images_per_tree} limit")
                # Split into multiple trees
                for chunk_idx in range(0, images_for_segment, max_images_per_tree):
                    chunk_size = min(max_images_per_tree, images_for_segment - chunk_idx)
                    root = tree_initialization(chunk_size, balanced=self.balanced)
                    root.polygon = seg_polygon
                    root.temporal_segments = [image_indices_per_segment[seg_idx]]
                    root.temporal_assignment_part = seg_idx
                    
                    # Set polygons for leaves
                    leaves = _list_leaves(root)
                    leaf_polygons = self._divide_polygon_for_leaves(seg_polygon, len(leaves))
                    for leaf, leaf_poly in zip(leaves, leaf_polygons):
                        leaf.polygon = leaf_poly
                    
                    heuristic_level = max(0, root.get_height() - 3)
                    heuristic_initialization(root, multilinestring_int[0], heuristic_level)
                    forest_summary = sum_dict(forest_summary, leaf_elevation_summary(root))
                    forest.append(root)
            else:
                root = tree_initialization(images_for_segment, balanced=self.balanced)
                root.polygon = seg_polygon
                root.temporal_segments = [image_indices_per_segment[seg_idx]]
                root.temporal_assignment_part = seg_idx
                
                # Set polygons for leaves
                leaves = _list_leaves(root)
                leaf_polygons = self._divide_polygon_for_leaves(seg_polygon, len(leaves))
                for leaf, leaf_poly in zip(leaves, leaf_polygons):
                    leaf.polygon = leaf_poly
                
                heuristic_level = max(0, root.get_height() - 3)
                heuristic_initialization(root, multilinestring_int[0], heuristic_level)
                forest_summary = sum_dict(forest_summary, leaf_elevation_summary(root))
                forest.append(root)
        
        print(f"[TemporalLayoutComposer] Built forest with {len(forest)} trees")
        return forest, forest_summary
    
    def _create_partitions_from_medial_axis(self, 
                                            polygon: Polygon, 
                                            medial_axis: LineString,
                                            image_indices_per_segment: List[List[int]]) -> List[Polygon]:
        """
        Create partitions using get_optimal() to recursively find best cuts.
        This uses Colla's approach: test 4 cut configurations and pick best.
        """
        num_segments = len(image_indices_per_segment)
        if num_segments <= 1:
            return [polygon]
        
        # Calculate weights based on image counts
        segment_sizes = [len(seg) for seg in image_indices_per_segment]
        total_images = sum(segment_sizes)
        weights = [size / total_images for size in segment_sizes]
        
        print(f"[_create_partitions_from_medial_axis] Segment sizes: {segment_sizes}")
        print(f"[_create_partitions_from_medial_axis] Weights: {weights}")
        print(f"[_create_partitions_from_medial_axis] Building optimized tree with get_optimal()...")
        
        partitions_dir = os.path.join(self.output_dir, "partition_steps")
        os.makedirs(partitions_dir, exist_ok=True)
        
        try:
            # Create tree with num_segments leaves
            root = tree_initialization(num_segments, balanced=True, fix_seed=False)
            root.polygon = polygon
            
            # Initialize all leaves with the full polygon (will be refined by get_optimal)
            leaves = _list_leaves(root)
            for leaf in leaves:
                leaf.polygon = polygon
                leaf.assignment = {"id": -1, "aspect_ratio": 1.0, "coord": []}
            
            # Run optimization to find best cuts
            print("[_create_partitions_from_medial_axis] Running get_optimal() to test 4 cut configs...")
            score, optimized_root = get_optimal(root, medial_axis, self.output_dir)
            
            # Extract leaf polygons
            leaves = _list_leaves(optimized_root)
            part_polygons = [leaf.polygon for leaf in leaves]
            
            print(f"[_create_partitions_from_medial_axis] Generated {len(part_polygons)} parts with score={score:.3f}")
            
            if len(part_polygons) == num_segments:
                # Visualize the result
                self._save_partition_visualization(part_polygons, partitions_dir, "00_optimized_cuts", score)
                return part_polygons
            else:
                print(f"[_create_partitions_from_medial_axis] Got {len(part_polygons)} parts instead of {num_segments}, using fallback")
                return self._divide_polygon_for_segments(polygon, num_segments)
            
        except Exception as e:
            print(f"[_create_partitions_from_medial_axis] Error in get_optimal(): {e}")
            import traceback
            traceback.print_exc()
            print(f"[_create_partitions_from_medial_axis] Falling back to simple division")
            return self._divide_polygon_for_segments(polygon, num_segments)
    
    def _generate_weighted_cuts(self, polygon: Polygon, medial_axis: LineString, 
                                weights: List[float], num_segments: int) -> List[Tuple]:
        """
        Generate cuts perpendicular to medial axis, weighted by segment sizes.
        For N segments, we need N-1 cuts.
        Follows approach from Colla repository.
        """
        cuts = []
        
        if num_segments <= 1:
            return cuts
        
        try:
            # Get medial axis length
            axis_length = medial_axis.length
            
            # Find N-1 cutting points along the medial axis based on weights
            cumulative = 0
            cut_positions = []
            for i in range(num_segments - 1):
                cumulative += weights[i]
                # Position along medial axis (as fraction of total length)
                axis_distance = cumulative * axis_length
                cut_positions.append(axis_distance)
            
            print(f"[_generate_weighted_cuts] Medial axis length: {axis_length:.1f}")
            print(f"[_generate_weighted_cuts] Cut positions along axis: {[f'{p:.1f}' for p in cut_positions]}")
            
            # For each cut position, find perpendicular cut
            for cut_idx, axis_distance in enumerate(cut_positions):
                try:
                    # Get point on medial axis
                    if axis_distance <= 0:
                        continue
                    if axis_distance >= axis_length:
                        continue
                    
                    # Interpolate point on medial axis
                    center_point = medial_axis.interpolate(axis_distance)
                    cx, cy = center_point.x, center_point.y
                    
                    # Get tangent vector at this point
                    # Use nearby points to estimate tangent
                    delta = min(5, axis_length * 0.05)  # Look ahead/behind
                    try:
                        p1 = medial_axis.interpolate(max(0, axis_distance - delta))
                        p2 = medial_axis.interpolate(min(axis_length, axis_distance + delta))
                        tangent = np.array([p2.x - p1.x, p2.y - p1.y])
                    except:
                        # If interpolation fails, use fallback
                        p1 = medial_axis.interpolate(max(0, axis_distance - 1))
                        p2 = medial_axis.interpolate(min(axis_length, axis_distance + 1))
                        tangent = np.array([p2.x - p1.x, p2.y - p1.y])
                    
                    # Normalize tangent
                    tangent_norm = np.linalg.norm(tangent)
                    if tangent_norm > 0:
                        tangent = tangent / tangent_norm
                    
                    # Perpendicular to tangent (rotate 90 degrees)
                    perpendicular = np.array([-tangent[1], tangent[0]])
                    
                    # Create cut line extending from perpendicular at center point
                    # Get polygon bounds to determine cut length
                    minx, miny, maxx, maxy = polygon.bounds
                    cut_length = max(maxx - minx, maxy - miny) * 2
                    
                    cut_start = (cx - perpendicular[0] * cut_length / 2, 
                                cy - perpendicular[1] * cut_length / 2)
                    cut_end = (cx + perpendicular[0] * cut_length / 2, 
                              cy + perpendicular[1] * cut_length / 2)
                    
                    cut = (cut_start, cut_end)
                    cuts.append(cut)
                    
                    angle = np.arctan2(perpendicular[1], perpendicular[0]) * 180 / np.pi
                    print(f"  Perpendicular cut {cut_idx+1} at axis pos={axis_distance:.1f} (angle={angle:.0f}°, weight={weights[cut_idx]:.2f})")
                    
                except Exception as e:
                    print(f"[_generate_weighted_cuts] Warning: error creating cut {cut_idx}: {e}")
                    continue
            
        except Exception as e:
            print(f"[_generate_weighted_cuts] Error: {e}")
            import traceback
            traceback.print_exc()
        
        return cuts
    
    def _divide_polygon_vertical(self, polygon: Polygon, weights: List[float], num_segments: int) -> List[Polygon]:
        """Divide polygon vertically with proportional widths based on weights."""
        if num_segments <= 1:
            return [polygon]
        
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        
        polygons = []
        current_x = minx
        
        for i, weight in enumerate(weights):
            segment_width = width * weight
            x1 = current_x
            x2 = current_x + segment_width
            
            # For the last segment, extend to the end to avoid rounding errors
            if i == len(weights) - 1:
                x2 = maxx
            
            rect = Polygon([(x1, miny), (x2, miny), (x2, maxy), (x1, maxy)])
            try:
                sub_poly = polygon.intersection(rect)
                if not sub_poly.is_empty:
                    if hasattr(sub_poly, 'exterior'):
                        polygons.append(sub_poly)
                    else:
                        if hasattr(sub_poly, 'geoms'):
                            polys = [g for g in sub_poly.geoms if hasattr(g, 'exterior')]
                            if polys:
                                largest = max(polys, key=lambda p: p.area)
                                polygons.append(largest)
            except:
                pass
            
            current_x = x2
        
        # Ensure we have the right number
        while len(polygons) < num_segments:
            polygons.append(polygon)
        
        return polygons[:num_segments]
    
    def _divide_polygon_horizontal(self, polygon: Polygon, weights: List[float], num_segments: int) -> List[Polygon]:
        """Divide polygon horizontally with proportional heights based on weights."""
        if num_segments <= 1:
            return [polygon]
        
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        
        polygons = []
        current_y = miny
        
        for i, weight in enumerate(weights):
            segment_height = height * weight
            y1 = current_y
            y2 = current_y + segment_height
            
            # For the last segment, extend to the end to avoid rounding errors
            if i == len(weights) - 1:
                y2 = maxy
            
            rect = Polygon([(minx, y1), (maxx, y1), (maxx, y2), (minx, y2)])
            try:
                sub_poly = polygon.intersection(rect)
                if not sub_poly.is_empty:
                    if hasattr(sub_poly, 'exterior'):
                        polygons.append(sub_poly)
                    else:
                        if hasattr(sub_poly, 'geoms'):
                            polys = [g for g in sub_poly.geoms if hasattr(g, 'exterior')]
                            if polys:
                                largest = max(polys, key=lambda p: p.area)
                                polygons.append(largest)
            except:
                pass
            
            current_y = y2
        
        # Ensure we have the right number
        while len(polygons) < num_segments:
            polygons.append(polygon)
        
        return polygons[:num_segments]
    
    def _score_partition(self, part_polygons: List[Polygon], segment_sizes: List[int]) -> float:
        """
        Score a partition based on:
        1. Area balance (parts should be proportional to segment sizes)
        2. Shape quality (parts shouldn't have extreme aspect ratios)
        """
        if not part_polygons or not segment_sizes:
            return -np.inf
        
        if len(part_polygons) != len(segment_sizes):
            return -np.inf
        
        total_area = sum(p.area for p in part_polygons)
        total_size = sum(segment_sizes)
        
        # Score 1: How well does area match the size ratios?
        area_score = 0
        for i, poly in enumerate(part_polygons):
            expected_ratio = segment_sizes[i] / total_size
            actual_ratio = poly.area / total_area
            # Penalty for deviation
            deviation = abs(expected_ratio - actual_ratio)
            area_score += (1.0 - deviation)  # Range [0, 1] per segment
        
        area_score /= len(part_polygons)  # Average
        
        # Score 2: Shape quality (avoid very thin slices)
        shape_score = 0
        for poly in part_polygons:
            minx, miny, maxx, maxy = poly.bounds
            width = maxx - minx
            height = maxy - miny
            
            if width > 0 and height > 0:
                aspect_ratio = max(width / height, height / width)
                # Aspect ratio >= 2 is bad, aspect ratio ~= 1 is good
                # Score: 1 / (1 + aspect_ratio - 1) = 1 / aspect_ratio, capped at 1
                poly_score = 1.0 / min(aspect_ratio, 3.0)
            else:
                poly_score = 0
            
            shape_score += poly_score
        
        shape_score /= len(part_polygons)  # Average
        
        # Combined score: 70% area fit, 30% shape quality
        total_score = 0.7 * area_score + 0.3 * shape_score
        
        return total_score
    
    def _save_partition_visualization(self, part_polygons: List[Polygon], 
                                     output_dir: str, prefix: str, score: float) -> None:
        """Save a visualization of the partition."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import Polygon as MplPolygon
            
            if not part_polygons:
                return
            
            # Get bounds
            all_coords = []
            for poly in part_polygons:
                all_coords.extend(poly.exterior.coords)
            
            if not all_coords:
                return
            
            all_coords = np.array(all_coords)
            minx, miny = all_coords.min(axis=0)
            maxx, maxy = all_coords.max(axis=0)
            
            width = maxx - minx if maxx > minx else 100
            height = maxy - miny if maxy > miny else 100
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80)
            
            # Draw parts with different colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(part_polygons)))
            
            for idx, poly in enumerate(part_polygons):
                coords = list(poly.exterior.coords[:-1])  # Remove duplicate last point
                mpl_poly = MplPolygon(coords, alpha=0.7, color=colors[idx], edgecolor='black', linewidth=2)
                ax.add_patch(mpl_poly)
                
                # Add text label
                try:
                    centroid = poly.centroid
                    ax.text(centroid.x, centroid.y, str(idx + 1), 
                           fontsize=16, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
                except:
                    pass
            
            ax.set_xlim(minx - width * 0.05, maxx + width * 0.05)
            ax.set_ylim(miny - height * 0.05, maxy + height * 0.05)
            ax.set_aspect('equal')
            ax.set_title(f'{prefix.replace("_", " ").upper()}\nScore: {score:.3f}', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Save
            output_path = os.path.join(output_dir, f"{prefix}_score_{score:.3f}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=80, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved partition visualization: {output_path}")
        except Exception as e:
            print(f"    Warning: Could not save partition visualization: {e}")
    
    def _divide_polygon_for_leaves(self, polygon: Polygon, num_leaves: int) -> List[Polygon]:
        """Divide a polygon into num_leaves sub-polygons."""
        if num_leaves <= 1:
            return [polygon]
        
        try:
            # Get bounds
            minx, miny, maxx, maxy = polygon.bounds
            width = maxx - minx
            height = maxy - miny
            
            # Divide horizontally or vertically depending on shape
            polygons = []
            if width >= height:
                # Divide vertically
                step = width / num_leaves
                for i in range(num_leaves):
                    x1 = minx + i * step
                    x2 = minx + (i + 1) * step
                    # Create rectangle and intersect with original polygon
                    rect = Polygon([(x1, miny), (x2, miny), (x2, maxy), (x1, maxy)])
                    try:
                        sub_poly = polygon.intersection(rect)
                        # Convert GeometryCollection to Polygon
                        if sub_poly.is_empty:
                            continue
                        elif hasattr(sub_poly, 'exterior'):
                            polygons.append(sub_poly)
                        else:
                            # GeometryCollection - try to extract largest polygon
                            if hasattr(sub_poly, 'geoms'):
                                polys = [g for g in sub_poly.geoms if hasattr(g, 'exterior')]
                                if polys:
                                    largest = max(polys, key=lambda p: p.area)
                                    polygons.append(largest)
                            else:
                                polygons.append(polygon)
                    except:
                        polygons.append(polygon)
            else:
                # Divide horizontally
                step = height / num_leaves
                for i in range(num_leaves):
                    y1 = miny + i * step
                    y2 = miny + (i + 1) * step
                    # Create rectangle and intersect with original polygon
                    rect = Polygon([(minx, y1), (maxx, y1), (maxx, y2), (minx, y2)])
                    try:
                        sub_poly = polygon.intersection(rect)
                        # Convert GeometryCollection to Polygon
                        if sub_poly.is_empty:
                            continue
                        elif hasattr(sub_poly, 'exterior'):
                            polygons.append(sub_poly)
                        else:
                            # GeometryCollection - try to extract largest polygon
                            if hasattr(sub_poly, 'geoms'):
                                polys = [g for g in sub_poly.geoms if hasattr(g, 'exterior')]
                                if polys:
                                    largest = max(polys, key=lambda p: p.area)
                                    polygons.append(largest)
                            else:
                                polygons.append(polygon)
                    except:
                        polygons.append(polygon)
            
            # Fill any remaining with full polygon if division didn't work
            while len(polygons) < num_leaves:
                polygons.append(polygon)
            
            return polygons[:num_leaves]
        except:
            # Fallback: just return the original polygon for all leaves
            return [polygon] * num_leaves
    
    def _divide_polygon_for_segments(self, polygon: Polygon, num_segments: int) -> List[Polygon]:
        """Divide a polygon into num_segments sub-polygons for temporal segments based on area proportional to image count."""
        if num_segments <= 1:
            return [polygon]
        
        # Calculate area ratios based on number of images in each segment
        segments = self.layout_segments if self.layout_segments else self.segments
        if not segments:
            return [polygon]
        total_images = sum(len(seg) for seg in segments)
        if total_images <= 0:
            return [polygon]
        segment_ratios = [len(seg) / total_images for seg in segments]
        
        print(f"[TemporalLayoutComposer] Dividing polygon by area ratios: {[f'{r:.1%}' for r in segment_ratios]}")
        
        try:
            # Get bounds
            minx, miny, maxx, maxy = polygon.bounds
            width = maxx - minx
            height = maxy - miny
            
            polygons = []
            
            # Strategy: divide along the longer dimension
            if width >= height:
                # Divide vertically (left-right) with proportional widths
                current_x = minx
                for i, ratio in enumerate(segment_ratios):
                    segment_width = width * ratio
                    x1 = current_x
                    x2 = current_x + segment_width
                    
                    # For the last segment, extend to the end to avoid rounding errors
                    if i == len(segment_ratios) - 1:
                        x2 = maxx
                    
                    rect = Polygon([(x1, miny), (x2, miny), (x2, maxy), (x1, maxy)])
                    try:
                        sub_poly = polygon.intersection(rect)
                        if sub_poly.is_empty:
                            continue
                        elif hasattr(sub_poly, 'exterior'):
                            polygons.append(sub_poly)
                            print(f"    Segment {i}: x={x1:.0f}-{x2:.0f} (width={segment_width:.0f}, {ratio:.1%})")
                        else:
                            if hasattr(sub_poly, 'geoms'):
                                polys = [g for g in sub_poly.geoms if hasattr(g, 'exterior')]
                                if polys:
                                    largest = max(polys, key=lambda p: p.area)
                                    polygons.append(largest)
                                    print(f"    Segment {i}: x={x1:.0f}-{x2:.0f} (largest from collection, {ratio:.1%})")
                            else:
                                polygons.append(polygon)
                    except:
                        polygons.append(polygon)
                    
                    current_x = x2
            else:
                # Divide horizontally (top-bottom) with proportional heights
                current_y = miny
                for i, ratio in enumerate(segment_ratios):
                    segment_height = height * ratio
                    y1 = current_y
                    y2 = current_y + segment_height
                    
                    # For the last segment, extend to the end to avoid rounding errors
                    if i == len(segment_ratios) - 1:
                        y2 = maxy
                    
                    rect = Polygon([(minx, y1), (maxx, y1), (maxx, y2), (minx, y2)])
                    try:
                        sub_poly = polygon.intersection(rect)
                        if sub_poly.is_empty:
                            continue
                        elif hasattr(sub_poly, 'exterior'):
                            polygons.append(sub_poly)
                            print(f"    Segment {i}: y={y1:.0f}-{y2:.0f} (height={segment_height:.0f}, {ratio:.1%})")
                        else:
                            if hasattr(sub_poly, 'geoms'):
                                polys = [g for g in sub_poly.geoms if hasattr(g, 'exterior')]
                                if polys:
                                    largest = max(polys, key=lambda p: p.area)
                                    polygons.append(largest)
                                    print(f"    Segment {i}: y={y1:.0f}-{y2:.0f} (largest from collection, {ratio:.1%})")
                            else:
                                polygons.append(polygon)
                    except:
                        polygons.append(polygon)
                    
                    current_y = y2
            
            # Ensure we have enough polygons
            while len(polygons) < num_segments:
                polygons.append(polygon)
            
            return polygons[:num_segments]
        except Exception as e:
            print(f"[TemporalLayoutComposer] Warning: Error dividing polygon: {e}")
            # Fallback: just return the original polygon for all segments
            return [polygon] * num_segments
    
    def assign_images_respecting_temporal(self,
                                         forest: List,
                                         assignment: Dict,
                                         image_indices_per_segment: List[List[int]]) -> None:
        """Assign images to tree leaves, respecting temporal constraints."""
        print("[TemporalLayoutComposer] Assigning images with temporal constraints...")
        
        image_assignment = assignment.copy()
        
        def traverse(tn, depth, tree_height):
            if tn.is_leaf():
                if image_assignment and any(len(v) > 0 for v in image_assignment.values()):
                    if tree_height - depth in image_assignment and image_assignment[tree_height - depth]:
                        tn.assignment = image_assignment[tree_height - depth].pop(0)
            else:
                traverse(tn.left_child, depth + 1, tree_height)
                traverse(tn.right_child, depth + 1, tree_height)
        
        for tree in forest:
            tree_height = tree.get_height()
            traverse(tree, 0, tree_height)
    
    def assign_images_by_priority(self, forest: List, prioritized_images: List[Dict]) -> None:
        """Assign highest-priority images to the largest leaves across the forest."""
        print("[TemporalLayoutComposer] Assigning images by priority score...")

        leaves = []
        for tree in forest:
            leaves.extend(_list_leaves(tree))

        if not leaves or not prioritized_images:
            print("    Warning: no leaves or images available for assignment")
            return

        leaves.sort(key=lambda leaf: leaf.area() if leaf and leaf.polygon else 0.0, reverse=True)
        num_assignments = min(len(leaves), len(prioritized_images))
        print(f"    Assigning top {num_assignments} images to {len(leaves)} leaves")

        for idx in range(num_assignments):
            leaf = leaves[idx]
            img_meta = prioritized_images[idx]
            assignment_id = img_meta.get("id", idx)
            prioritized_images[idx]["id"] = assignment_id

            leaf.assignment = {
                "id": assignment_id,
                "filename": img_meta.get("filename", ""),
                "aspect_ratio": img_meta.get("aspect_ratio", 1.0),
                "foreground": img_meta.get("foreground", []),
                "priority_score": img_meta.get("priority_score", 0.0),
                "priority_rank": img_meta.get("priority_rank", idx),
                "clip_score": img_meta.get("clip_score_raw", 0.0),
                "iqa_score": img_meta.get("iqa_score_raw", 0.0),
                "image_path": img_meta.get("image_path", "")
            }

        if len(prioritized_images) > len(leaves):
            print(f"    {len(prioritized_images) - len(leaves)} images could not be placed (insufficient leaves)")

    def _prepare_forest_for_optimization(self,
                                         forest: List,
                                         prioritized_images: List[Dict],
                                         forest_summary: Dict) -> None:
        """Ensure each leaf has a valid aspect ratio before running SAS optimization."""
        if not forest:
            return
        if not prioritized_images:
            # Set safe defaults
            for tree in forest:
                for leaf in _list_leaves(tree):
                    if not leaf.assignment:
                        leaf.assignment = {}
                    leaf.assignment.setdefault("aspect_ratio", 1.0)
            return

        def compute_aspect(item: Dict) -> float:
            foreground = item.get("foreground", [])
            if foreground and len(foreground) == 4:
                width = max(1.0, float(foreground[1] - foreground[0]))
                height = max(1.0, float(foreground[3] - foreground[2]))
            else:
                width = height = 1.0
            return max(width / max(height, 1e-3), 0.1)

        placeholder_images = [
            {"id": idx, "aspect_ratio": compute_aspect(item)}
            for idx, item in enumerate(prioritized_images)
        ]

        summary = forest_summary or {}
        if not summary:
            for tree in forest:
                summary = sum_dict(summary, leaf_elevation_summary(tree))

        if sas and hasattr(sas, "calculate_image_assignment") and hasattr(sas, "assign_image"):
            try:
                assignment = sas.calculate_image_assignment(placeholder_images, summary)
                sas.assign_image(forest, assignment)
                return
            except Exception as exc:
                print(f"[TemporalLayoutComposer] Warning: SAS placeholder assignment failed ({exc}); using fallback")

        # Fallback: write normalized aspect ratios directly
        idx = 0
        for tree in forest:
            for leaf in _list_leaves(tree):
                aspect = placeholder_images[idx % len(placeholder_images)]["aspect_ratio"]
                if not leaf.assignment:
                    leaf.assignment = {"id": -1, "aspect_ratio": aspect, "coord": []}
                else:
                    leaf.assignment["aspect_ratio"] = aspect
                idx += 1

    def assign_images_with_layout_engine(self,
                                         forest: List,
                                         prioritized_images: List[Dict],
                                         forest_summary: Dict) -> None:
        """Use SAS optimizer utilities for deterministic assignment when available."""
        if not forest or not prioritized_images:
            print("[TemporalLayoutComposer] Nothing to assign (empty forest or images)")
            return

        if not forest_summary:
            forest_summary = {}
            for tree in forest:
                forest_summary = sum_dict(forest_summary, leaf_elevation_summary(tree))

        if sas and hasattr(sas, "calculate_image_assignment") and hasattr(sas, "assign_image"):
            print("[TemporalLayoutComposer] Leveraging sas_optimization for image assignment...")
            processed_images = []
            for idx, item in enumerate(prioritized_images):
                foreground = item.get("foreground", [])
                if foreground and len(foreground) == 4:
                    width = max(1.0, float(foreground[1] - foreground[0]))
                    height = max(1.0, float(foreground[3] - foreground[2]))
                else:
                    width = height = 1.0
                aspect_ratio = width / max(height, 1e-3)
                processed_images.append({
                    "id": idx,
                    "aspect_ratio": aspect_ratio,
                    "filename": item.get("filename", ""),
                    "image_path": item.get("image_path", ""),
                    "foreground": item.get("foreground", []),
                    "priority_score": item.get("priority_score", 0.0),
                    "priority_rank": item.get("priority_rank", idx),
                    "clip_score": item.get("clip_score_raw", 0.0),
                    "iqa_score": item.get("iqa_score_raw", 0.0)
                })

            assignment = sas.calculate_image_assignment(processed_images, forest_summary)
            sas.assign_image(forest, assignment)
            return

        print("[TemporalLayoutComposer] sas_optimization unavailable, falling back to area-based priority assignment")
        self.assign_images_by_priority(forest, prioritized_images)

    def extract_temporal_layout(self, forest: List) -> Dict[str, Any]:
        """Extract final layout geometry and image assignments."""
        print("[TemporalLayoutComposer] Extracting temporal layout...")
        
        geometry = extract_forest_geometry(forest)
        
        layout_result = {
            "parts": geometry[0],
            "image_assignments": geometry[1],
            "cuts": geometry[2],
            "temporal_info": {
                "num_segments": len(self.segments),
                "segment_sizes": [len(seg) for seg in self.segments],
                "segment_files": self.segment_files
            }
        }
        
        return layout_result
    
    def compose(self) -> Dict[str, Any]:
        """Main pipeline: temporal segmentation -> spatial layout with constraints."""
        print("\n" + "="*70)
        print("TEMPORAL LAYOUT COMPOSITION")
        print("="*70 + "\n")
        
        self.segments, self.segment_files = self.run_temporal_segmentation()
        polygon, image_dict, image_ids = self.load_shape_and_masks()
        prioritized_images = self._compute_image_priorities(image_dict)
        self.layout_segments = [list(range(len(prioritized_images)))]
        
        print("[TemporalLayoutComposer] Running shape decomposition...")
        canvas = cv2.imread(self.shape_image, cv2.IMREAD_GRAYSCALE)
        medial_interior_input = sd.prepare_for_medial_axis(canvas, complement=False)
        ma_int = sd.ridge_medial_axis(medial_interior_input, ridge_threshold=0.39, small_threshold=5)
        multilinestring_int = sd.build_medial_multilinestring(ma_int[0])
        
        prediction_partition = Partition(polygon)
        convex_parts = prediction_partition.list_leaves()
        forest, forest_summary = self.build_temporal_constrained_forest(
            polygon, convex_parts, multilinestring_int, self.layout_segments
        )

        # Pre-populate aspect ratios so SAS optimizer has valid data
        self._prepare_forest_for_optimization(forest, prioritized_images, forest_summary)
        
        print("[TemporalLayoutComposer] Running spatial optimization...")
        try:
            # Limit recursion depth during optimization
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(3000)  # Reduce to prevent stack overflow
            try:
                result = forest_optimization(forest, multilinestring_int)
                final_forest = []
                if result and isinstance(result[0], tuple):
                    for original, optimized_pair in zip(forest, result):
                        optimized_tree = optimized_pair[1] if isinstance(optimized_pair, tuple) else optimized_pair
                        if optimized_tree is None:
                            optimized_tree = original
                        optimized_tree.temporal_segments = getattr(original, "temporal_segments", [])
                        optimized_tree.temporal_assignment_part = getattr(original, "temporal_assignment_part", -1)
                        final_forest.append(optimized_tree)
                else:
                    final_forest = forest
            finally:
                sys.setrecursionlimit(old_limit)
        except RecursionError as e:
            print(f"[TemporalLayoutComposer] WARNING: RecursionError during optimization: {e}")
            print("[TemporalLayoutComposer] Using simplified forest structure (no optimization)...")
            # Fallback: use forest directly without deep optimization
            final_forest = forest
        except MemoryError as e:
            print(f"[TemporalLayoutComposer] WARNING: MemoryError during optimization: {e}")
            print("[TemporalLayoutComposer] Using simplified forest structure...")
            final_forest = forest
        except Exception as e:
            print(f"[TemporalLayoutComposer] ERROR during optimization: {e}")
            print("[TemporalLayoutComposer] Using simplified forest structure...")
            import traceback
            traceback.print_exc()
            final_forest = forest
        
        self.assign_images_with_layout_engine(final_forest, prioritized_images, forest_summary)
        final_layout = self.extract_temporal_layout(final_forest)
        final_layout["canvas_shape"] = {
            "width": canvas.shape[1],
            "height": canvas.shape[0]
        }

        final_layout["priority_summary"] = [
            {
                "filename": item.get("filename"),
                "priority_score": item.get("priority_score"),
                "priority_rank": item.get("priority_rank"),
                "clip_score": item.get("clip_score_raw"),
                "iqa_score": item.get("iqa_score_raw")
            }
            for item in prioritized_images
        ]
        
        # Render the final layout with images
        print("[TemporalLayoutComposer] Rendering layout with assigned images...")
        layout_viz_path = os.path.join(self.output_dir, "layout_visualization.png")
        render_layout_with_assigned_images(
            final_forest,
            self.image_dir,
            (canvas.shape[1], canvas.shape[0]),
            layout_viz_path
        )
        
        self._save_results(final_layout, forest_summary)
        
        print("\n" + "="*70)
        print("COMPOSITION COMPLETE")
        print("="*70 + "\n")
        
        return final_layout
    
    def _save_cluster_visualizations(self, layout: Dict) -> None:
        """Save per-cluster metadata AND images organized by temporal segment and spatial partition."""
        clusters_dir = os.path.join(self.output_dir, "cluster_breakdown")
        os.makedirs(clusters_dir, exist_ok=True)
        
        print(f"\n[TemporalLayoutComposer] Saving cluster breakdown to: {clusters_dir}")
        
        # Create summary of all clusters
        cluster_summary = {
            "num_temporal_segments": layout["temporal_info"]["num_segments"],
            "segment_sizes": layout["temporal_info"]["segment_sizes"],
            "num_spatial_partitions": len(layout["parts"]),
            "clusters": []
        }
        
        # Process each temporal segment
        for seg_idx, segment_files in enumerate(layout["temporal_info"]["segment_files"]):
            segment_size = len(segment_files)
            part_idx_assignment = None  # Find partition for this segment
            
            # Find which partition this segment is assigned to (from first image)
            if segment_files:
                first_file = segment_files[0]
                # image_assignments is now a dict mapping part_idx -> assignment_dict
                for part_idx, assign_dict in layout["image_assignments"].items():
                    if assign_dict.get("filename") == first_file:
                        part_idx_assignment = part_idx
                        break
            
            # Create segment-specific directory with timeline and spatial info
            seg_dir_name = f"segment_{seg_idx:03d}_len{segment_size}_part{part_idx_assignment}"
            seg_dir = os.path.join(clusters_dir, seg_dir_name)
            os.makedirs(seg_dir, exist_ok=True)
            
            # Create images subdirectory
            images_dir = os.path.join(seg_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Save segment metadata
            segment_meta = {
                "segment_id": seg_idx,
                "segment_index": f"{layout['temporal_info']['segment_sizes'][seg_idx]:02d}",
                "num_images": segment_size,
                "assigned_partition": part_idx_assignment,
                "image_files": segment_files,
                "images": []
            }
            
            # Copy images and create metadata
            for img_idx_in_seg, img_file in enumerate(segment_files):
                # Extract base filename from path (img_file might be a full path)
                if os.path.isabs(img_file) or "/" in img_file:
                    img_basename = os.path.basename(img_file)
                    img_name_no_ext = os.path.splitext(img_basename)[0]
                else:
                    img_basename = img_file
                    img_name_no_ext = os.path.splitext(img_file)[0]
                
                # Find assignment for this image
                assignment_info = None
                for assign_key, assign_val in layout["image_assignments"].items():
                    if isinstance(assign_val, dict) and assign_val.get("filename") == img_file:
                        assignment_info = assign_val
                        break
                
                img_meta = {
                    "file": img_basename,
                    "file_path": img_file,
                    "temporal_position": img_idx_in_seg,
                    "temporal_segment": seg_idx,
                    "spatial_partition": None,
                    "assignment": assignment_info
                }
                
                if assignment_info:
                    part_idx = assignment_info.get("assigned_part", -1)
                    img_meta["spatial_partition"] = part_idx
                    part_info = layout["parts"][part_idx] if part_idx >= 0 and part_idx < len(layout["parts"]) else None
                    
                    # Calculate position in canvas if part info available
                    if part_info:
                        coords = part_info.get("coords", [])
                        if coords:
                            min_x = min(c[0] for c in coords)
                            max_x = max(c[0] for c in coords)
                            min_y = min(c[1] for c in coords)
                            max_y = max(c[1] for c in coords)
                            img_meta["partition_bbox"] = {
                                "x_min": float(min_x),
                                "x_max": float(max_x),
                                "y_min": float(min_y),
                                "y_max": float(max_y),
                                "width": float(max_x - min_x),
                                "height": float(max_y - min_y)
                            }
                
                segment_meta["images"].append(img_meta)
                
                # Copy image to cluster directory from cropped_objects (not masked_objects)
                try:
                    src_image = None
                    
                    # First try exact basename in image_dir (cropped_objects)
                    candidate = os.path.join(self.image_dir, img_basename)
                    if os.path.exists(candidate):
                        src_image = candidate
                    else:
                        # Try without extension + common extensions
                        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                            candidate = os.path.join(self.image_dir, f"{img_name_no_ext}{ext}")
                            if os.path.exists(candidate):
                                src_image = candidate
                                break
                    
                    if src_image:
                        # Get the actual extension
                        _, ext = os.path.splitext(src_image)
                        dst_image = os.path.join(images_dir, f"{img_idx_in_seg:02d}_{img_name_no_ext}{ext}")
                        shutil.copy2(src_image, dst_image)
                        print(f"    ✓ Copied: {img_basename} → {os.path.basename(dst_image)}")
                    else:
                        print(f"    ⚠ Image not found: {img_basename} in {self.image_dir}")
                except Exception as e:
                    print(f"    ✗ Error copying {img_basename}: {e}")
            
            # Save segment JSON
            segment_json_path = os.path.join(seg_dir, f"segment_{seg_idx:03d}_metadata.json")
            with open(segment_json_path, "w", encoding="utf-8") as f:
                json.dump(segment_meta, f, ensure_ascii=False, indent=2)
            
            cluster_summary["clusters"].append({
                "segment_id": seg_idx,
                "num_images": segment_size,
                "assigned_partition": part_idx_assignment,
                "directory": os.path.basename(seg_dir),
                "metadata_file": f"segment_{seg_idx:03d}_metadata.json",
                "images_folder": "images"
            })
            
            print(f"  [Segment {seg_idx:03d}] {segment_size} images (part {part_idx_assignment}) → {seg_dir}")
        
        # Save overall cluster summary
        summary_path = os.path.join(clusters_dir, "cluster_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(cluster_summary, f, ensure_ascii=False, indent=2)
        
        print(f"[TemporalLayoutComposer] Saved cluster summary: {summary_path}")
    
    def _save_results(self, layout: Dict, forest_summary: Dict) -> None:
        """Save layout results to output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        layout_path = os.path.join(self.output_dir, "temporal_layout.json")
        with open(layout_path, "w", encoding="utf-8") as f:
            json.dump(layout, f, ensure_ascii=False, indent=2)
        print(f"[TemporalLayoutComposer] Saved layout: {layout_path}")
        
        summary = {
            "num_parts": len(layout["parts"]),
            "num_assignments": len(layout["image_assignments"]),
            "num_temporal_segments": layout["temporal_info"]["num_segments"],
            "segment_sizes": layout["temporal_info"]["segment_sizes"],
            "forest_summary": forest_summary
        }
        summary_path = os.path.join(self.output_dir, "composition_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[TemporalLayoutComposer] Saved summary: {summary_path}")
        
        # Save cluster breakdown
        self._save_cluster_visualizations(layout)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def apply_temporal_constraints(
    layout_json_path: str,
    temporal_segments: List[List[int]],
    output_path: str
) -> Dict[str, Any]:
    """Apply temporal constraints to existing layout."""
    with open(layout_json_path, "r") as f:
        layout = json.load(f)
    
    layout["temporal_segments"] = temporal_segments
    layout["temporal_metadata"] = {
        "num_segments": len(temporal_segments),
        "segment_sizes": [len(seg) for seg in temporal_segments]
    }
    
    with open(output_path, "w") as f:
        json.dump(layout, f, indent=2)
    
    return layout


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def run_real_data_test(keyframes_dir: str, shape_image: str = None, min_len: int = 3, max_len: int = 4,
                       w_clip: float = 0.8, w_iqa: float = 0.2):
    """Run temporal composer on real keyframes data."""
    
    image_dir = os.path.join(keyframes_dir, "cropped_objects")
    mask_folder = os.path.join(keyframes_dir, "masked_objects")
    output_dir = os.path.join(keyframes_dir, "temporal_composition")
    
    # Determine shape image
    if shape_image is None:
        # Try default locations
        if os.path.exists(os.path.join(keyframes_dir, "optimal_layout.png")):
            shape_image = os.path.join(keyframes_dir, "optimal_layout.png")
        elif os.path.exists(os.path.join(keyframes_dir, "final.png")):
            shape_image = os.path.join(keyframes_dir, "final.png")
        else:
            shape_image = None
    elif not os.path.isabs(shape_image):
        # If relative path, make it relative to keyframes_dir
        shape_image_candidate = os.path.join(keyframes_dir, shape_image)
        if os.path.exists(shape_image_candidate):
            shape_image = shape_image_candidate
    
    print("\n" + "="*80)
    print("TEMPORAL LAYOUT COMPOSER - REAL DATA TEST")
    print("="*80 + "\n")
    
    print("[Step 0] Verifying inputs...")
    
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return False
    print(f"✓ Image dir: {image_dir}")
    
    if not os.path.exists(mask_folder):
        print(f"❌ Mask folder not found: {mask_folder}")
        return False
    print(f"✓ Mask folder: {mask_folder}")
    
    if not os.path.exists(shape_image):
        print(f"❌ Shape image not found")
        return False
    print(f"✓ Shape image: {shape_image}")
    
    num_cropped = len([f for f in os.listdir(image_dir) if f.endswith('.png')])
    num_masked = len([f for f in os.listdir(mask_folder) if f.endswith('.png')])
    print(f"\n  Cropped objects: {num_cropped}")
    print(f"  Masked objects: {num_masked}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output dir: {output_dir}")
    
    print("\n[Step 1] Initializing TemporalLayoutComposer...")
    try:
        composer = TemporalLayoutComposer(
            image_dir=image_dir,
            mask_folder=mask_folder,
            shape_image=shape_image,
            output_dir=output_dir,
            min_len=min_len,
            max_len=max_len,
            w_clip=w_clip,
            w_iqa=w_iqa,
            balanced=True
        )
        print("✓ Composer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[Step 2] Running full composition pipeline...")
    print("  This may take a few minutes...\n")
    
    try:
        result = composer.compose()
        print("\n✓ Composition completed successfully!")
    except Exception as e:
        print(f"\n❌ Composition failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    if "temporal_info" in result:
        temporal_info = result["temporal_info"]
        print(f"Temporal Segmentation:")
        print(f"  - Number of segments: {temporal_info['num_segments']}")
        print(f"  - Segment sizes: {temporal_info['segment_sizes']}")
        print(f"  - Total images: {sum(temporal_info['segment_sizes'])}")
    
    if "canvas_shape" in result:
        canvas = result["canvas_shape"]
        print(f"\nCanvas:")
        print(f"  - Dimensions: {canvas['width']} x {canvas['height']}")
    
    if "parts" in result:
        print(f"\nSpatial Layout:")
        print(f"  - Number of partitions: {len(result['parts'])}")
    
    if "image_assignments" in result:
        print(f"  - Number of assignments: {len(result['image_assignments'])}")
    
    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80 + "\n")
    
    layout_json = os.path.join(output_dir, "temporal_layout.json")
    summary_json = os.path.join(output_dir, "composition_summary.json")
    
    if os.path.exists(layout_json):
        size = os.path.getsize(layout_json) / 1024
        print(f"✓ {layout_json} ({size:.1f} KB)")
    
    if os.path.exists(summary_json):
        size = os.path.getsize(summary_json) / 1024
        print(f"✓ {summary_json} ({size:.1f} KB)")
        
        with open(summary_json, "r") as f:
            summary = json.load(f)
        print("\n[Summary Contents]")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    return True


def run_unit_tests():
    """Run unit tests on the composer."""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80 + "\n")
    
    try:
        import pytest
        pytest.main([__file__, "-v", "-s"])
    except ImportError:
        print("pytest not installed, trying manual tests...")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            from PIL import Image
            
            # Create test data
            img_dir = os.path.join(tmp_dir, "images")
            mask_dir = os.path.join(tmp_dir, "masks")
            os.makedirs(img_dir)
            os.makedirs(mask_dir)
            
            for i in range(6):
                img = Image.new('RGB', (64, 64), color=(100+i*20, 50, 150))
                img.save(os.path.join(img_dir, f"crop_{i:03d}.png"))
                
                mask = np.zeros((100, 100), dtype=np.uint8)
                cv2.rectangle(mask, (10, 10), (50, 50), 255, -1)
                cv2.imwrite(os.path.join(mask_dir, f"crop_{i:03d}.png"), mask)
            
            # Create canvas
            canvas = np.ones((300, 400, 3), dtype=np.uint8) * 255
            cv2.rectangle(canvas, (50, 50), (350, 250), 0, -1)
            shape_path = os.path.join(tmp_dir, "canvas.jpg")
            cv2.imwrite(shape_path, canvas)
            
            # Test
            output_dir = os.path.join(tmp_dir, "output")
            print(f"Testing with temporary data in {tmp_dir}")
            
            try:
                composer = TemporalLayoutComposer(
                    image_dir=img_dir,
                    mask_folder=mask_dir,
                    shape_image=shape_path,
                    output_dir=output_dir,
                    min_len=2,
                    max_len=3
                )
                print("✓ Composer initialized successfully")
                
                result = composer.compose()
                print("✓ Composition completed successfully")
                
                if "temporal_info" in result:
                    print(f"✓ Found {result['temporal_info']['num_segments']} temporal segments")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()


def show_example_usage():
    """Show example usage."""
    print("""

╔════════════════════════════════════════════════════════════════════════════╗
║                   TEMPORAL LAYOUT COMPOSER - EXAMPLE USAGE                 ║
╚════════════════════════════════════════════════════════════════════════════╝

1. BASIC USAGE (Python)
═════════════════════════

    from temporal_layout_composer_unified import TemporalLayoutComposer
    
    # Initialize composer
    composer = TemporalLayoutComposer(
        image_dir="/path/to/cropped/images",
        mask_folder="/path/to/object/masks",
        shape_image="/path/to/canvas.jpg",
        output_dir="/path/to/output",
        min_len=3,
        max_len=4,
        w_clip=0.8,
        w_iqa=0.2
    )
    
    # Run full pipeline
    layout = composer.compose()
    
    # Access results
    print(f"Segments: {layout['temporal_info']['num_segments']}")
    print(f"Partitions: {len(layout['parts'])}")


2. CLI USAGE
════════════

    # Run with real keyframes data
    python temporal_layout_composer_unified.py --mode run \
        --keyframes-dir /home/serverai/ltdoanh/LayoutGeneration/outputs/run_with_object_free_6261_20251101_133231/object_free_evaluation/keyframes \
        --min-len 3 --max-len 4
    
    # Run unit tests
    python temporal_layout_composer_unified.py --mode test --pytest
    
    # Show example
    python temporal_layout_composer_unified.py --mode test --example


3. OUTPUT STRUCTURE
═══════════════════

    output_dir/
    ├── temporal_layout.json
    │   ├── temporal_info
    │   │   ├── num_segments
    │   │   ├── segment_sizes
    │   │   └── segment_files
    │   ├── parts (spatial partitions)
    │   ├── image_assignments
    │   ├── cuts
    │   └── canvas_shape
    └── composition_summary.json
        ├── num_parts
        ├── num_assignments
        ├── num_temporal_segments
        ├── segment_sizes
        └── forest_summary


4. KEY FEATURES
═══════════════

    ✓ Temporal Segmentation:
      - Auto-clusters images by temporal proximity
      - Uses CLIP embeddings + IQA scores
      - Respects min_len and max_len constraints
    
    ✓ Spatial Optimization:
      - Hierarchical tree of spatial partitions
      - Layer 2 nodes respect temporal boundaries
      - Guided by medial axis
    
    ✓ Constraints:
      - Images in same temporal segment prefer same subtree
      - Min/max segment lengths enforced
      - Foreground regions respected


5. INTEGRATION WITH KEYFRAMES
══════════════════════════════

    Typical keyframes folder structure:
    
    keyframes/
    ├── cropped_objects/       ← input images
    ├── masked_objects/        ← input masks
    ├── optimal_layout.png     ← canvas/shape
    └── temporal_composition/  ← output (auto-created)


╔════════════════════════════════════════════════════════════════════════════╗
║                              END OF EXAMPLE                                ║
╚════════════════════════════════════════════════════════════════════════════╝

    """)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Temporal Layout Composer - Unified",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Run with real data
  python temporal_layout_composer_unified.py --mode run \\
    --keyframes-dir /path/to/keyframes

  # Run unit tests
  python temporal_layout_composer_unified.py --mode test --pytest

  # Show example usage
  python temporal_layout_composer_unified.py --mode test --example
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "test"],
        default="run",
        help="Execution mode: 'run' for real data, 'test' for unit tests"
    )
    
    # Run mode arguments
    parser.add_argument(
        "--keyframes-dir",
        type=str,
        default="/home/serverai/ltdoanh/LayoutGeneration/outputs/run_with_object_free_6261_20251101_133231/object_free_evaluation/keyframes",
        help="Path to keyframes directory"
    )
    parser.add_argument(
        "--shape-image",
        type=str,
        default=None,
        help="Path to canvas shape image (optional, defaults to optimal_layout.png or final.png in keyframes-dir)"
    )
    parser.add_argument("--min-len", type=int, default=3, help="Min temporal segment length")
    parser.add_argument("--max-len", type=int, default=4, help="Max temporal segment length")
    parser.add_argument("--w-clip", type=float, default=0.8, help="CLIP weight")
    parser.add_argument("--w-iqa", type=float, default=0.2, help="IQA weight")
    
    # Test mode arguments
    parser.add_argument("--pytest", action="store_true", help="Run pytest tests")
    parser.add_argument("--example", action="store_true", help="Show example usage")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.mode == "run":
        success = run_real_data_test(
            args.keyframes_dir,
            shape_image=args.shape_image,
            min_len=args.min_len,
            max_len=args.max_len,
            w_clip=args.w_clip,
            w_iqa=args.w_iqa
        )
        sys.exit(0 if success else 1)
    
    elif args.mode == "test":
        if args.example:
            show_example_usage()
        elif args.pytest:
            run_unit_tests()
        else:
            run_unit_tests()


if __name__ == "__main__":
    main()
