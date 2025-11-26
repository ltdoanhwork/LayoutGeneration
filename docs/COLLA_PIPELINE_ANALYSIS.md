# Colla Layout Pipeline - Chi Tiết Từng Bước

## 📋 OVERVIEW

```
Input RGB Image
      ↓
[STEP 0] U2-Net Mask Generation → shape_mask_refined.png
      ↓
[STEP 1] Shape Decomposition → Cuts + Medial Axis
      ↓
[STEP 2] SAS Spatial Optimization → Forest + Assignments
      ↓
[STEP 3] Content-Aware Warping + Rendering → Final Collage
```

---

## 🔷 STEP 0: U2-Net Mask Generation from Canvas Image

### Input
- **File**: `input_shape` (RGB image)
- **Example**: `/repos/Colla/input_data/image_collections/cars/01.jpg`
- **Format**: Color image (H×W×3)
- **Purpose**: Define canvas shape for layout

### Process

#### 0a. Image Loading
```python
image = cv2.imread(input_shape)  # Load RGB image
# Output: numpy array (H, W, 3), dtype=uint8, range [0-255]
```

#### 0b. Preprocessing for U2-Net
```python
inputs, orig_h, orig_w = preprocess_image(image)
```

**What it does**:
- Resize image to fit U2-Net input size (usually 320×320)
- Convert BGR to RGB (if needed)
- Normalize to [0, 1] or [-1, 1] depending on model
- Save original dimensions for later upsampling

**Output**:
- `inputs`: (1, 320, 320, 3) normalized tensor
- `orig_h, orig_w`: Original dimensions

#### 0c. U2-Net Saliency Prediction
```python
pred_mask = predict_mask(net, inputs)
# From collage_assembly.py: compute_u2net_saliency(image)
```

**U2-Net Algorithm** (Qin et al. 2020):
- **Architecture**: U-Net with Residual blocks (ReSide)
- **Input**: 320×320 RGB image
- **Output**: Single-channel saliency map (0-1)
- **How it works**:
  1. Encoder: 6 levels of downsampling with skip connections
  2. Decoder: Upsampling with feature fusion
  3. Residual connections at each level
  4. Multi-scale output fusion
  5. Final sigmoid → probability map [0, 1]

**Why U2-Net?**
- ✓ Superior saliency detection (salient objects = white, background = black)
- ✓ Works on arbitrary image shapes
- ✓ Fast inference
- ✗ Doesn't understand layout semantics, just visual prominence

**Output**:
- `pred_mask`: (orig_h, orig_w), dtype=float32, range [0, 1]
- High values (close to 1) = salient regions
- Low values (close to 0) = background

#### 0d. Mask Refinement
```python
mask_refined = refine_mask(pred_mask, orig_h, orig_w)
```

**What refine_mask() does**:
1. **Thresholding**: `pred_mask > threshold` (usually 0.5)
   - Creates binary mask: 0 or 255
2. **Morphological closing**: Remove small holes
   ```
   kernel = cv2.getStructuringElement(cv2.MORPH_CLOSE, (5,5))
   closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
   ```
3. **Keep largest component**: Extract main object
   ```
   contours = cv2.findContours(closed, ...)
   largest_contour = max(contours, key=cv2.contourArea)
   mask_final = cv2.drawContours(zeros, [largest_contour], ...)
   ```
4. **Edge smoothing**: Apply Gaussian blur
   ```
   smooth = cv2.GaussianBlur(mask_final, (5,5), 0)
   ```

**Output**:
- `mask_refined`: Binary mask (H, W), dtype=uint8, values [0, 255]
- Shape: Same as original image
- White (255) = canvas area
- Black (0) = background

#### 0e. Save Refined Mask
```python
cv2.imwrite(shape_mask_path, mask_refined)
# Output: shape_mask_refined.png
```

### Output Summary
| File | Format | Usage |
|------|--------|-------|
| `shape_mask_refined.png` | Binary (H×W) uint8 [0,255] | Input to Step 1 (Shape Decomposition) |

---

## 🔶 STEP 1: Shape Decomposition

### Input
- **File**: `shape_mask_refined.png` (Binary canvas mask from STEP 0)
- **Format**: (H, W), dtype=uint8, [0, 255]
- **Also uses**: `input_shape` (original RGB image) for reference

### Process
```python
sd.generate_cuts(shape_mask_path, output_dir)
```

### Substeps

#### 1a. Convert Mask to Polygon
```python
# From shape_decomposition.py: generate_canvas_polygon()
polygons = sd.generate_canvas_polygon(canvas_binary_mask)
```

**Algorithm**:
1. **Flip Y-axis**: Convert image coords (Y down) to math coords (Y up)
   ```
   img = cv2.flip(img, 0)  # Vertical flip
   ```

2. **Threshold**: Ensure binary
   ```
   _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
   ```

3. **Find contours**:
   ```
   contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
   ```

4. **Convert contours to Shapely polygons**:
   ```
   for contour in contours:
       polygon = Polygon(contour.reshape(-1, 2))
       polygons.append(polygon)
   ```

**Output**:
- `polygon`: Shapely Polygon object representing canvas boundary
- Coordinates in (x, y) format
- Holes supported (for complex shapes)

#### 1b. Prepare for Medial Axis Extraction
```python
# From shape_decomposition.py: prepare_for_medial_axis()
medial_input = sd.prepare_for_medial_axis(canvas, complement=False)
```

**Process**:
1. Convert to grayscale (if RGB)
2. Apply Gaussian blur: `GaussianBlur(img, (5,5))`
   - Smooths edges, reduces noise
3. Threshold: `img > 127`
   - Creates binary image
4. Invert if needed (complement=True)
5. Output: Boolean array [True, False]

**Why blur before medial axis?**
- Eliminates tiny spikes that would create spurious branches
- Makes medial axis smoother and more interpretable

#### 1c. Extract Medial Axis (Skeleton)
```python
# From shape_decomposition.py: ridge_medial_axis()
ma_int = sd.ridge_medial_axis(medial_input, ridge_threshold=0.39, small_threshold=5)
```

**Algorithm - Distance Transform + Ridge Detection**:

1. **Distance Transform**: `scipy.ndimage.distance_transform_edt(image)`
   - For each pixel, compute distance to nearest boundary
   - Output: Distance map D(x,y) = minimum distance to edge
   - Example:
     ```
     Edge:    100...001
     D map:   100...010
              111...111
              111...111
     ```

2. **Ridge Detection**: Find local maxima in distance map
   - Use Hessian matrix (second derivatives)
   - Eigenvalues of Hessian:
     - λ1, λ2: principal curvatures
     - Ridge: λ2 < threshold (usually negative)
   - This finds "centerlines" of shapes

3. **Cleanup**:
   ```
   detected_ridges = (eigenvalue_2 < -ridge_threshold)  # Usually -0.39
   cleanup = morphology.remove_small_objects(detected_ridges, min_size=5)
   ```

4. **Connect broken lines**:
   ```
   kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
   connected = cv2.dilate(cleanup, kernel, iterations=1)
   ```

**Output**:
- `ma_int`: Binary image showing medial axis
- Skeleton of the canvas shape
- Typically 1-3 pixels wide

**Visualization**:
```
Canvas:        Medial Axis:
████████          ██
████████    →      ██
████████          ██
                   ██
```

#### 1d. Convert Medial Axis to MultiLineString
```python
# From shape_decomposition.py: build_medial_multilinestring()
multilinestring_int = sd.build_medial_multilinestring(ma_int[0])
```

**Algorithm**:
1. **Find connected components** in medial axis
   ```
   labels, num_features = cv2.connectedComponents(ma_int)
   ```

2. **For each component**, trace pixels to form LineString
   ```
   for label_id in range(num_features):
       pixels = np.where(labels == label_id)
       # Sort pixels to form continuous line
       linestring = LineString(sorted_pixels)
   ```

3. **Merge into MultiLineString**
   ```
   multilinestring = MultiLineString(all_linestrings)
   ```

**Output**:
- `multilinestring_int`: Shapely MultiLineString
- Each LineString represents a skeleton branch
- Used for spatial partition guidance in STEP 2

#### 1e. Generate Medial Graph
```python
# From shape_decomposition.py: build_medial_graph()
G = sd.build_medial_graph(multilinestring_int, ...)
```

**Algorithm**:
1. Convert medial axis to graph nodes and edges
2. Nodes = line endpoints, intersections
3. Edges = LineString segments between nodes
4. Remove small branches < 8 pixels
5. Output: NetworkX Graph

#### 1f. Find Projection Pairs (Boundary to Medial Axis)
```python
# From shape_decomposition.py: find_projection_pair()
projection_pairs = sd.extract_projection_pair(G, boundary_vertices)
```

**Purpose**: Find perpendicular distances from boundary to medial axis
- Maps boundary points to closest medial axis points
- Used to guide cutting directions

#### 1g. Generate Raw Cuts
```python
# From shape_decomposition.py: generate_raw_cuts()
raw_cuts = sd.generate_raw_cuts(projection_pairs, corner_dict)
```

**Algorithm**:
1. For each boundary-to-medial projection:
   - Direction = perpendicular to medial axis
   - Extend cut across canvas
2. Generate LineString for each cut

#### 1h. Select Representative Cuts
```python
# From shape_decomposition.py: select_representative_cuts()
final_cuts = sd.select_representative_cuts(raw_cuts, ...)
```

**Heuristics**:
- Remove redundant/overlapping cuts
- Keep cuts with:
  - Balanced distance from center
  - High "extension strength" (covers boundary)
  - Low "protrusion" (avoids re-entrant shapes)
- Sort by quality score

### Output Summary
| File | Content | Usage |
|------|---------|-------|
| `final_cut.json` | List of cut lines + metadata | Input to STEP 2 |
| `medial_axis.json` | Medial axis geometry | Input to STEP 2 |
| Various .png | Debug visualizations | Analysis only |

---

## 🔷 STEP 2: Spatial Assignment Optimization (SAS)

### Input
- **File 1**: `shape_mask_refined.png` (Binary canvas)
- **Folder**: `input_mask_folder` (Object masks, one per cropped object)
  - Example: `crop_001.png, crop_002.png, ...`
  - Format: Each = binary mask or U2-Net saliency output
- **From STEP 1**: `final_cut.json`, `medial_axis.json`

### Process
```python
so.optimization(shape_mask_path, input_mask_folder, output_dir)
```

### Substeps

#### 2a. Load Canvas Polygon
```python
# From sas_optimization.py: optimization()
polygon = sd.generate_canvas_polygon(canvas_mask)[0]
```

#### 2b. Apply Cuts to Create Partitions
```python
# From sas_optimization.py: Partition class
partition = Partition(polygon)
for cut in final_cuts:
    partition.add_cut(cut)
```

**Algorithm - Binary Space Partition (BSP)**:
1. Start with full canvas polygon
2. For each cut (LineString):
   ```
   parts_left = polygon.intersection(left_half_plane)
   parts_right = polygon.intersection(right_half_plane)
   ```
3. Recursively apply cuts
4. Result: Tree of convex/nearly-convex regions

**Output**: `convex_parts = partition.list_leaves()`
- List of Shapely Polygon objects
- Each represents a spatial region
- Typically 5-20 regions depending on canvas complexity

#### 2c. Load Object Masks and Extract Foreground
```python
# From sas_optimization.py: optimization()
for image_id in os.listdir(input_mask_folder):
    mask_path = os.path.join(input_mask_folder, f"{image_id}.png")
    label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    x1, x2, y1, y2, exists = extract_foreground(label)
    # Store bounding box and aspect ratio
    item["foreground"] = [x1, x2, y1, y2]
    item["aspect_ratio"] = (x2 - x1) / (y2 - y1)
```

**extract_foreground() implementation**:
```python
def extract_foreground(label):
    # Find non-zero pixels (foreground)
    foreground = (label > 0).astype(int)  # Binary mask
    area = np.sum(foreground)
    total_area = label.shape[0] * label.shape[1]
    
    # Threshold check: foreground must be > 0.5% of image
    if area > total_area / 200:
        rows, cols = np.where(foreground)
        x1, x2 = int(np.min(cols)), int(np.max(cols))
        y1, y2 = int(np.min(rows)), int(np.max(rows))
        foreground_exist = True
    else:
        # Default: 80% of image
        x1 = int(label.shape[1] * 0.1)
        x2 = int(label.shape[1] * 0.9)
        y1 = int(label.shape[0] * 0.1)
        y2 = int(label.shape[0] * 0.9)
        foreground_exist = False
    
    return x1, x2, y1, y2, foreground_exist
```

**Output**: `image_dict`
```python
[
    {
        "filename": "crop_001.png",
        "foreground": [100, 400, 50, 300],  # [x1, x2, y1, y2]
        "aspect_ratio": 1.2,  # width / height
        "assigned_part": -1
    },
    ...
]
```

#### 2d. Initialize Slicing Forest
```python
# From sas_optimization.py: forest_initialization()
forest = []
for part in convex_parts:
    # Determine how many images should go in this part
    num_images_for_part = estimate_images(part.area, total_area, num_images)
    
    # Create slicing tree with N leaves
    root = tree_initialization(num_images_for_part, balanced=True)
    root.polygon = part
    forest.append(root)
```

**Tree Initialization Algorithm**:
1. Create binary tree with `N` leaves
2. Uses balanced strategy:
   - Always insert next node into shallower subtree
   - Prevents degenerate trees
3. Result: Roughly balanced tree

**Data Structure - TreeNode**:
```python
class TreeNode:
    polygon = None           # Spatial region (Shapely Polygon)
    type = "A" or "C"       # Axial or Crosswise cut direction
    configuration = 0-3      # Which cut to use (0-1: axial, 2-3: crosswise)
    left_child = None        # Image 0
    right_child = None       # Image 1
    assignment = {
        "id": image_id,
        "aspect_ratio": w/h,
        "coord": [x, y, w, h]
    }
    cut = []                 # Cut line used (if not leaf)
```

#### 2e. Heuristic Initialization
```python
# From sas_optimization.py: heuristic_initialization()
heuristic_initialization(tree, medial_axis, depth=tree.height() - 2)
```

**Algorithm**:
```python
def heuristic_initialization(node, medial_axis, depth):
    if depth == 0 or node.is_leaf():
        return
    
    # Get medial axis direction at this node's centroid
    tangent = medial_axis_tangent(medial_axis, node.centroid())
    
    # Choose cut direction based on aspect ratio
    size_axial = node.get_size(tangent)         # Along medial axis
    size_crosswise = node.get_size(perpendicular)  # Across medial axis
    
    if size_axial > size_crosswise:
        node.type = "A"  # Axial cut
    else:
        node.type = "C"  # Crosswise cut
    
    node.configuration = random.randint(0, 1)  # Random order
    
    # Recurse to children
    heuristic_initialization(node.left_child, medial_axis, depth - 1)
    heuristic_initialization(node.right_child, medial_axis, depth - 1)
```

**Purpose**: Reduce search space for optimization
- If shape is tall, cut horizontally
- If shape is wide, cut vertically

#### 2f. Forest Optimization (get_optimal)
```python
# From sas_optimization.py: forest_optimization()
solution = []
for tree in forest:
    score, optimized_tree = get_optimal(tree, medial_axis)
    solution.append((score, optimized_tree))
```

**get_optimal() Algorithm - Recursive Score Maximization**:

```python
def get_optimal(node, medial_axis):
    if node.is_leaf():
        # Leaf quality score
        quality = cell_quality(node.polygon)
        return 1.0 if quality else 0.5, node
    
    # Try 4 configurations
    results = []
    
    for config in [0, 1, 2, 3]:
        # 0,1: Axial cuts (different orders)
        # 2,3: Crosswise cuts (different orders)
        
        # Generate cut perpendicular to direction
        cut = generate_cut(node, config, medial_axis)
        
        try:
            # Split polygon by cut
            splits = list(ops.split(node.polygon, cut))
            if len(splits) < 2:
                continue
            
            # Assign to children
            left, right = splits[0], splits[1]
            node.left_child.polygon = left
            node.right_child.polygon = right
            
            # Recursively optimize children
            score_left, _ = get_optimal(node.left_child, medial_axis)
            score_right, _ = get_optimal(node.right_child, medial_axis)
            
            total_score = score_left + score_right
            results.append((total_score, config, splits))
        
        except:
            continue
    
    # Pick best configuration
    if results:
        best_score, best_config, best_splits = max(results, key=lambda x: x[0])
        node.configuration = best_config
        node.left_child.polygon = best_splits[0]
        node.right_child.polygon = best_splits[1]
        
        # Recurse to get final scores
        score_l, _ = get_optimal(node.left_child, medial_axis)
        score_r, _ = get_optimal(node.right_child, medial_axis)
        return score_l + score_r, node
    else:
        return 0.0, node
```

**Complexity**: O(4^h) where h = tree height
- Reasonable for h ≤ 4 (16 leaves max)
- Can be slow for larger trees

**Scoring - cell_quality()**:
```python
def cell_quality(polygon):
    # Penalize if:
    # 1. Too few sides (< 4)
    # 2. Has sharp angles (< 35°)
    # Returns True if "good" partition
```

#### 2g. Image Assignment to Forest
```python
# From sas_optimization.py: calculate_image_assignment()
assignment = calculate_image_assignment(images, leaf_elevation_summary)
```

**Algorithm**:
1. Get depth of each leaf: `leaf_elevation_summary(tree)`
   ```
   {3: 2,   # 2 leaves at depth 3
    4: 4}   # 4 leaves at depth 4
   ```

2. Sort images by priority (aspect ratio works too)
3. Assign sequentially from shallowest to deepest leaves
   - Higher priority images → shallower leaves (larger)
   - Lower priority images → deeper leaves (smaller)

### Output Summary
| File | Content | Usage |
|------|---------|-------|
| `final_assignment.json` | Image-to-part assignments | Input to STEP 3 |
| `forest_structure.json` | Tree cuts and topology | Input to STEP 3 |

---

## 🟡 STEP 3: Content-Aware Warping + Rendering

### Input
- Folder: `input_image_collection_folder` (Cropped objects)
  - Example: `crop_001.png, crop_002.png, ...`
  - Format: RGBA or RGB (with/without transparency)
- From STEP 2:
  - `final_assignment.json` (which image → which part)
  - `forest_structure.json` (partition geometry)
- `scaling_factor`: Output upscaling (default=2)

### Process
```python
ca.render_collage(input_image_collection_folder, output_dir, 
                  scaling_factor, enable_debug=enable_debug)
```

### Substeps

#### 3a. Load Canvas + Assignment Info
```python
# From collage_assembly.py: render_collage()
# Load canvas shape
canvas_shape = ...  # (W, H) from earlier
# Load assignments
assignments = json.load("final_assignment.json")
```

#### 3b. For Each Image → Partition

```python
# From collage_assembly.py: generate_image_patch()
for part_idx, image_dict in assignments.items():
    part_polygon = parts[part_idx]
    image_path = os.path.join(input_image_collection_folder, image_dict["filename"])
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Generate placement + warp
    warped_patch = generate_image_patch(part_polygon, image_dict, image, ...)
    
    # Composite onto canvas
    canvas = image_overlay(canvas, warped_patch, origin=(x, y))
```

#### 3c. Content-Aware Mesh Warping (KEY ALGORITHM)

**Purpose**: Deform image to fit partition while preserving salient content

##### 3c-i. Compute U2-Net Saliency Map
```python
# From collage_assembly.py: compute_u2net_saliency()
saliency = compute_u2net_saliency(image)
```

**Output**: Saliency map S(x,y) ∈ [0, 1]
- High values: important content
- Low values: background

##### 3c-ii. Create Dense Mesh Grid
```python
# From collage_assembly.py: create_dense_mesh()
src_pts, src_triangles = create_dense_mesh(img_height, img_width, grid_size=16)
```

**Algorithm**:
1. Create uniform grid: 16×16 control points
   ```
   (0,0)   (w/16,0) ... (w,0)
   (0,h/16) ...
   ...
   (0,h)   ...       (w,h)
   ```
2. Vertices: (w/16*i, h/16*j) for i,j ∈ [0,16]
3. Generate triangles: Each quad → 2 triangles
4. Output: 
   - `src_pts`: Source grid points (18×18 = 324 control points)
   - `src_triangles`: 512 triangles

##### 3c-iii. Compute Destination Grid (Target Partition)
```python
# Map source grid to target partition coordinates
outer_rect = partition_bounding_box
inner_rect = foreground_bounding_box

# Create uniform grid in destination space
dst_pts = create_destination_mesh(outer_rect, inner_rect, grid_size=16)
```

##### 3c-iv. Optimize Mesh Warp (CORE ALGORITHM)

```python
# From collage_assembly.py: optimize_mesh_warp()
dst_pts_optimized = optimize_mesh_warp(
    src_pts, dst_pts_init,
    saliency_map, inner_rect, outer_rect,
    salient_weight=350.0,
    background_weight=0.4,
    smooth_lambda=0.8
)
```

**Algorithm - Energy Minimization**:

Minimize:
```
E(p) = E_salient(p) + E_background(p) + E_smooth(p)
```

Where:
- `p`: Destination mesh points (variables)
- `E_salient`: Preserve salient regions
- `E_background`: Compress background
- `E_smooth`: Smooth deformation

**Implementation**:

```python
def optimize_mesh_warp(src_pts, dst_pts_init, saliency,
                       inner_rect, outer_rect,
                       salient_weight=350, background_weight=0.4,
                       smooth_lambda=0.8):
    
    def energy(dst_pts_flat):
        dst_pts = dst_pts_flat.reshape(-1, 2)
        E = 0
        
        # 1. Salient Content Preservation
        # For salient regions, try to keep original coordinates
        for (x, y) in dst_pts:
            if saliency[int(y), int(x)] > threshold:
                # Distance from original position
                original_pos = ...
                E += salient_weight * ||dst_pts - src_pts||^2
        
        # 2. Background Compression
        # For background, allow more deformation (compress)
        for (x, y) in dst_pts:
            if saliency[int(y), int(x)] < threshold:
                # Penalty for staying in place
                E += background_weight * ||dst_pts - src_pts||^2
        
        # 3. Smoothness (Thin Plate Spline energy)
        # Penalize mesh distortion
        for triangle in triangles:
            p1, p2, p3 = triangle_vertices
            # Local area should not change too much
            original_area = compute_area(src_pts[triangle])
            warped_area = compute_area(dst_pts[triangle])
            E += smooth_lambda * (warped_area - original_area)^2
        
        return E
    
    # Minimize using L-BFGS-B
    from scipy.optimize import minimize
    result = minimize(energy, dst_pts_init.flatten(),
                     method='L-BFGS-B',
                     bounds=destination_bounds)
    
    return result.x.reshape(-1, 2)
```

**Optimization Details**:
- **Method**: L-BFGS-B (limited-memory BFGS with box constraints)
- **Iterations**: ~100-200
- **Convergence**: When gradient < 1e-5
- **Constraints**: Points stay within partition boundary

**Hyperparameters Analysis**:
| Parameter | Value | Effect |
|-----------|-------|--------|
| `salient_weight` | 350 | Strong preservation of salient objects |
| `background_weight` | 0.4 | Allow background to compress more |
| `smooth_lambda` | 0.8 | Moderate mesh smoothness |

**Why these values?**
- High `salient_weight`: Objects are important, don't distort
- Low `background_weight`: Background flexible, can compress
- Medium `smooth_lambda`: Some distortion allowed but not extreme

##### 3c-v. Apply Mesh Warp using RBF or Affine Transform
```python
# From collage_assembly.py: apply_mesh_warp_tps()
warped_image = apply_mesh_warp_tps(image, src_pts, dst_pts_optimized, 
                                   partition_shape)
```

**Algorithm - Triangle-by-Triangle Affine Warp**:

```python
def apply_mesh_warp_remap(image, src_pts, dst_pts, target_shape):
    # For each triangle:
    warped_canvas = zeros(target_shape)
    
    for idx, src_triangle in enumerate(src_triangles):
        dst_triangle = dst_pts[src_triangle]  # Corresponding dest triangle
        
        # Compute affine transform
        # Maps src_triangle → dst_triangle
        warp_mat = cv2.getAffineTransform(
            np.float32(src_triangle[:3]),
            np.float32(dst_triangle[:3])
        )
        
        # Warp this triangle
        warped_region = cv2.warpAffine(image, warp_mat, target_shape)
        
        # Create mask for destination triangle
        mask = zeros(target_shape, dtype=uint8)
        cv2.drawContours(mask, [dst_triangle], 0, 255, -1)
        
        # Composite onto canvas
        warped_canvas += cv2.bitwise_and(warped_region, warped_region, mask=mask)
    
    return warped_canvas
```

**Or using TPS (Thin Plate Spline)**:

```python
def apply_mesh_warp_tps(image, src_pts, dst_pts, target_shape):
    from scipy.interpolate import RBFInterpolator
    
    # Create RBF interpolators for X and Y coordinates
    rbf_x = RBFInterpolator(src_pts, dst_pts[:, 0], kernel='thin_plate_spline')
    rbf_y = RBFInterpolator(src_pts, dst_pts[:, 1], kernel='thin_plate_spline')
    
    # Create destination coordinate grid
    yy, xx = np.mgrid[0:target_shape[0], 0:target_shape[1]]
    
    # For each destination pixel, find source coordinate
    src_x = rbf_x(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    src_y = rbf_y(np.column_stack([xx.ravel(), yy.ravel()])).reshape(yy.shape)
    
    # Remap image: dst[y,x] = src[src_y[y,x], src_x[y,x]]
    warped = cv2.remap(image, src_x.astype(np.float32), src_y.astype(np.float32),
                      cv2.INTER_CUBIC)
    
    return warped
```

**TPS Advantage**:
- Smooth, continuous deformation
- Preserves local structure
- No triangle-by-triangle seams
- Slower but higher quality

##### 3c-vi. Alpha Blending
```python
# From collage_assembly.py: image_overlay()
# Create alpha mask from original image
alpha = image[:, :, 3]  # Extract alpha channel
alpha_normalized = alpha / 255.0

# Composite warped image onto canvas
canvas[region] = warped_image * alpha_normalized[region] \
                + canvas[region] * (1 - alpha_normalized[region])
```

#### 3d. Scale Output
```python
# From collage_assembly.py: render_collage()
if scaling_factor > 1:
    final_w = canvas.shape[1] * scaling_factor
    final_h = canvas.shape[0] * scaling_factor
    canvas_scaled = cv2.resize(canvas, (final_w, final_h), cv2.INTER_CUBIC)
```

#### 3e. Save Final Collage
```python
cv2.imwrite(os.path.join(output_dir, "final_collage.png"), canvas_scaled)
```

### Output Summary
| File | Format | Description |
|------|--------|-------------|
| `final_collage.png` | RGBA (W×scaling×H×scaling) | Final composite image |
| `warp_debug_visualization/` | PNGs | Saliency, mesh, results (if --debug) |

---

## ✅ OVERALL QUALITY ASSESSMENT

### Strengths

| Component | Assessment | Why |
|-----------|-----------|-----|
| **U2-Net Saliency** | ✓ Excellent | State-of-art object detection |
| **Medial Axis Extraction** | ✓ Good | Robust skeleton-based decomposition |
| **Binary Space Partition** | ✓ Good | Clean, structured regions |
| **Tree-based Optimization** | ✓ Good | Efficient O(4^h) search |
| **Mesh Warping** | ✓ Excellent | Content-aware deformation |
| **TPS Interpolation** | ✓ Excellent | Smooth, artifact-free warping |

### Weaknesses

| Component | Issue | Impact |
|-----------|-------|--------|
| **Fixed grid size (16×16)** | May be coarse for large images | Visible mesh artifacts in high-res |
| **Saliency weight tuning** | Hyperparameters are fixed | May not adapt to different content types |
| **No semantic awareness** | Pure visual saliency, no objects | Could compress important text/details |
| **Single canvas input** | Requires pre-defined shape | Not suitable for free-form layouts |
| **No temporal awareness** | Each image processed independently | No coherence between adjacent objects |

### Algorithm Ranking by Quality

```
1. TPS Mesh Warping         ★★★★★  (Excellent - smooth, artifact-free)
2. U2-Net Saliency          ★★★★★  (Excellent - state-of-art)
3. Tree Optimization        ★★★★   (Good - efficient, reasonable quality)
4. Medial Axis              ★★★★   (Good - reliable skeleton extraction)
5. Binary Space Partition   ★★★    (Fair - rigid, may over-partition)
6. Hyperparameter Tuning    ★★     (Poor - fixed for all cases)
```

---

## 🔄 Data Flow Diagram

```
Input RGB Canvas
    ↓
[STEP 0] U2-Net Saliency Detection
    ↓ shape_mask_refined.png
[STEP 1] Medial Axis + Shape Decomposition
    ↓ final_cut.json, medial_axis.json
    │
    ├─→ Binary Space Partition
    │   ↓ convex_parts (5-20 regions)
    │
[STEP 2] SAS Forest Optimization
    ↓ forest_structure.json, final_assignment.json
    │
    ├─→ Tree-based Layout (4^h search)
    │   ↓ Optimized cuts
    │
Input Images (Cropped Objects)
    ↓
[STEP 3] Content-Aware Warping
    ├─→ U2-Net Saliency per image
    ├─→ Mesh Grid (16×16)
    ├─→ Energy Minimization (L-BFGS-B)
    ├─→ TPS Warping
    ├─→ Compositing
    │
    ↓
Final Collage (Scaled)
```

---

## 📊 Complexity Analysis

| Step | Time Complexity | Space | Notes |
|------|---|---|---|
| U2-Net | O(W×H) | O(W×H) | Forward pass only, ~50ms for 640×480 |
| Medial Axis | O(W×H) | O(W×H) | Distance transform + ridge detection |
| BSP Cuts | O(C × P²) | O(C × P) | C=cuts, P=partition complexity |
| Tree Optimization | O(4^h) | O(4^h) | h=tree height, pruning helps |
| Mesh Warping | O(K × M) | O(K × M) | K=iterations, M=mesh points |
| **Total** | **O(W×H + 4^h + K×M)** | **O(W×H)** | **Dominated by high-res processing** |

---

## 🎯 Conclusion

The Colla pipeline is a **well-engineered content-aware collage generation system**:

✓ **Strengths**:
- Uses U2-Net saliency for intelligent object preservation
- Medial axis guides layout structures naturally
- Mesh warping with energy minimization preserves content
- Modular, tested implementation

✗ **Limitations**:
- Requires pre-defined canvas shape (not generative)
- Fixed hyperparameters (not adaptive)
- No temporal coherence (each image independent)
- Computational cost O(4^tree_height) can be high

**Best Use Case**: Multi-object composition with fixed canvas shape and importance on content preservation.

**For Sakuga Animation**: Could work well for static frame layout, but temporal segmentation should influence tree structure (current implementation ignores it).
