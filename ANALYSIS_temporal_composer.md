# Phân Tích Chi Tiết: temporal_layout_composer_unified.py

## 1️⃣ QUY TRÌNH CHÍNH (compose())

### Bước 1: Run Temporal Segmentation
```python
self.segments, self.segment_files = self.run_temporal_segmentation()
```

**Định nghĩa**: `run_temporal_segmentation()` dòng ~1500
```python
def run_temporal_segmentation(self) -> Tuple[List[List[int]], List[List[str]]]:
    segmenter = AutoSegmenter(w_clip=0.8, w_iqa=0.2, min_len=3, max_len=4, mode="auto")
    segments_idx, segments_files = segmenter.segment(self.image_dir)
    return segments_idx, segments_files
```

**❌ PROBLEM 1**: Chỉ nhìn vào folder `self.image_dir = cropped_objects/`
- Không dùng thông tin từ `masked_objects/` (các mask)
- **Kết quả**: Temporal segmentation purely based on CLIP+IQA, không biết object shapes

**Ý tưởng sai**: Temporal segments nên bao gồm thông tin về objects trong từng segment

---

### Bước 2: Load Canvas và Masks
```python
polygon, image_dict, image_ids = self.load_shape_and_masks()
```

**Định nghĩa**: `load_shape_and_masks()` dòng ~1555
```python
def load_shape_and_masks(self) -> Tuple[Polygon, List[Dict], List[str]]:
    # Load canvas
    canvas = cv2.imread(self.shape_image, cv2.IMREAD_GRAYSCALE)
    polygon = sd.generate_canvas_polygon(canvas)[0]  # ← GLOBAL canvas polygon
    
    # Load ALL masks từ masked_objects/ folder
    for image_id in image_ids:
        mask_path = os.path.join(self.mask_folder, f"{image_id}.png")
        label = load_mask(mask_path)
        x1, x2, y1, y2, foreground_exists = extract_foreground(label)
        # Store foreground bounding box
        item["foreground"] = [x1, x2, y1, y2]
```

**Vấn đề**:
1. Loads `mask_id.png` từ `masked_objects/` 
2. **Nhưng**: Mask này là pixel-level object mask (từ Cartoon Detector)
3. `extract_foreground()` chỉ tính bounding box từ foreground pixels
4. Không phân biệt được object nào từ ảnh nào trong temporal segment

**❌ PROBLEM 2**: `image_id` từ folder name = `crop_001.png`, nhưng:
- Temporal segment files = `["crop_001.png", "crop_002.png", ...]` 
- Mask file = `crop_001.png` (tất cả objects merged?)
- **Không rõ**: 1 mask file có bao nhiêu objects? Objects từ ảnh nào?

---

### Bước 3: Compute Priority (CLIP + IQA)
```python
prioritized_images = self._compute_image_priorities(image_dict)
```

**Định nghĩa**: `_compute_image_priorities()` dòng ~1640
```python
def _compute_image_priorities(self, image_dict: List[Dict]) -> List[Dict]:
    for item in image_dict:
        image_path = self._resolve_image_path(image_id)
        clip_vec = get_clip_embedding(pil_rgb)  # ← Lấy từ cropped_objects/
        iqa_score = get_iqa_score(pil_rgb)
        priority = (w_clip * clip_norm + w_iqa * iqa_norm) / weight_sum
        item["priority_score"] = priority
    
    resolved_items.sort(key=lambda d: d.get("priority_score"), reverse=True)
    for rank, item in enumerate(resolved_items):
        item["priority_rank"] = rank
```

**Vấn đề**:
- Priority score dựa trên **individual image quality**, không temporal context
- Images từ cùng temporal segment có thể bị scattered vào different partitions
- **❌ PROBLEM 3**: Ignores temporal constraints hoàn toàn

---

### Bước 4: Build Temporal-Constrained Forest
```python
forest, forest_summary = self.build_temporal_constrained_forest(
    polygon, convex_parts, multilinestring_int, self.layout_segments
)
```

**Định nghĩa**: `build_temporal_constrained_forest()` dòng ~1752
```python
def build_temporal_constrained_forest(self, polygon, convex_parts, 
                                      multilinestring_int, image_indices_per_segment):
    num_segments = len(image_indices_per_segment)  # ← This is [0,1,2,...,N-1] (ALL images!)
    
    segment_polygons = self._create_partitions_from_medial_axis(
        polygon, multilinestring_int[0], image_indices_per_segment
    )
    
    for seg_idx in range(num_segments):
        images_for_segment = len(image_indices_per_segment[seg_idx])
        root = tree_initialization(images_for_segment, balanced=True)
        root.polygon = seg_polygon
        forest.append(root)
```

**Issue**: 
```python
self.layout_segments = [list(range(len(prioritized_images)))]  # ← SINGLE segment with ALL images!
```

**❌ PROBLEM 4 (CRITICAL)**: 
- `self.layout_segments` được set lại thành `[0,1,2,...,N-1]` (tất cả images = 1 segment)
- **Đó là WHY**: Hàm không dùng `self.segments` from temporal segmentation!
- **Result**: Ignores all temporal constraints từ `run_temporal_segmentation()`

---

## 2️⃣ DETAILED ANALYSIS: KEYFRAME LOADING

### Actual folder structure:
```
keyframes/
├── cropped_objects/          ← Individual object images (từ Cartoon Detector)
│   ├── crop_001.png
│   ├── crop_002.png
│   └── ...
├── masked_objects/           ← Per-object masks (từ U2-Net Saliency)
│   ├── crop_001.png
│   ├── crop_002.png
│   └── ...
└── optimal_layout.png        ← Canvas shape
```

### How objects get here:
1. **Cartoon Detector** runs on each keyframe
2. For keyframe `keyframe_N.jpg`:
   - Detects all objects (e.g., 3 characters)
   - Exports as `crop_001.png, crop_002.png, crop_003.png`
3. **U2-Net Saliency** runs on each cropped object
   - Generates mask_001.png, mask_002.png, mask_003.png

---

## 3️⃣ PROBLEM ANALYSIS: MASK FILTERING

### Current approach (WRONG):
```python
def load_shape_and_masks(self):
    image_ids = sorted([
        f.split(".")[0]  # ← Just the ID like "crop_001"
        for f in os.listdir(self.mask_folder)
        if f.endswith('.png')
    ])
    
    # Load EACH mask file
    for image_id in image_ids:
        mask_path = os.path.join(self.mask_folder, f"{image_id}.png")
        label = load_mask(mask_path)
        x1, x2, y1, y2, _ = extract_foreground(label)
        item["foreground"] = [x1, x2, y1, y2]  # ← SINGLE bounding box
```

**Issues**:
1. ✓ Loads correct mask files
2. ✓ Extracts foreground bounding box
3. ✗ Treats mask as **raw pixel image**, not processed saliency
4. ✗ No thresholding on saliency map
5. ✗ Foreground extraction expects 255 = foreground, but U2-Net outputs grayscale [0-255]

### extract_foreground() implementation (dòng ~410):
```python
def extract_foreground(label):
    total_area = label.shape[0] * label.shape[1]
    foreground = (label == 255).astype(int)  # ← Only exact 255?
    foreground_area = np.sum(foreground)
    
    if foreground_area > total_area / 200:  # ← Threshold check
        x1 = int(np.min(np.where(foreground)[1]))
        x2 = int(np.max(np.where(foreground)[1]))
        y1 = int(label.shape[0] - np.max(np.where(foreground)[0]))
        y2 = int(label.shape[0] - np.min(np.where(foreground)[0]))
    else:
        x1, x2 = int(label.shape[1] / 10), int(label.shape[1] * 9 / 10)
        y1, y2 = int(label.shape[0] / 10), int(label.shape[0] * 9 / 10)
    return x1, x2, y1, y2, foreground_exist
```

**Problems**:
1. ❌ **`label == 255`**: U2-Net output is grayscale [0-255], not binary
   - Should be: `label > threshold` (e.g., > 128)
   - Or: `label > 0` for binary mask
2. ❌ **Y-coordinate flip**: `y1 = label.shape[0] - max(...)` looks wrong
   - Should be: `y1 = min(...), y2 = max(...)`
3. ❌ **Aspect ratio calculation** (dòng ~1593):
   ```python
   if foreground_exists and (x2 - x1) > 0 and (y2 - y1) > 0:
       aspect_ratio = (x2 - x1) / (y2 - y1)  # width / height
   ```
   This is width/height, but usually aspect_ratio = width/height is OK.
   **However**: If `foreground_exist = False`, aspect_ratio = 1.0 (default), which might hide problems.

---

## 4️⃣ LAYOUT ASSEMBLY: IMAGE ASSIGNMENT

### Current approach (dòng ~2070):
```python
def render_layout_with_assigned_images(forest, image_dir, canvas_shape, output_path):
    all_leaves = []
    for tree in forest:
        leaves = _list_leaves(tree)
        all_leaves.extend(leaves)
    
    canvas = np.ones((canvas_shape[1], canvas_shape[0], 3), dtype=np.uint8) * 255
    
    for leaf in all_leaves:
        if not leaf.polygon or not leaf.assignment:
            continue
        
        image_id = leaf.assignment.get("id", -1)
        filename = leaf.assignment.get("filename", "")
        
        # Find image file
        img_path = self._resolve_image_path(filename)  # From cropped_objects/
        img = cv2.imread(img_path)
        
        # Get partition bounds
        coords = list(leaf.polygon.exterior.coords)
        pts = np.array([[int(c[0]), canvas_shape[1] - int(c[1])] for c in coords])
        
        # Draw into canvas
        mask = np.zeros((canvas_shape[1], canvas_shape[0]), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        # Resize and blend
        resized = cv2.resize(img, (w, h))
        partition_region = canvas[y_min:y_max, x_min:x_max]
        # ... blend logic
```

**Issues**:
1. ✗ **Canvas coordinate system**: 
   - Shapely uses mathematical coords (origin bottom-left, Y up)
   - OpenCV uses image coords (origin top-left, Y down)
   - Flip: `canvas_shape[1] - int(c[1])` ← Is this correct?
2. ✗ **Aspect ratio**: Image resized to fit partition bounding box
   - Doesn't preserve aspect ratio from mask
   - Could distort objects
3. ✗ **Saliency NOT used**: Renders entire cropped_objects image
   - Should use masked_objects saliency to guide placement?

---

## 5️⃣ CRITICAL BUGS SUMMARY

| Bug | Location | Severity | Impact |
|-----|----------|----------|--------|
| **Temporal segments ignored** | compose() line 1770 | 🔴 CRITICAL | `self.layout_segments = [range(N)]` overwrites temporal segmentation |
| **Mask loading wrong** | extract_foreground() line 410 | 🔴 CRITICAL | `label == 255` doesn't work with U2-Net grayscale output |
| **Y-coordinate flip** | extract_foreground() line 417 | 🟡 HIGH | Y-axis calculation looks incorrect |
| **Foreground not thresholded** | extract_foreground() line 411 | 🟡 HIGH | Should threshold U2-Net output (> 128 or > 0) |
| **Saliency mask ignored in rendering** | render_layout_with_assigned_images() | 🟡 MEDIUM | Only uses cropped_objects, not masked_objects |
| **Canvas coords confusion** | render_layout_with_assigned_images() line 2090 | 🟡 MEDIUM | Y-flip might be wrong |
| **Aspect ratio not preserved** | render_layout_with_assigned_images() line 2125 | 🟢 LOW | Objects stretched to fit partition |

---

## 6️⃣ HOW TO FIX

### Fix 1: Use actual temporal segments
```python
def compose(self):
    self.segments, self.segment_files = self.run_temporal_segmentation()
    # ✓ DO THIS:
    self.layout_segments = self.segments  # Use real temporal segments!
    # NOT: self.layout_segments = [list(range(len(prioritized_images)))]
```

### Fix 2: Proper mask thresholding
```python
def extract_foreground(label):
    # U2-Net output is grayscale [0-255], threshold at 128
    foreground = (label > 128).astype(int)  # ← FIX: threshold, not == 255
    # OR for binary: foreground = (label > 0).astype(int)
```

### Fix 3: Fix Y-coordinate calculation
```python
def extract_foreground(label):
    if foreground_area > total_area / 200:
        rows, cols = np.where(foreground)
        x1 = int(np.min(cols))
        x2 = int(np.max(cols))
        y1 = int(np.min(rows))      # ← FIX: just min/max
        y2 = int(np.max(rows))      # ← FIX: just min/max
        # NO flip: labels already in image coords
    return x1, x2, y1, y2, foreground_exist
```

### Fix 4: Use saliency in rendering
```python
def render_layout_with_assigned_images(forest, image_dir, mask_dir, canvas_shape, output_path):
    for leaf in all_leaves:
        # Load BOTH image and mask
        img = cv2.imread(os.path.join(image_dir, filename))
        mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
        
        # Apply mask to image (mask out background)
        mask_binary = (mask > 128).astype(np.uint8) * 255
        img_masked = cv2.bitwise_and(img, img, mask=mask_binary)
        
        # Place masked image into partition
```

---

## 7️⃣ EXECUTION FLOW DIAGRAM

```
cropped_objects/          masked_objects/
  crop_001.png      +       crop_001.png
  crop_002.png      +       crop_002.png
       ...                       ...
         |                         |
         v                         v
   [Load Images]            [Load Masks]
   (CLIP embed)             (Extract foreground)
         |                         |
         +----------+--+-----------+
                    |
                    v
          [Compute Priority]
          sort by CLIP+IQA
                    |
         ❌ Ignores temporal segments!
                    |
                    v
          [Build Forest]
          ❌ Treats all N images as 1 segment
                    |
                    v
          [Assign to Leaves]
          ❌ Just by priority, not temporal
                    |
                    v
          [Render Layout]
          Canvas: cropped_objects only
          ❌ Saliency masks not used
                    |
                    v
               layout.png
               ❌ Objects distorted
```

---

## 8️⃣ ROOT CAUSE

The code **pretends to use temporal constraints** but actually:
1. ✓ Computes temporal segments from AutoSegmenter
2. ✗ **Throws them away** with `self.layout_segments = [range(N)]`
3. ✗ Treats all images as one big segment
4. ✗ Sorts by priority and assigns randomly to forest leaves
5. ✗ Renders without saliency information
6. ✗ Doesn't match Colla's original pipeline (which uses pre-computed convex partitions)

The fix requires **3 main changes**:
1. **Use temporal segments in forest building**
2. **Fix mask loading/thresholding**
3. **Use saliency in final rendering**
