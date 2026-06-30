# Colla Utils - Layout Generation Pipeline

## Overview

This module provides utilities for generating collage layouts from video keyframes, with special handling for edge cells to preserve important content (bounding boxes of detected objects).

---

## Grid Layout V3 - Weighted Treemap with Edge Cell Optimization

### Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT                                         │
│  - Shape mask (e.g., car silhouette)                            │
│  - Keyframes with detected object bounding boxes                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Analyze Frames                                         │
│  - Calculate bbox ratio for each frame                          │
│  - bbox_ratio = bbox_area / frame_area                          │
│  - Higher ratio = more important (object fills more of frame)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Create Weighted Treemap                                │
│  - Use squarified treemap algorithm                             │
│  - Cell size proportional to bbox_ratio                         │
│  - Large bbox → Large cell, Small bbox → Small cell             │
│  - Clip cells to shape boundary                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Assign Frames to Cells                                 │
│  - Sort frames by bbox_ratio (descending)                       │
│  - Sort cells by distance to shape centroid (ascending)         │
│  - Large bbox frames → Center cells                             │
│  - Small bbox frames → Edge cells                               │
│  - Reason: Small bbox has more background, easier to crop       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Process Edge Cells (Key Innovation)                    │
│                                                                  │
│  For each cell with coverage < 70%:                             │
│                                                                  │
│  4a. Identify as edge cell (intersects shape boundary)          │
│                                                                  │
│  4b. Try to MERGE with neighbor cell:                           │
│      - Find adjacent cells                                       │
│      - Select best neighbor (highest coverage, good AR match)   │
│      - Merge geometries (union)                                  │
│      - Re-split proportionally to original weights              │
│                                                                  │
│  4c. Set padding strategy for rendering:                        │
│      - crop_tight: Crop to bbox with padding                    │
│      - crop_to_bbox: Crop exactly to bbox area                  │
│      - shift_content: Shift frame so bbox is in visible region  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Render Collage                                         │
│                                                                  │
│  For each frame:                                                │
│    5a. If edge cell with padding strategy:                      │
│        - EdgeCrop: Crop frame to focus on bbox region           │
│        - EdgeShift: Shift content so bbox center aligns with    │
│          visible region center                                   │
│                                                                  │
│    5b. Apply smart crop/warp to fit cell shape                  │
│                                                                  │
│    5c. Composite onto final canvas                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Debug Visualization (Optional)                         │
│  - Draw bounding boxes on collage                               │
│  - Show which boxes are fully visible vs clipped                │
│  - Color code: Green=visible, Red=clipped                       │
└─────────────────────────────────────────────────────────────────┘
```

---

### Key Concepts

#### 1. Weighted Treemap
Instead of uniform grid cells, cell sizes are proportional to the importance (bbox ratio) of assigned frames.

```
Traditional Grid:          Weighted Treemap:
┌───┬───┬───┬───┐         ┌─────────┬───┬───┐
│   │   │   │   │         │         │   │   │
├───┼───┼───┼───┤         │  LARGE  ├───┼───┤
│   │   │   │   │         │         │ S │ S │
├───┼───┼───┼───┤         ├────┬────┼───┴───┤
│   │   │   │   │         │ M  │ M  │   M   │
└───┴───┴───┴───┘         └────┴────┴───────┘
```

#### 2. Frame-to-Cell Assignment Strategy
```
Frames sorted by bbox_ratio:    Cells sorted by centroid distance:
[Large, Large, Med, Med, Small] [Center, Center, Mid, Mid, Edge]
       ↓           ↓                  ↓          ↓
   MATCHED: Large→Center, Small→Edge
```

**Rationale**: 
- Large bbox = important content, needs prominent placement
- Small bbox = more background, can tolerate cropping at edges

#### 3. Edge Cell Handling

**Problem**: Cells at shape boundary get clipped, potentially cutting off important content.

**Solution Pipeline**:
```
Edge Cell Detected (coverage < 70%)
         │
         ▼
    ┌─────────────────┐
    │ Try Merge with  │
    │ Neighbor Cell   │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Re-split with   │
    │ Better Coverage │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Set Padding     │
    │ Strategy        │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Render with     │
    │ EdgeCrop+Shift  │
    └─────────────────┘
```

#### 4. EdgeCrop + EdgeShift

**EdgeCrop**: Crop frame to focus on bbox region
```
Original Frame:              After EdgeCrop:
┌────────────────────┐      ┌──────────────┐
│     background     │      │  ┌────────┐  │
│   ┌────────┐       │  →   │  │ OBJECT │  │
│   │ OBJECT │       │      │  └────────┘  │
│   └────────┘       │      └──────────────┘
└────────────────────┘
```

**EdgeShift**: Shift content so bbox is in visible region of clipped cell
```
Cell clipped by shape:       After EdgeShift:
┌─────────╲                  ┌─────────╲
│  bbox   │╲                 │    ✓    │╲
│  HERE   │ ╲  (bbox         │  bbox   │ ╲
│ (clip!) │  ╲  might be  →  │ VISIBLE │  ╲
└─────────┘   ╲ cut off)     └─────────┘   ╲
```

---

### Usage

```bash
# Run with V3 grid layout
python run.py input_shape.jpg images_folder output_dir 2 \
    --object-detection --grid-layout-v3 --debug

# Flags:
#   --grid-layout-v3    Use weighted treemap with edge optimization
#   --object-detection  Use YOLOE to detect object bboxes
#   --debug             Generate debug visualizations
```

---

### Output Files

```
output_dir/
├── collage.png              # Final collage
├── v3_treemap_layout.png    # Debug: Cell layout visualization
├── collage_bbox_debug.png   # Debug: Collage with bbox overlays
├── slicing_result.json      # Cell assignments and metadata
└── warp_debug_visualization/
    └── ...                  # Per-frame debug images
```

---

### Configuration

Key parameters in `rectangle_grid_layout_v3.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coverage_threshold` | 0.7 | Min coverage to consider cell valid |
| `merge_threshold` | 0.5 | Coverage below this triggers merge |
| `enable_merge` | True | Enable neighbor cell merging |
| `min_cell_area_ratio` | 0.01 | Skip cells smaller than this |

---

### Comparison: V1 vs V3

| Feature | V1 (Uniform Grid) | V3 (Weighted Treemap) |
|---------|-------------------|----------------------|
| Cell sizes | Equal | Proportional to bbox |
| Frame assignment | Sequential | Large→center, Small→edge |
| Edge handling | Basic clip | Merge + Crop + Shift |
| Bbox preservation | May be cut | Optimized to stay visible |
| Best for | Simple layouts | Complex shapes, important content |

---

### Known Limitations

1. **Very small shapes**: If shape area is too small relative to number of frames, some content will inevitably be cropped
2. **Extreme aspect ratios**: Very wide/tall bboxes may not fit well in certain cell shapes
3. **Dense clustering**: If many frames have similar bbox ratios, assignment may not be optimal

---

### Future Improvements

- [ ] Neural optimization for placement (like original Colla paper)
- [ ] Multi-scale rendering for better detail preservation
- [ ] Automatic frame count recommendation based on shape area
- [ ] Support for overlapping placements with transparency
