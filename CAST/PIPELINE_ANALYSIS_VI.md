# CAST Pipeline - Phân Tích Chi Tiết & Đánh Giá

> **Latest Update**: 2026-02-08 - Pipeline đã được đơn giản hóa: Bỏ Phase 2 (Refinement) và Hungarian Matching

## 📋 Tổng Quan Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           CAST PIPELINE FLOW (v3)                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STEP 0: Mask Generation                                                        │
│    └─→ U2-Net: RGB Shape Image → Binary Mask (shape_mask_refined.png)           │
│                                                                                  │
│  STEP 0.5: Metadata Loading                                                     │
│    └─→ Load `summary.json`: Real BBox, Confidence, Num Objects                  │
│    └─→ Filter: Bỏ frames không có object                                        │
│                                                                                  │
│  STEP 1: Shape Decomposition                                                    │
│    └─→ Medial Axis Skeleton + Ridge Detection (for site initialization)         │
│                                                                                  │
│  STEP 2: Voronoi Layout Optimization (SINGLE-PHASE)                             │
│    ├─→ 2.1: Site Initialization (Medial Axis / Grid)                            │
│    ├─→ 2.2: Compute Target Caps/Aspects (từ bbox_area trong JSON)               │
│    ├─→ 2.3: Iterative Optimization (L_cap, L_asp, L_ov)                        │
│    ├─→ 2.4: Generate Polygons (Voronoi → Hard cells)                            │
│    └─→ 2.5: Spatial Order Assignment (reading order: TopLeft → BottomRight)     │
│                                                                                  │
│  STEP 3: Collage Assembly                                                        │
│    ├─→ Smart Cover Crop V2 (Input: Real BBox từ JSON)                           │
│    ├─→ Polygon Clipping (apply cell mask to each image)                         │
│    └─→ Alpha Compositing (blend onto canvas)                                    │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### ❌ ĐÃ LOẠI BỎ (so với v2)

- **Phase 2 (Refinement)**: 200 iterations thừa, coverage giảm thay vì tăng (-2.2%), tốn ~130s
- **Hungarian Matching**: Không cần vì dùng timeline order (spatial rank)
- **`--refine-iters` flag**: Đã xóa khỏi `run.py`

---

## 📥 INPUTS

| Input | Format | Mục đích |
|-------|--------|----------|
| **Layout Image** | `poke.png` (RGB) | Hình dạng layout |
| **Keyframes Directory** | `/keyframes/*.jpg` | Video frames |
| **Summary JSON** | `summary.json` | BBox thật [x1,y1,x2,y2] + confidence mỗi frame |

**Summary JSON Format:**
```json
{
  "frames": [
    {
      "name": "scene_000_frame_000025.jpg",
      "num_objects": 1,
      "objects": [{
        "bbox": [156, 144, 488, 508],
        "pixel_area": 177511,
        "confidence": 0.62
      }]
    }
  ]
}
```

---

## 🔄 PIPELINE CHI TIẾT

### STEP 0: Mask Generation ⏱️ ~2s

```python
image = cv2.imread("poke.png")
inputs, orig_h, orig_w = preprocess_image(image)
pred_mask = predict_mask(net, inputs)
mask_refined = refine_mask(pred_mask, orig_h, orig_w)
# Output: shape_mask_refined.png (WHITE=foreground, BLACK=background)
```

### STEP 0.5: Load Metadata ⏱️ ~0.1s

```python
with open("summary.json") as f:
    data = json.load(f)

frame_infos = []
for frame in data['frames']:
    if frame['num_objects'] > 0:
        best_obj = max(frame['objects'], key=lambda x: x['pixel_area'])
        frame_infos.append({
            'path': f"keyframes/{frame['name']}",
            'bbox': best_obj['bbox'],
            'bbox_area': (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]),
            'confidence': best_obj['confidence']
        })
```

### STEP 1: Shape Decomposition ⏱️ ~2s

- Medial axis skeleton cho site initialization
- Không ảnh hưởng trực tiếp đến optimization

---

### STEP 2: Voronoi Layout Optimization ⏱️ ~60-70s

**Pipeline chính xác (giống test_timeline_order.py):**

```python
# 1. Initialize engine
engine = VoronoiLayoutEngine(
    mask_path=mask_path,
    frame_infos=frame_infos,
    use_timeline_order=True,
    output_dir=output_dir
)

# 2. Optimize (SINGLE PHASE - chỉ 1 vòng optimize)
sites, weights = engine.optimize()

# 3. Generate hard polygons từ soft probs
polygons = engine.generate_polygons(sites, weights)

# 4. Spatial order assignment (reading order, giữ timeline)
assignment = engine.match_images_spatial_order(polygons)
# → image[0] → top-left cell, image[N] → bottom-right cell

# XONG. Không có Phase 2, không có Hungarian.
```

#### 2.1: Site Initialization

| Method | Cách hoạt động | Khi nào dùng |
|--------|---------------|-------------|
| **Grid** | Sites đều theo lưới | `use_timeline_order=True` (default) |
| **Medial Axis** | Endpoints + DT fill | Khi cần respect topology |

#### 2.2: Target Capacities & Aspects

```python
# Từ summary.json bbox thật
for info in frame_infos:
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    target_caps.append(bbox_area / total_area)   # → diện tích cell
    target_aspects.append(bbox_w / bbox_h)         # → hình dạng cell (clamp [0.3, 3.0])
```

#### 2.3: Optimization Loop

Ta tối ưu một objective hợp phần theo dạng weighted multi-objective:

$$
\mathcal{L}_{total} = \lambda_{cap}\mathcal{L}_{cap} + \lambda_{asp}\mathcal{L}_{asp} + \lambda_{ov}\mathcal{L}_{ov}
$$

với thiết lập mặc định hiện tại:

```python
CONFIG = {
    'resolution': 512,
    'num_iterations': 150,
    'tau': 60.0,
    'w_capacity': 400.0,   # lambda_cap
    'w_aspect': 600.0,     # lambda_asp
    'w_overlap': 1500.0,   # lambda_ov
}
```

Trong đó, mỗi iteration tạo soft Voronoi assignment:

$$
p_i(x) = \operatorname{softmax}\big(-\tau\,(d_i(x)-w_i+b(x))\big)
$$

với $p_i(x)$ là xác suất pixel $x$ thuộc cell $i$, $d_i$ là anisotropic distance, $w_i$ là site weight, $b(x)$ là mask penalty ngoài shape.

##### 2.3.1 Capacity Loss $\mathcal{L}_{cap}$ (điều khiển phân bổ diện tích)

Mục tiêu: diện tích soft của mỗi cell bám theo tỷ lệ diện tích bbox mục tiêu lấy từ `summary.json`.

$$
A_i = \sum_{x \in \Omega} p_i(x)\,m(x), \quad
\hat{A}_i = \rho_i\,|\Omega|,
$$

$$
\mathcal{L}_{cap} = \frac{1}{N}\sum_{i=1}^{N}(A_i-\hat{A}_i)^2
$$

- $m(x) \in \{0,1\}$ là shape mask.
- $\rho_i$ là tỷ lệ diện tích bbox của frame $i$ sau chuẩn hóa tổng.

Ý nghĩa thực nghiệm: loss này giữ tổng thể bố cục "đúng budget diện tích" giữa các frame lớn/nhỏ, tránh cell bị lệch cực đoan khi chỉ tối ưu overlap.

##### 2.3.2 Aspect Loss $\mathcal{L}_{asp}$ (điều khiển hình dạng cell)

Mục tiêu: ép độ dẹt/kéo dài của cell gần với aspect ratio bbox mục tiêu.

Gọi $\sigma_{x,i}^2, \sigma_{y,i}^2$ là phương sai có trọng số theo $p_i(x)$ quanh centroid cell, ta có:

$$
r_i = \sqrt{\frac{\sigma_{x,i}^2}{\sigma_{y,i}^2}},
\quad
r_i^* = \operatorname{clip}\left(\frac{w_i^{bbox}}{h_i^{bbox}}, 0.3, 3.0\right)
$$

$$
\mathcal{L}_{asp} = \frac{1}{N}\sum_{i=1}^{N}\left(\log r_i - \log r_i^*\right)^2
$$

Dùng miền log để đối xứng sai số cho trường hợp quá rộng và quá cao, giúp ổn định hơn khi dữ liệu có phân bố aspect dài đuôi.

##### 2.3.3 Overlap Retention Loss $\mathcal{L}_{ov}$ (loss lõi cho giữ nội dung)

Mục tiêu: tối đa phần bbox "quan trọng" của ảnh nằm trong chính cell được tối ưu.

Với mỗi cell $i$, tạo bbox mask chuẩn hóa tâm theo centroid cell (anchor theo identity mapping trong optimize), rồi tính:

$$
o_i = \frac{\sum_{x \in \Omega} p_i(x)\,b_i(x)\,m(x)}{\sum_{x \in \Omega} b_i(x)\,m(x)+\varepsilon}
$$

$$
\mathcal{L}_{ov} = \frac{1}{N}\sum_{i=1}^{N}(1-o_i)^2
$$

- $b_i(x)$ là bbox prior mask của frame neo cho cell $i$.
- $o_i \in [0,1]$ là tỷ lệ overlap chuẩn hóa.

Đây là thành phần quan trọng nhất vì liên quan trực tiếp đến khả năng giữ foreground/object sau khi cắt cell.

##### 2.3.4 Cách chọn hệ số $\lambda$ (paper-style, có thể tái lập)

Thiết lập hiện tại:

$$
(\lambda_{cap},\lambda_{asp},\lambda_{ov}) = (400, 600, 1500)
$$

được chọn theo 3 nguyên tắc:

1. **Ưu tiên semantic retention**: đặt $\lambda_{ov}$ lớn nhất để giảm lỗi bỏ sót vùng bbox quan trọng.
2. **Giữ hình học trung gian ổn định**: $\lambda_{asp} > \lambda_{cap}$ vì sai lệch aspect gây méo crop/warp dễ thấy hơn sai lệch diện tích nhẹ.
3. **Cân bằng độ lớn gradient ban đầu**: ở 10-20 iteration đầu, điều chỉnh sao cho ba gradient term cùng bậc độ lớn, sau đó bias nhẹ về $\mathcal{L}_{ov}$.

##### 2.3.5 Quy trình tuning khuyến nghị

Để chuyển domain (anime khác phong cách, bbox nhiễu hơn), nên tune theo thứ tự:

1. Cố định $\tau$, tăng/giảm $\lambda_{ov}$ trước đến khi bbox retention đạt ngưỡng mục tiêu.
2. Tune $\lambda_{asp}$ để giảm méo hình (quan sát phân phối $\log r_i-\log r_i^*$).
3. Tune $\lambda_{cap}$ cuối để sửa sai lệch diện tích tổng thể mà không phá retention.

Khoảng thử nghiệm thực dụng:

- $\lambda_{ov} \in [1000, 2500]$
- $\lambda_{asp} \in [300, 900]$
- $\lambda_{cap} \in [200, 800]$

Heuristic nhanh:

- Nếu cell đúng size nhưng object hay bị cắt: tăng $\lambda_{ov}$.
- Nếu object giữ được nhưng cell quá dẹt/quá cao: tăng $\lambda_{asp}$.
- Nếu bố cục bị lệch tỷ lệ lớn-nhỏ giữa các frame: tăng $\lambda_{cap}$.

##### 2.3.6 Lưu ý khi diễn giải loss

- Vì optimize ở soft assignment, còn render dùng hard polygon (`argmin`), nên giá trị loss thấp không đảm bảo tuyệt đối coverage trên polygon cuối.
- `match_images_spatial_order()` diễn ra sau optimize, nên có thể có anchor mismatch nhẹ; cấu hình lambda hiện tại đã thực nghiệm đủ bền với mismatch này.

#### 2.4: Polygon Generation

```python
# render_res=4096 grid trên CPU (high-res cho polygon accuracy)
labels = argmin(aniso_distance - weights)     # Hard assignment
labels[mask < 127] = -1                       # Background
contours → simplify → Polygon → buffer
```

#### 2.5: Spatial Order Assignment

```python
def match_images_spatial_order(polygons):
    """
    Sort cells by spatial rank: Top-Left → Bottom-Right (reading order)
    Assign image[i] → sorted_cell[i]
    """
    # Cell reading order = y_centroid * weight + x_centroid
    # image[0] (earliest frame) → top-left cell
    # image[N] (latest frame) → bottom-right cell
```

**Tại sao không dùng Hungarian?**
- Timeline order là yếu tố quan trọng nhất cho video summarization
- Hungarian có thể xáo trộn thứ tự → khó hiểu cho người xem
- Spatial order + L_ov optimization đã cho coverage đủ tốt, Smart Crop V2 bù phần còn lại

---

### STEP 3: Collage Assembly ⏱️ ~2-6s

#### Smart Cover Crop V2 (khi `--no-warp`)

```python
def smart_cover_crop(image, foreground_box, target_w, target_h):
    # 1. Scale full frame cover target
    scale_cover = max(target_w / w_src, target_h / h_src)
    
    # 2. Check overflow → Blur Padding fallback
    if scaled_bbox > target * 1.2:
        return _crop_with_blur_padding(image, foreground_box, target_w, target_h)
    
    # 3. Sliding window optimization (giữ bbox trong crop)
    # 4. Head priority (ưu tiên giữ phần đầu bbox)
```

| Feature | Mô tả |
|---------|--------|
| **Cover Crop** | Scale full frame lấp đầy cell |
| **Sliding Window** | Shift crop để bbox nằm trong |
| **Blur Padding** | Fallback khi bbox quá to (fit + blur background) |
| **Head Priority** | Ưu tiên giữ phần đầu bbox |

#### Content-Aware Warp (mặc định, tắt bằng `--no-warp`)

```
Source image → ISNet saliency mask → Dense mesh → L-BFGS-B optimization → TPS warp
```

| Method | Time (25 imgs) | Coverage | Khi nào dùng |
|--------|---------------|----------|-------------|
| **Smart Crop V2** | ~0.1s | ~98% bbox | Fast, prototyping |
| **Content-Aware Warp** | ~15-25s | Full frame | Production output |

---

## 📊 PERFORMANCE

| Step | Time | Key Metric |
|------|------|------------|
| 0. Mask Generation | ~2s | U2-Net inference |
| 0.5. Load Summary | ~0.1s | Filter empty frames |
| 1. Shape Decomposition | ~2s | Medial axis |
| 2. Voronoi Optimization | ~60-70s | N iterations, 3 losses |
| 3. Collage Assembly (crop) | ~2s | Smart crop |
| 3. Collage Assembly (warp) | ~15-25s | Content-aware warp |
| **TOTAL (crop)** | **~70s** | |
| **TOTAL (warp)** | **~90s** | |

---

## ⚠️ KNOWN ISSUES

### 1. Anchor Mismatch (Lý thuyết, ảnh hưởng nhỏ)

```python
# DURING optimization: anchor_img_idx = [0, 1, 2, ..., N]
# → Cell 0 optimized cho Image 0, Cell 1 cho Image 1, ...

# AFTER optimization: spatial order reassigns
# → Cell 0 → Image 5, Cell 1 → Image 12, ... (theo reading order)
```

**Tại sao chấp nhận được?**
1. **L_cap và L_asp là image-agnostic** — chỉ phụ thuộc bbox statistics, không phụ thuộc assignment
2. **Grid init → sites tự nhiên sắp xếp reading order** → mismatch nhỏ
3. **Smart Crop V2 bù** → Sliding window + blur padding xử lý phần còn lại

### 2. Soft → Hard Gap

```
Optimization (soft probs, gradient)  ≠  Reality (hard polygon, argmin)
→ L_ov = 0.1 KHÔNG có nghĩa 90% bbox nằm trong polygon thật
```

### 3. Small Cells ở vùng hẹp (tai, chân, đuôi)

- Outlier frames với object nhỏ → cell rất nhỏ
- Fix: `min_cap_ratio=0.02` ngăn cells quá nhỏ

---

## 🔧 Commands

```bash
# Standard (crop mode, nhanh)
python run.py shape.jpg keyframes/ output/ 2 \
    --voronoi-layout --timeline-order --no-warp \
    --num-iterations=100 --summary-json=summary.json

# Warp mode (chất lượng cao)
python run.py shape.jpg keyframes/ output/ 2 \
    --voronoi-layout --timeline-order \
    --num-iterations=100 --summary-json=summary.json \
    --warp-mask-dir=masks/

# Full debug
python run.py shape.jpg keyframes/ output/ 2 \
    --voronoi-layout --timeline-order --no-warp --debug \
    --num-iterations=200 --summary-json=summary.json

# Rotation search
python run.py shape.jpg keyframes/ output/ 2 \
    --voronoi-layout --timeline-order --rotation-search
```

---

## 🧪 Test Scripts

```bash
# Test full pipeline (REFERENCE cho correct flow)
python tests/test_timeline_order.py
# Flow: optimize() → generate_polygons() → match_images_spatial_order() → visualize

# Test optimization loss analysis
python tests/test_optimization.py

# Test real pipeline end-to-end  
python tests/test_real_pipeline.py

# Test site initialization
python tests/test_site_initialization.py
```

---

## 📁 Files Reference

| File | Chức năng |
|------|----------|
| `run.py` | Entry point, argument parsing |
| `voronoi_layout.py` | VoronoiLayoutEngine, optimize(), spatial order, polygon generation |
| `collage_assembly.py` | Smart crop V2, content-aware warp, rendering |
| `sas_optimization.py` | Orchestrator: load data → voronoi → save JSON |
| `shape_decomposition.py` | Medial axis, skeleton, site init data |

---

## Content-Aware Warp (Chi tiết)

### Architecture

```
INPUT
├─ Source image: 848×480 (RGBA)
├─ Foreground BBox: [x1, y1, x2, y2] from summary.json
├─ ISNet saliency mask (--warp-mask-dir)
└─ Target cell: Voronoi polygon bounding rect

PHASE 1: SALIENCY
├─ Option A: Load ISNet mask → GaussianBlur → [0,1] saliency map
└─ Option B: U2-Net inference (fallback)

PHASE 2: MESH CREATION
└─ grid_size = max(24, min(w,h)//16) → ~490 control points

PHASE 3: MESH OPTIMIZATION (L-BFGS-B, vectorized + analytical gradient)
├─ L_salient: Pull fg mesh points → proportional target (weight=350)
├─ L_background: Keep bg points near init (weight=0.4)
├─ L_boundary: Hard penalty for out-of-bounds (1e6)
└─ L_smooth: Consecutive point regularization (lambda=0.8)

PHASE 4: TPS WARP
├─ RBFInterpolator (thin_plate_spline, inverse mapping)
└─ cv2.remap (INTER_CUBIC + BORDER_REFLECT)

OUTPUT: Warped image (target_h × target_w × RGBA)
```

### Loss Function

$$L_{total} = L_{salient} + L_{bg} + L_{boundary} + L_{smooth}$$

$$L_{salient} = \sum_{i \in \text{fg}} w_i \cdot 350 \cdot \|d_i - t_i\|_2$$

$$L_{bg} = \sum_{i \notin \text{fg}} \max(0.4, w_i \cdot 0.4) \cdot \|d_i - d_i^{init}\|_2$$

$$L_{boundary} = 10^6 \cdot \sum_{i} \mathbb{1}[d_i \notin \text{cell}]$$

$$L_{smooth} = 0.8 \cdot \sum_{i} \|d_{i+1} - d_i\|_2$$

### Inner Rectangle Mapping

```python
# Proportional mapping: fg lands at same relative position in target
# → dst_pts_init ≈ inner_rect_dst → minimal deformation
# → fg preserved, bg stretches mildly to fill aspect ratio difference
```

### Performance

| Config | Time (25 imgs) |
|--------|---------------|
| No warp (`--no-warp`) | ~0.1s |
| Warp + ISNet masks | ~15-25s |
| Warp + U2-Net live | ~60-90s |

---

*Last Updated: 2026-02-08 - Simplified pipeline (removed Phase 2 refinement + Hungarian)*
