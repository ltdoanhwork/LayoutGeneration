# Đánh giá định lượng collage (CAST) — căn cứ paper §5.4.2

Tài liệu này mô tả **đầu vào**, **ý nghĩa từng độ đo** theo văn bản kiểu SoftCollage / collage có shape, và **chỗ code hiện tại** ([`evaluation.py`](evaluation.py)) **khớp hay lệch** so với paper. Mục đích: bạn tự kiểm tra và quyết định có cần chỉnh pipeline đánh giá hay bổ sung metric.

---

## 1. Chạy nhanh (CLI)

```bash
cd CAST
python evaluation.py \
  --output_dir ./tests/output_onepiece \
  --shape ./tests/output_onepiece/_voronoi_temp.png
```

- `--shape`: mask **cùng kích thước và không gian** với layout (thường `_voronoi_temp.png` sau Voronoi).
- Kết quả: `evaluation_metrics.json`, `evaluation_metrics.csv` trong `output_dir`.

---

## 2. Đầu vào bắt buộc & nên có

| Nguồn | Đường dẫn (mặc định) | Vai trò |
|--------|----------------------|---------|
| **Shape mask** | `--shape` (vd. `.../_voronoi_temp.png`) | Định nghĩa vùng canvas hợp lệ \(P_X\) = số pixel **foreground** của shape (trắng / alpha). |
| **`slicing_result.json`** | `{output_dir}/slicing_result.json` | Đa giác Voronoi từng ô + gán ảnh → tạo **mask ô** (`image_masks`) và **vị trí chuẩn hóa** \(L_i\) (centroid ô / W,H). |
| **`collage.png`** (RGBA) | `{output_dir}/collage.png` | **Không** dùng trực tiếp cho Ma–Ms trong `evaluate_all`; chỉ dùng thêm `collage_coverage`, `white_space` ở bước sau. |
| **Saliency từng frame** | `{output_dir}/isnet_output/saliency_masks/*.jpg` | Mask grayscale/boolean **đã căn theo keyframe**, sau đó code **resize full canvas** và **cắt ∩ ô** tương ứng → \(S_i\) trong Ma, Ms. |

### Ràng buộc để đánh giá **đúng nghĩa paper** cho Ma / Ms

1. **`saliency_masks/` phải tồn tại** và tên file **khớp** `filename` trong `slicing_result.json` → `images[]`. Nếu thiếu: code **fallback** saliency = **mask ô** → Ma/Ms không còn là “foreground IS-Net trong collage”, mà gần như “diện tích ô”.
2. **Cùng độ phân giải**: shape (H,W) = grid đánh giá; saliency và cell mask được resize/vẽ trên cùng (H,W).
3. **`P_X`** = `shape_mask.sum()` = chỉ pixel **trong shape**, không phải toàn canvas hình chữ nhật (nếu shape là mask lỗ khuyết thì đúng hướng paper).

---

## 3. Năm metric trong paper §5.4.2 — định nghĩa & mapping code

Quy ước paper (rút gọn): \(P_X\) = số pixel shape; \(S_i\) = saliency của ảnh \(i\) sau khi đặt lên collage; union trên canvas.

### 3.1 Saliency area — \(M_a\)

- **Paper:** \(M_a = \|\bigcup_i S_i\| / P_X\) — tỷ lệ diện tích shape được **phủ bởi vùng salient** (collage được “thay” từng ảnh bằng saliency mask tương ứng).
- **Code:** `saliency_area()` — OR tất cả `image_saliency_masks` ∩ `shape_mask`, chia `P_X`.
- **Khớp:** Có — nếu mỗi `S_i` là saliency **trên canvas** trong ô (code: resize IS-Net mask rồi `& cell_masks[i]`).
- **Ràng buộc:** Bắt buộc có saliency thật (thư mục `saliency_masks`); nếu không, metric **không** tương đương paper.

### 3.2 Compactness — \(M_c\)

- **Paper:** \(M_c = P_w / P_X\), \(P_w\) = pixel **khoảng trắng** (trong shape, không bị ảnh phủ).
- **Code:** `compactness()` — \(P_w\) = pixel thuộc shape mà **không** thuộc **union các mask ô Voronoi** (`image_masks` từ `slicing_result.json`).
- **Lệch có thể xảy ra:** Voronoi thường **phủ kín** shape → union ô ≈ shape → **\(M_c \approx 0\)** kể cả khi **ảnh render** trong ô còn trong suốt (alpha lỗ). Paper/bảng baseline có thể đo **khoảng trắng trên ảnh collage cuối** (pixel không có màu ảnh).
- **Trong code có thêm:** `white_space` từ **alpha `collage.png`** (gần nghĩa “khoảng trống thật” hơn) nhưng **không** được gán là `Mc` trong dict chính.
- **Kết luận:** `Mc` hiện tại = **compactness theo hình học ô**, không nhất thiết = paper nếu paper dùng **raster collage**.

### 3.3 Non-overlapping — \(M_o\)

- **Paper:** \(M_o = P_o / P_X\), \(P_o\) = tổng mức **chồng lấn** giữa các ảnh (các cặp ảnh đồng thời chiếm cùng pixel).
- **Code:** với mỗi pixel trong shape, đếm số ô phủ \(k\), cộng \(\binom{k}{2}\), tổng = \(P_o\), chia \(P_X\).
- **Khớp:** Cách định nghĩa \(P_o = \sum_p \binom{k(p)}{2}\) tương ứng “tổng (theo pixel) số cặp ảnh chồng nhau”.

### 3.4 Correlation preservation — \(M_n\)

- **Paper:** \(M_n = \frac{1}{N}\sum_i \|L_i - L_{c_i}\|\), \(L_i\) vị trí ảnh \(i\) trên collage, \(L_{c_i}\) **centroid danh mục** (AIC), tọa độ **chuẩn hóa** theo W,H của shape — **càng nhỏ càng tốt**.
- **Code:** `correlation_preservation()` + `cell_centroids` từ centroid ô (đã norm). **Nhưng** trong `evaluate_all()`, nếu `category_centroids is None` thì code gán `category_centroids = image_locations.copy()` → **\(L_i = L_{c_i}\)** → **\(M_n = 0\) luôn**.
- **Khớp paper:** **Không** khi chạy CLI mặc định — thiếu file/metadata **category centroids** từ dataset (AIC).
- **Để đúng paper:** Cần truyền danh sách \(L_{c_i}\) cùng thứ tự với `images` (mở rộng API / file JSON riêng).

### 3.5 Saliency loss — \(M_s\)

- **Paper:** \(M_s = 1 - \|\bigcup_i S_i\| / \sum_i |S_i|\) — mức “mất” salient khi chồng/ghép (union nhỏ hơn tổng từng mảnh).
- **Code:** `saliency_loss()` cùng công thức trên mask bool full-canvas.
- **Khớp:** Có — với điều kiện \(S_i\) như mục Ma (saliency trong ô, không overlap giữa các ô thì thường \(M_s \approx 0\)).

---

## 4. Metric thêm trong code (không nằm trong đoạn paper bạn trích)

| Tên | Ý nghĩa ngắn |
|-----|----------------|
| **LQ** | \(1 - M_c\) theo union ô — “tỷ lệ shape bị ô phủ”. |
| **DRE** | MAE giữa “target size” (diện tích ô chuẩn hóa từ slicing) và diện tích mask ô — đo **sai lệch bố cục mong muốn vs ô**, không có trong bảng paper. |
| **collage_coverage** / **white_space** | Từ alpha collage — gần **Mc theo ảnh thật** hơn là theo ô. |
| **avg_bbox_coverage** | Từ `bbox_retention.json` nếu có — liên quan crop/warp IS-Net, không phải 5 metric paper. |

---

## 5. Tóm tắt: độ đo nào “phù hợp / chưa phù hợp” với phương pháp bạn

| Metric | Phù hợp với paper (với điều kiện đầu vào) | Ghi chú |
|--------|------------------------------------------|---------|
| **Ma** | Có — nếu có `saliency_masks/` đúng file | CAST dùng IS-Net → bbox → mask ô; đúng hướng “event support → saliency trên canvas”. |
| **Ms** | Có — cùng điều kiện Ma | |
| **Mo** | Có — theo mask ô | Voronoi thường không chồng ô → \(M_o\) thường 0; baseline SHP chồng ô mới khác biệt. |
| **Mc** | **Một phần** | Code = khoảng trống theo **ô Voronoi**, không phải chắc chắn khoảng trống **render** như paper có thể hàm ý. |
| **Mn** | **Chưa** (CLI hiện tại) | Thiếu centroid category; cần dữ liệu ngoài (AIC hoặc tự định nghĩa cluster). |
| **Mn_timeline** | Phù hợp **keyframe** | Không cần category; đo khớp timeline vs thứ tự đọc ô. |

---

## 6. Có nên thêm độ đo khác?

Tùy mục tiêu bài báo / ablation:

1. **Mc từ collage (đề xuất nếu so baseline “white space” như paper):** dùng alpha `collage.png` làm `image_masks` hoặc định nghĩa `Mc_render = P_w_render / P_X` và báo cáo song song với `Mc_cell`.
2. **Mn (AIC):** chỉ khi có **nhãn category** → file `category_centroids.json` cùng thứ tự ảnh. Với chỉ keyframe, dùng **`Mn_timeline`** đã tích hợp CLI.
3. **Bảo toàn saliency sau warp:** so sánh IoU hoặc coverage bbox IS-Net trong ô trước/sau warp (đã có phần `bbox_retention` — có thể chuẩn hóa tên metric).
4. **Temporal / reading order:** paper nhấn timeline — có thể thêm penalty thứ tự ô vs thứ tự frame (metric riêng CAST, không trong 5 metric gốc).

---

## 7. Checklist trước khi so sánh với bảng paper

- [ ] `slicing_result.json` + `_voronoi_temp.png` đúng run.
- [ ] `isnet_output/saliency_masks/` đầy đủ file trùng `filename` trong JSON.
- [ ] Hiểu rõ `Mc` đang là **theo ô**, không nhầm với `white_space` từ collage.
- [ ] `Mn` (AIC): chỉ interpret nếu có **category centroids**; default CLI → `Mn=0` (không có nghĩa).
- [ ] `Mn_timeline`: dùng cho **keyframe** — kiểm tra thứ tự `images` = timeline; đọc kết quả trong `evaluation_metrics.json`.
- [ ] Baseline TB/SHP/SC+Mask: cùng **định nghĩa** \(P_X\), \(S_i\), và **nguồn saliency** (cùng IS-Net + threshold nếu so công bằng).

---

*Tài liệu này phản ánh implementation tại `CAST/evaluation.py` (evaluate_run_output / evaluate_all). Khi chỉnh code, nên cập nhật lại mục 3–5 cho khớp.*
