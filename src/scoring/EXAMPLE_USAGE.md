# Ví dụ sử dụng temporal_segmenter.py

## 1. Sử dụng qua Python API

```python
from src.scoring.temporal_segmenter import AutoSegmenter

# Khởi tạo segmenter
segmenter = AutoSegmenter(
    w_clip=0.8,        # 80% CLIP, 20% IQA
    w_iqa=0.2,
    min_len=3,         # Mỗi segment tối thiểu 3 ảnh
    max_len=5,         # Mỗi segment tối đa 5 ảnh
    mode="auto",       # Tự động chọn mode
    kmin=2,            # Tối thiểu 2 segments
    kmax=10            # Tối đa 10 segments
)

# Phân đoạn ảnh
segments_idx, segments_files = segmenter.segment("/path/to/image/folder")

# Xử lý kết quả
print(f"Tìm thấy {len(segments_idx)} segments")
for i, (idx, files) in enumerate(zip(segments_idx, segments_files), 1):
    print(f"Segment {i}: {len(idx)} ảnh")
    print(f"  Files: {files[0]} ... {files[-1]}")
```

## 2. Sử dụng qua Command Line

```bash
# Chạy với tham số mặc định
python src/scoring/temporal_segmenter.py --dir /path/to/images

# Chạy với tham số tùy chỉnh
python src/scoring/temporal_segmenter.py \
    --dir /path/to/images \
    --mode boundary \
    --min-len 3 \
    --max-len 4 \
    --w-clip 0.8 \
    --w-iqa 0.2 \
    --kmin 2 \
    --kmax 10 \
    --save output.json
```

## 3. Cấu trúc thư mục đầu vào

```
/path/to/images/
├── scene_01_frame_001.jpg
├── scene_01_frame_002.jpg
├── scene_01_frame_003.jpg
├── scene_02_frame_001.jpg
├── scene_02_frame_002.jpg
└── ...
```

## 4. Kết quả đầu ra

### Console Output:
```
Detected 3 segments
Segment 01: idx [0..2], len=3
  First: /path/to/images/scene_01_frame_001.jpg
  Last : /path/to/images/scene_01_frame_003.jpg
Segment 02: idx [3..5], len=3
  First: /path/to/images/scene_02_frame_001.jpg
  Last : /path/to/images/scene_02_frame_003.jpg
```

### JSON Output (nếu có --save):
```json
{
  "segments_idx": [[0, 1, 2], [3, 4, 5]],
  "segments_files": [
    ["/path/.../scene_01_frame_001.jpg", ...],
    ["/path/.../scene_02_frame_001.jpg", ...]
  ],
  "params": {
    "min_len": 3,
    "max_len": 4,
    "mode": "boundary",
    "w_clip": 0.8,
    "w_iqa": 0.2,
    "kmin": 2,
    "kmax": 10
  }
}
```

