# Object-Free Pipeline với Story Coherence & Annotated Images

Pipeline hoàn chỉnh để đánh giá video keyframes với object detection, story coherence filtering, và tạo annotated images với bbox màu sắc.

## Tổng quan

Pipeline này thực hiện:
1. **Object Detection**: Phát hiện objects trong keyframes bằng Grounding DINO + SAM
2. **Story Coherence Filtering**: Lọc bbox dựa trên độ tương đồng semantic với caption toàn cục
3. **Annotated Images**: Tạo ảnh đã annotate với bbox có màu sắc và similarity scores
4. **Comprehensive Evaluation**: Phân tích entropy, edge density, color variance

## Cấu trúc Output

```
outputs/[video_name]_object_free_[timestamp]/
├── keyframes/                    # Ảnh keyframes gốc
├── scene_previews/              # Preview scenes
├── object_free_evaluation/
│   └── keyframes/
│       ├── object_detection/     # Detection results JSON
│       ├── story_coherence/      # Coherence evaluation results
│       ├── annotated_images/     # ⭐ ANH ĐÃ ANNOTATE VỚI BBOX MÀU
│       ├── comprehensive_eval/   # Quality analysis
│       └── final_report.json     # Báo cáo tổng hợp
```

## Cấu hình

Chỉnh sửa `objectfree/config.yaml`:

```yaml
config:
  TEXT_PROMPT: "objects"                    # Prompt cho object detection
  BOX_THRESHOLD: 0.02                      # Ngưỡng confidence cho bbox
  TEXT_THRESHOLD: 0.02                     # Ngưỡng matching text
  SIMILARITY_THRESHOLD: 0.3                # ⭐ Ngưỡng lọc bbox theo similarity
  # ... other configs
```

## Chạy Pipeline

### Từ Main Pipeline

```bash
python pipeline.py \
  --video data/samples/Sakuga/14652.mp4 \
  --run_object_free_pipeline \
  --detection_config objectfree/config.yaml \
  --out_dir outputs/test_annotated
```

### Standalone Object-Free Pipeline

```python
from objectfree.eval_objectfree import CompletePipeline

# Khởi tạo pipeline
pipeline = CompletePipeline(
    device="cuda",
    config_path="objectfree/config.yaml"
)
pipeline.initialize_detectors()

# Chạy trên folder keyframes
result = pipeline.process_single_folder(
    keyframes_folder="path/to/keyframes",
    output_base="outputs"
)
```

## Story Coherence Filtering

### Cách hoạt động:

1. **BLIP Caption**: Tạo caption cho ảnh toàn cục và từng crop
2. **Semantic Similarity**: Tính cosine similarity giữa embeddings
3. **Color Assignment**: Gán màu cho từng bbox (12 màu khác nhau)
4. **Filtering**: Chỉ giữ bbox có similarity >= threshold

### Màu sắc BBOX:

- **#FF0000**: Red
- **#00FF00**: Green
- **#0000FF**: Blue
- **#FFFF00**: Yellow
- **#FF00FF**: Magenta
- **#00FFFF**: Cyan
- **#FFA500**: Orange
- **#800080**: Purple
- **#FFC0CB**: Pink
- **#A52A2A**: Brown
- **#808080**: Gray
- **#000000**: Black

## Annotated Images

Ảnh output trong `annotated_images/` sẽ có:
- **BBOX màu sắc** cho mỗi detection được giữ lại
- **Label** hiển thị: `ID{detection_id}: {similarity:.3f}`
- **Background màu** cho text để dễ đọc

### Ví dụ:

```
scene_0000_frame_00000030_annotated.png
scene_0001_frame_00000115_annotated.png
...
```

## Điều chỉnh Threshold

### Thay đổi trong config:

```yaml
SIMILARITY_THRESHOLD: 0.5  # Tăng để lọc strict hơn
# hoặc
SIMILARITY_THRESHOLD: 0.1  # Giảm để giữ nhiều bbox hơn
```

### Thay đổi trong code:

```python
# Trong eval_objectfree.py
story_results = self.run_story_coherence(
    detection_json, keyframes_folder, story_output,
    similarity_threshold=0.5  # Override config
)
```

## Troubleshooting

### Không có annotated images:

- Kiểm tra `story_coherence` có chạy thành công không
- Kiểm tra `similarity_threshold` không quá cao

### BBOX màu không hiển thị:

- Kiểm tra PIL và font được cài đặt
- Kiểm tra bbox coordinates hợp lệ

### Memory issues:

- Giảm `batch_size` trong inference_dino.py
- Sử dụng model nhỏ hơn

## Dependencies

```bash
pip install torch torchvision transformers sentence-transformers supervision
pip install pyyaml pillow matplotlib tqdm scikit-learn
```

## API Reference

### CompletePipeline

```python
class CompletePipeline:
    def __init__(self, device="cuda", output_dir=None, config_path="objectfree/config.yaml")
    def process_single_folder(self, keyframes_folder, output_base)
    def annotate_image_with_bbox(self, image_path, crop_results, output_path=None)
```

### StoryCoherenceEvaluator

```python
class StoryCoherenceEvaluator:
    def evaluate_batch(self, detection_results_json, keyframes_folder, output_dir,
                      save_crops=False, similarity_threshold=0.3)
    def compute_similarity(self, caption1, caption2) -> float
    def filter_by_similarity(self, similarity, threshold=0.3) -> bool
    def assign_bbox_color(self, det_idx) -> str
```

## Performance Notes

- **GPU Memory**: ~4GB cho batch_size=1
- **Processing Time**: ~2-3 giây per image (tùy config)
- **Storage**: ~50KB per annotated image

## Examples

### Chạy với threshold cao (strict filtering):

```yaml
SIMILARITY_THRESHOLD: 0.5
```

### Chạy với threshold thấp (keep more bboxes):

```yaml
SIMILARITY_THRESHOLD: 0.1
```

### Custom color scheme:

Sửa method `assign_bbox_color()` trong `story_coherence_evaluator.py`.