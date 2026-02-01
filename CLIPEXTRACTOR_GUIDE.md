# Shared CLIPExtractor Guide

## Overview

`CLIPExtractor` is a centralized utility used across multiple pipelines in the LayoutGeneration project:
- **run_inference_v11.py**: DSN keyframe selection (V11 inference)
- **objectfree_pipeline.py**: Character detection & feature retrieval
- **anime_seg_detector.py**: Character segmentation & feature extraction

## Location

**Primary source:**
```
scripts/precompute_script/precompute_all_v11.py
```

**Shared module wrapper:**
```
utils/shared_extractors.py
```

## Usage

### Import from shared module (recommended)
```python
from utils.shared_extractors import CLIPExtractor, MultiPromptScorer

# Initialize
clip_extractor = CLIPExtractor(device='cuda:0')
anime_scorer = MultiPromptScorer(device='cuda:0')

# Extract CLIP features from images/frames
frames = [...]  # List of numpy arrays (H, W, 3) in RGB format
clip_features = clip_extractor.extract(frames)  # Returns (N, 512)

# Extract anime attributes
anime_attrs = anime_scorer.score_frames(frames)  # Returns (N, 6)

# Combine for DSN model
combined_features = np.concatenate([clip_features, anime_attrs], axis=1)  # (N, 518)
```

### Using in each pipeline:

#### 1. run_inference_v11.py (DSN Keyframe Selection)
```python
from utils.shared_extractors import CLIPExtractor, MultiPromptScorer

class V11Predictor:
    def __init__(self, checkpoint_path, device='cuda'):
        self.clip_extractor = CLIPExtractor(device=device)
        self.anime_scorer = MultiPromptScorer(device=device)
    
    def process_video(self, video_path, ...):
        # Extract features for DSN model
        clip_feats = self.clip_extractor.extract(frames)       # (T, 512)
        anime_attrs = self.anime_scorer.score_frames(frames)   # (T, 6)
        feats_full = np.concatenate([clip_feats, anime_attrs], axis=1)  # (T, 518)
        
        # Pass to DSN model
        feats_t = torch.from_numpy(feats_full).unsqueeze(0).to(device)
        probs, _ = self.model(feats_t)
```

#### 2. objectfree_pipeline.py (Character Detection)
```python
from utils.shared_extractors import CLIPExtractor

class Detector:
    def __init__(self, cfg):
        self.clip_extractor = CLIPExtractor(device=cfg["runtime"]["device"])
    
    def extract(self, crops):
        """Extract CLIP features from character crops"""
        feat = self.clip_extractor.extract(crops)
        feat_normalized = F.normalize(torch.from_numpy(feat).float(), dim=1)
        return feat_normalized.cpu().numpy()
```

#### 3. anime_seg_detector.py (Character Segmentation)
```python
# Already integrated - can be extended to use shared extractors
from utils.shared_extractors import CLIPExtractor

detector = AnimeSegmentationDetector(...)
clip_extractor = CLIPExtractor(device='cuda:0')

# Extract features from segmented characters
char_features = clip_extractor.extract(char_crops)
```

## CLIPExtractor Features

### Input
- **Type**: List of numpy arrays or PIL Images
- **Format**: RGB, values 0-255
- **Shape**: (H, W, 3)
- **Batch size**: 1-32 images (configurable)

### Output
- **Shape**: (N, 512) for OpenAI CLIP ViT-B/32
- **Type**: numpy.ndarray
- **Normalized**: Not normalized (you can normalize with `F.normalize()`)

### Models Available
```python
# Default (recommended)
clip_extractor = CLIPExtractor(device='cuda:0')  # "openai/clip-vit-base-patch32"

# Can be customized in precompute_all_v11.py if needed
# "openai/clip-vit-large-patch14"
# "openai/clip-vit-large-patch14-336"
```

## MultiPromptScorer Features

Scores anime-specific visual attributes:
- Action intensity
- Motion blur
- Character detail
- Composition
- Lighting
- Color vibrancy

### Output
- **Shape**: (N, 6) for 6 anime attributes
- **Type**: numpy.ndarray
- **Range**: Typically 0-1 or normalized scores

## Performance Notes

### CLIPExtractor
- **Model**: CLIP ViT-B/32 (63M parameters)
- **Speed**: ~10-15 FPS per GPU (batch of 32)
- **Memory**: ~2GB GPU RAM
- **Latency**: ~50-100ms per batch

### MultiPromptScorer
- **Models**: Multiple small models
- **Speed**: ~20-30 FPS per GPU
- **Memory**: ~1GB GPU RAM
- **Latency**: ~30-50ms per batch

### Batch Processing
```python
# Process multiple images efficiently
frames_batch = frames[0:32]  # Process 32 at a time
features_batch = clip_extractor.extract(frames_batch)

# For 1000 frames: ~1.5 seconds total
```

## Advantages of Centralization

✅ **Single source of truth**: All pipelines use identical feature extraction  
✅ **Consistency**: Same CLIP embeddings across experiments  
✅ **Maintainability**: Update once, applies everywhere  
✅ **Memory efficient**: Shared cache and model initialization  
✅ **Easy testing**: All components use same features  

## Integration with Anime Segmentation

The anime-segmentation detector can be extended to use CLIPExtractor:

```python
from anime_seg_detector import AnimeSegmentationDetector
from utils.shared_extractors import CLIPExtractor

# Initialize both
seg_detector = AnimeSegmentationDetector(model_name="isnet_is")
clip_extractor = CLIPExtractor(device='cuda:0')

# Segment characters
image_path = "scene.jpg"
boxes = seg_detector.get_character_boxes(image_path)

# Load image and extract crops
img = cv2.imread(image_path)
crops = []
for x1, y1, x2, y2 in boxes:
    crop = img[y1:y2, x1:x2]
    crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# Extract CLIP features
features = clip_extractor.extract(crops)  # (N, 512)

# Use for character retrieval/matching
for feat in features:
    # Compare with character bank
    similarity = cosine_similarity(feat[None], bank)[0]
```

## Troubleshooting

### OutOfMemory Error
**Problem**: "CUDA out of memory"  
**Solution**: Reduce batch size in CLIPExtractor
```python
# Process smaller batches
for i in range(0, len(frames), 16):
    batch = frames[i:i+16]
    features = clip_extractor.extract(batch)
```

### Model Download Issues
**Problem**: "Cannot download model from HuggingFace"  
**Solution**: Pre-download or set cache directory
```bash
# Pre-download once
python -c "import clip; clip.load('ViT-B/32', 'cuda')"

# Or set cache
export HF_HOME=/path/to/cache
```

### Feature Dimension Mismatch
**Problem**: "Expected 512 features, got X"  
**Solution**: Check model configuration
```python
# Verify feature dimension
clip_ext = CLIPExtractor(device='cuda')
test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
feat = clip_ext.extract([test_frame])
print(f"Feature shape: {feat.shape}")  # Should be (1, 512)
```

## References

- CLIP Paper: https://arxiv.org/abs/2103.14030
- OpenAI CLIP: https://github.com/openai/CLIP
- Implementation: `scripts/precompute_script/precompute_all_v11.py`

---

**Status**: ✅ Centralized and in use across multiple pipelines  
**Version**: V11 compatible  
**Last Updated**: Jan 30, 2026
