# Anime Segmentation Integration Guide

## Overview

This guide explains how to use the anime-segmentation models for character detection in the ObjectFree pipeline. The anime-segmentation models use ISNet (Deep Interactive Segmentation) to detect and mask anime characters, replacing the previous SAM3 + YOLOE approach.

## Components

### 1. **anime_seg_detector.py**
Standalone inference wrapper for anime-segmentation models.

**Key Classes:**
- `AnimeSegmentationDetector`: Main detector class with various utility methods

**Key Methods:**
```python
detector = AnimeSegmentationDetector(
    model_name="isnet_is",      # isnet_is, isnet, u2net, u2netl
    device="cuda:0",             # GPU device
    model_path=None,             # Path to weights (auto-downloads if None)
    img_size=1024                # Input resolution
)

# Get mask
mask = detector.get_mask(numpy_image_rgb)

# Segment image
segmented = detector.segment(image_path, output_mask=False, threshold=0.5)

# Batch process
detector.segment_batch(folder, output_folder, output_format='png')

# Get character boxes
boxes = detector.get_character_boxes(image_path, min_area=100)

# Extract character crops
crops = detector.extract_characters(image_path, output_folder)
```

### 2. **objectfree_pipeline.py**
Updated pipeline that integrates anime-segmentation with CLIP for character retrieval.

**Changes:**
- Replaced `sam3.model_builder` with `anime_seg_detector`
- `Detector.detect()` now uses `anime_seg.get_character_boxes()`
- CLIP encoder remains for feature extraction

### 3. **config_anime_seg.yaml**
Configuration file for anime-segmentation pipeline.

## Installation

### Step 1: Clone the anime-segmentation repo
```bash
cd /home/serverai/ltdoanh/LayoutGeneration/objectfree
git clone https://github.com/SkyTNT/anime-segmentation.git
cd anime-segmentation
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
# May need additional: pytorch-lightning huggingface_hub
```

### Step 3: Download pretrained weights (optional)
The models are automatically downloaded from HuggingFace when first used. Or download manually:

```bash
# Download ISNetIS model
wget -O saved_models/isnetis.ckpt \
  https://huggingface.co/skytnt/anime-seg-isnet-is/resolve/main/model.pth

# Or other models
# isnet: https://huggingface.co/skytnt/anime-seg-isnet/resolve/main/model.pth
# u2net: https://huggingface.co/skytnt/anime-seg-u2net/resolve/main/model.pth
```

## Usage

### Single Image
```bash
python objectfree/anime_seg_detector.py \
  --input /path/to/image.jpg \
  --output ./output \
  --model isnet_is \
  --device cuda:0 \
  --format png
```

### Batch Processing
```bash
python objectfree/anime_seg_detector.py \
  --input /path/to/image/folder \
  --output ./output \
  --model isnet_is \
  --format png
```

### Get Character Bounding Boxes
```bash
python objectfree/anime_seg_detector.py \
  --input /path/to/image.jpg \
  --get-boxes \
  --model isnet_is
```

### Extract Character Crops
```bash
python objectfree/anime_seg_detector.py \
  --input /path/to/image.jpg \
  --output ./crops \
  --extract-chars \
  --model isnet_is
```

### Full ObjectFree Pipeline
```bash
python objectfree/objectfree_pipeline.py \
  --config objectfree/config_anime_seg.yaml
```

## Configuration

Edit `config_anime_seg.yaml` to customize:

```yaml
# Model choice
anime_seg:
  model: "isnet_is"        # isnet_is, isnet, u2net, u2netl
  weights: null            # Path to custom weights
  min_area: 100           # Minimum character area (pixels)

# CLIP settings
clip:
  model_name: "openai/clip-vit-base-patch32"
  cache_dir: ./models/clip

# Character bank
bank:
  high_score_thresh: 0.9   # High confidence threshold
  match_sim_thresh: 0.75   # Similarity threshold
```

## Model Comparisons

| Model | Quality | Speed | Memory | Recommended Use |
|-------|---------|-------|--------|---|
| isnet_is | Highest | Slowest | High | Production, best results |
| isnet | High | Fast | Medium | Balanced |
| u2net | Good | Faster | Lower | Real-time |
| u2netl | Good | Fastest | Lowest | Lightweight |

## Performance Notes

- **Speed**: ISNetIS (~200ms/image), ISNet (~150ms), U2Net (~100ms)
- **GPU Memory**: 4-8 GB GPU RAM recommended
- **Batch Processing**: Currently processes one image at a time; parallelization can be added

## Advantages over SAM3 + YOLOE

✅ **Faster character detection** - Direct segmentation vs two-stage detection  
✅ **Better accuracy** - ISNet specifically trained on anime characters  
✅ **Lower memory** - Lighter models available  
✅ **Transparent backgrounds** - Native RGBA output support  
✅ **Better for dense scenes** - Handles overlapping characters well  

## Troubleshooting

### ImportError: cannot import name 'AnimeSegmentation' from train.py
**Solution**: Make sure anime-segmentation repo is cloned in `objectfree/anime-segmentation/`

```bash
cd objectfree
git clone https://github.com/SkyTNT/anime-segmentation.git
```

### Model loading fails
**Solution**: Delete cache and re-download

```bash
rm -rf ~/.cache/huggingface/hub/models--skytnt--anime-seg*
```

### CUDA out of memory
**Solution**: Use smaller model or reduce batch size

```bash
# Use U2Net instead of ISNetIS
python anime_seg_detector.py --model u2net --img-size 512
```

### Poor detection quality
**Solution**: Adjust threshold or try different model

```bash
# Lower threshold for sensitive detection
python anime_seg_detector.py --threshold 0.3

# Try u2net if isnet_is is oversegmenting
python anime_seg_detector.py --model u2net
```

## Integration with Layout Generation Pipeline

The anime-segmentation detector can be integrated into the full layout generation pipeline:

```python
from objectfree.anime_seg_detector import AnimeSegmentationDetector

detector = AnimeSegmentationDetector(model_name="isnet_is", device="cuda:0")

# Get character boxes for layout analysis
boxes = detector.get_character_boxes(image_path)

# Extract character features for embedding
for box in boxes:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    # Pass to CLIP for feature extraction
```

## Future Improvements

- [ ] Batch processing for multiple images in parallel
- [ ] Fine-tuning on specific anime styles
- [ ] Ensemble detection with SAM3 fallback
- [ ] Real-time streaming inference
- [ ] Mobile/lightweight deployment

## References

- Repo: https://github.com/SkyTNT/anime-segmentation
- ISNet: https://github.com/xuebinqin/DIS
- Dataset: https://huggingface.co/datasets/skytnt/anime-segmentation

