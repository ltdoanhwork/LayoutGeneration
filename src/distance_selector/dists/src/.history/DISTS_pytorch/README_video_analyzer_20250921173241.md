# Video DISTS Analyzer - Usage Examples

## Installation
First, install the required dependencies:
```bash
pip install -r video_requirements.txt
```

## Basic Usage

### Analyze a video file:
```bash
python video_dists_analyzer.py --video path/to/your/video.mp4
```

### Analyze only first 100 frames:
```bash
python video_dists_analyzer.py --video path/to/your/video.mp4 --max_frames 100
```

### Save the plot to file:
```bash
python video_dists_analyzer.py --video path/to/your/video.mp4 --output results_plot.png
```

### Force CPU usage (if you have GPU but want to use CPU):
```bash
python video_dists_analyzer.py --video path/to/your/video.mp4 --device cpu
```

### Detect keyframes and save them to file:
```bash
python video_dists_analyzer.py --video path/to/your/video.mp4 --save_keyframes
```

### Save keyframe images to disk:
```bash
python video_dists_analyzer.py --video path/to/your/video.mp4 --save_images
```

### Limit maximum keyframes (default: 100):
```bash
python video_dists_analyzer.py --video path/to/your/video.mp4 --max_keyframes 50
```

### Save both text file and images with custom limit:
```bash
python video_dists_analyzer.py --video path/to/your/video.mp4 --save_keyframes --save_images --max_keyframes 100
```

### Create timeline visualization (grid of all keyframes):
```bash
python video_dists_analyzer.py --video path/to/your/video.mp4 --create_timeline
```

### Customize timeline layout:
```bash
# 15 columns, larger thumbnails
python video_dists_analyzer.py --video path/to/your/video.mp4 --create_timeline --timeline_cols 15 --thumb_size 160 120

# Dense layout with smaller thumbnails
python video_dists_analyzer.py --video path/to/your/video.mp4 --create_timeline --timeline_cols 25 --thumb_size 100 60
```

### Adjust keyframe detection sensitivity:
```bash
# More selective (fewer keyframes)
python video_dists_analyzer.py --video path/to/your/video.mp4 --prominence 0.1 --distance 10

# Less selective (more keyframes)
python video_dists_analyzer.py --video path/to/your/video.mp4 --prominence 0.02 --distance 3
```

## What it does:

1. **Extracts frames** from the input video using OpenCV
2. **Uses the first frame as reference** for all comparisons
3. **Calculates DISTS scores** between the reference frame and each subsequent frame
4. **Detects keyframes** by finding significant peaks and valleys in the DISTS scores
5. **Plots the results** showing how the video quality changes over time with keyframes highlighted
6. **Shows statistics** including mean, standard deviation, min and max scores
7. **Exports keyframe list** for further processing or video editing
8. **Saves keyframe images** to disk with proper naming convention
9. **Limits keyframes** to a maximum number (default: 100) by selecting the most prominent ones
10. **Creates timeline visualization** showing all keyframes in chronological order with timestamps and scores

## Output:

- **Interactive plot** showing DISTS scores over frame numbers with keyframes highlighted
- **Console statistics** with summary information and keyframe details
- **Optional saved plot** if --output is specified
- **Keyframes text file** with detailed information about detected keyframes
- **Keyframe images folder** with individual frame images saved as JPEG files
- **Timeline visualization** showing all keyframes in a grid layout with timestamps and scores

## DISTS Score Interpretation:

- **Higher scores** indicate better perceptual similarity to the reference frame
- **Lower scores** indicate more perceptual difference from the reference frame
- **Score range** is typically between 0 and 1, where 1 means identical

## Keyframe Detection:

**Peaks**: Frames where DISTS score suddenly increases (high similarity after low similarity)
**Valleys**: Frames where DISTS score suddenly decreases (low similarity after high similarity)

These represent significant changes in video content and can serve as keyframes for:
- Video summarization
- Scene change detection
- Thumbnail generation
- Video editing markers

## Parameters:

- **--prominence**: Controls sensitivity (0.02-0.2, default: 0.05)
  - Lower values = more keyframes detected
  - Higher values = fewer, more significant keyframes
- **--distance**: Minimum frames between keyframes (default: 5)
- **--max_keyframes**: Maximum number of keyframes to detect (default: 100)
- **--save_images**: Save keyframe images to disk
- **--save_keyframes**: Save keyframe information to text file

## Keyframe Image Naming:
Images are saved with descriptive filenames:
- `keyframe_0156_peak_score0.845.jpg`
- `keyframe_0203_valley_score0.234.jpg`

## Example Output:
```
Video info: 300 frames, 30.0 FPS
Extracted 300 frames total
Calculating DISTS scores...
Processed 299/299 frame comparisons
Detecting keyframes...

Analysis complete!
Processed 300 frames
Average DISTS score: 0.7543
Score range: 0.2156 - 0.9876

Keyframes detected:
  Total keyframes: 12
  Peaks: 6
  Valleys: 6

Keyframe details:
  Frame  15: peak   (score: 0.8234, prominence: 0.0856)
  Frame  42: valley (score: 0.3421, prominence: 0.1234)
  Frame  78: peak   (score: 0.7891, prominence: 0.0945)
  ...

Keyframe list: [15, 42, 78, 125, 156, 189, 203, 234, 267, 289, 295, 298]
```