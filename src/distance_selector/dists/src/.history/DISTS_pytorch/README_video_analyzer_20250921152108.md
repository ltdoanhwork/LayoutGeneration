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

## What it does:

1. **Extracts frames** from the input video using OpenCV
2. **Uses the first frame as reference** for all comparisons
3. **Calculates DISTS scores** between the reference frame and each subsequent frame
4. **Plots the results** showing how the video quality changes over time
5. **Shows statistics** including mean, standard deviation, min and max scores

## Output:

- **Interactive plot** showing DISTS scores over frame numbers
- **Console statistics** with summary information
- **Optional saved plot** if --output is specified

## DISTS Score Interpretation:

- **Higher scores** indicate better perceptual similarity to the reference frame
- **Lower scores** indicate more perceptual difference from the reference frame
- **Score range** is typically between 0 and 1, where 1 means identical

## Example Output:
```
Video info: 300 frames, 30.0 FPS
Extracted 300 frames total
Calculating DISTS scores...
Processed 299/299 frame comparisons

Analysis complete!
Processed 300 frames
Average DISTS score: 0.7543
Score range: 0.2156 - 0.9876
```