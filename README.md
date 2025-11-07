# LayoutGeneration

🎬 **Unified Scene Detection + Keyframe Selection Pipeline**

A modular, production-ready pipeline for automatic keyframe extraction from videos using pluggable scene detection backends (PySceneDetect, TransNetV2) and distance metrics (LPIPS, DISTS).

---

## ✨ Features

- **🔍 Multiple Scene Detectors**
  - PySceneDetect (fast, CPU-based)
  - TransNetV2 (deep learning, GPU-accelerated)

- **📏 Distance Metrics**
  - LPIPS (perceptual loss)
  - DISTS (deep image similarity)

- **✂️ Keyframe Selection Strategies**
  - Medoid selection (optimal representatives)
  - Random sampling

- **🔄 Post-Processing**
  - Cosine similarity filtering for duplicate removal
  - Scene-based normalization and merging

- **📊 Comprehensive Evaluation**
  - Representativeness, Coverage, Diversity metrics
  - Image quality assessment (sharpness, exposure, noise)
  - Comparison with baselines (uniform, middle, motion-peak)

- **⚡ Batch Processing**
  - Parallel video processing with worker pools
  - Custom output directory specification
  - Grid search hyperparameter optimization

---

## 📋 Requirements

```
Python 3.8+
CUDA 11.0+ (optional, for GPU acceleration)
```

**Core Dependencies:**
```
opencv-python>=4.5.0
numpy>=1.19.0
torch>=1.9.0
torchvision>=0.10.0
tqdm>=4.62.0
pandas>=1.3.0
scenedetect[opencv]>=0.41.0
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ltdoanh2004/LayoutGeneration.git
cd LayoutGeneration

# Create conda environment
conda create -n layout python=3.10
conda activate layout

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
**! Run train optimizer for choosing parameter for pipeline**
```bash
python pipeline/train_optimizer.py \
  --samples_dir "data/samples/Sakuga" \
  --pipeline_script "scripts/pipeline_keyframes.py" \
  --param_config_json "configs/optimizer_grid.json" \
  --out_dir "outputs/optimizer_out" \
  --model_dir "src/models/TransNetV2" \
  --eval_backbone "resnet50" \
  --eval_device "cuda" \
  --max_videos 5 \
  --distance_backend dists \
  --dists_as_distance 1
```
**1️⃣ Single Video Processing**

```bash
# TransNetV2 + LPIPS
python pipeline.py \
  --video data/samples/Sakuga/42853.mp4 \
  --backend transnetv2  \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 5 --max_frames_per_scene 40 \
  --keyframes_per_scene 1 --nms_radius 2 \
  --resize_w 320 --resize_h 180 \
  --out_dir data/outputs/Dang/check/run_tv2_lpips
```
```bash
python pipeline.py \
  --video data/samples/Sakuga/42853.mp4 \
  --backend transnetv2  \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --distance_backend dists --dists_as_distance 1 \
  --sample_stride 5 --max_frames_per_scene 40 \
  --keyframes_per_scene 1 --nms_radius 2 \
  --resize_w 320 --resize_h 270 \
  --out_dir data/outputs/Dang/check/run_tv2_dists
```

**2️⃣ Evaluate Results**

```bash
python scipts/eval_keyframes.py \
  --video samples/vssum/v11.mp4 \
  --scenes_json outputs/run_tv2_lpips_v11/scenes.json \
  --keyframes_csv outputs/run_tv2_lpips_v11/keyframes.csv \
  --out_dir outputs/eval_tv2_lpips_v11 \
  --backbone resnet50 \
  --device cuda \
  --sample_stride 1 \
  --max_frames_eval 200 \
  --tau 0.5 \
  --with_baselines
```

**3️⃣ Batch Processing Multiple Videos**

```bash
python batch_processing.py \
  --run_pipeline \
  --run_evalviz \
  --data_folder samples \
  --distance_backend lpips \
  --prob_threshold 0.5 \
  --sample_stride 5 \
  --nms_radius 2 \
  --resize_w 320 --resize_h 270 \
  --pipeline_out_dir outputs/pipeline \
  --eval_out_dir outputs/eval \
  --num_workers 4
```

**4️⃣ Hyperparameter Optimization**

```bash
python -m pipeline.train_optimizer \
  --samples_dir samples \
  --pipeline_script pipeline.py \
  --model_dir src/models/TransNetV2 \
  --param_config_json pipeline/config/optimizer_grid.json \
  --out_dir outputs/optimizer_results_dists \
  --distance_backend dists \
  --dists_as_distance 1 \
  --eval_backbone resnet50 \
  --eval_device cuda \
  --max_videos 2 \
  --early_stopping_patience 5
```

**5️⃣ Quality Analysis**

```bash
python outputs/outputs_eval/Evaluation_All.py \
  --eval_dir outputs_eval \
  --output_report quality_report.txt \
  --output_json detailed_results.json \
  --output_csv summary_results.csv
```

---

## 📂 Project Structure

```
LayoutGeneration/
├── src/                          # Core modules
│   ├── scene_detection/          # Scene detectors (PySceneDetect, TransNetV2)
│   ├── distance_selector/        # Distance metrics (LPIPS, DISTS)
│   ├── keyframe/                 # Keyframe selectors (Medoid, Random)
│   ├── models/                   # Pre-trained models
│   ├── evaluation/               # Evaluation & metrics
│   └── utils/                    # Utilities (video I/O, timecode, etc.)
│
├── pipeline/                     # Pipeline code and configs
│   ├── pipeline_new.py           # Main pipeline (with cosine filtering)
│   ├── train_optimizer.py        # Hyperparameter optimization
│   └── config/
│       └── optimizer_grid.json   # Hyperparameter grid
│
├── eval/                         # Evaluation modules
│   ├── evaluator.py
│   ├── metrics.py
│   └── visualize/
│
├── scripts/                      # CLI entry points
│   ├── pipeline.py               # Pipeline wrapper
│   ├── batch_processing.py       # Batch video processing
│   ├── eval_keyframes.py         # Evaluation script
│   └── scene_detection_pipeline.py
│
├── examples/                     # Example scripts
│   ├── grid_layout.py
│   ├── keyframe.py
│   └── summary_metrics.py
│
├── data/                         # Data folder (videos, results)
│   ├── samples/                  # Sample videos
│   └── outputs/                  # Results
│
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## 🔧 Configuration

### Scene Detection Backends

**PySceneDetect**
```bash
--backend pyscenedetect \
--threshold 27.0
```

**TransNetV2**
```bash
--backend transnetv2 \
--model_dir src/models/TransNetV2 \
--prob_threshold 0.5 \
--scene_device cuda
```

### Distance Metrics

**LPIPS (Perceptual Loss)**
```bash
--distance_backend lpips \
--lpips_net alex              # alex | vgg | squeeze
```

**DISTS (Deep Image Similarity)**
```bash
--distance_backend dists \
--dists_as_distance 1         # 1 = use as distance, 0 = as similarity
```

### Keyframe Selection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sample_stride` | 10 | Sample every N frames within scene |
| `--max_frames_per_scene` | 30 | Max frames to consider per scene |
| `--keyframes_per_scene` | 1 | Number of keyframes per scene |
| `--nms_radius` | 3 | NMS radius for multiple keyframes |
| `--resize_w`, `--resize_h` | 320, 180 | Frame resize for distance computation |

### Post-Processing

**Duplicate Keyframe Filtering**
```bash
--filter_duplicate_keyframes \
--cosine_similarity_threshold 0.9    # 0-1, higher = stricter filtering
```

---

## 📊 Outputs

### Pipeline Outputs

```
outputs/run_tv2_lpips_v11/
├── scenes.json                  # Scene boundaries and metadata
├── scenes.csv
├── keyframes.csv                # Selected keyframe indices
└── keyframes/                   # Exported keyframe images
    ├── scene_0000_frame_00000042.jpg
    ├── scene_0001_frame_00000123.jpg
    └── ...
```

### Evaluation Outputs

```
outputs/eval_tv2_lpips_v11/
├── eval_results.json            # Evaluation metrics
├── eval_results.csv
└── visualizations/              # (optional)
    ├── distance_matrix.png
    └── keyframe_grid.jpg
```

### Quality Report

```
quality_report.txt              # Human-readable report
detailed_results.json           # All metrics in JSON
summary_results.csv            # CSV for spreadsheet analysis
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Test
```bash
# Test pipeline on one video
python pipeline.py \
  --video samples/v1.mp4 \
  --backend pyscenedetect \
  --distance_backend lpips \
  --out_dir outputs/test

# Evaluate it
python eval_keyframes.py \
  --video samples/v1.mp4 \
  --scenes_json outputs/test_v1/scenes.json \
  --keyframes_csv outputs/test_v1/keyframes.csv \
  --out_dir outputs/eval_test
```

### Workflow 2: Compare Backends (LPIPS vs DISTS)
```bash
# Run with LPIPS
python batch_processing.py \
  --data_folder samples \
  --distance_backend lpips \
  --num_workers 4 \
  --pipeline_out_dir outputs/lpips/pipeline \
  --eval_out_dir outputs/lpips/eval

# Run with DISTS
python batch_processing.py \
  --data_folder samples \
  --distance_backend dists \
  --num_workers 4 \
  --pipeline_out_dir outputs/dists/pipeline \
  --eval_out_dir outputs/dists/eval

# Analyze quality
python outputs/outputs_eval/Evaluation_All.py
```

### Workflow 3: Optimize Hyperparameters
```bash
# Grid search with early stopping
python -m pipeline.train_optimizer \
  --samples_dir samples \
  --distance_backend dists \
  --early_stopping_patience 5 \
  --max_videos 2

# Use best params for batch processing
python batch_processing.py \
  --prob_threshold 0.5 \
  --sample_stride 5 \
  --nms_radius 2 \
  --resize_w 320 \
  --resize_h 270 \
  --num_workers 4
```

---

## 📈 Evaluation Metrics

### Representativeness (30% weight)
- **RecErr**: Reconstruction error (lower is better)
- **Frechet**: Frechet distance to full video (lower is better)

### Coverage (30% weight)
- **SceneCoverage**: Scene coverage ratio (higher is better)
- **TemporalCoverage@tau**: Temporal coverage with threshold (higher is better)

### Diversity (20% weight)
- **RedundancyMeanCos**: Mean cosine similarity (lower is better)
- **MinPairwiseDist**: Minimum pairwise distance (higher is better)

### Image Quality (20% weight)
- **Sharpness_med**: Median sharpness (higher is better)
- **Exposure_med**: Median exposure level (optimal: 88-168)
- **Noise_med**: Median noise level (lower is better)

---

## 🐛 Troubleshooting

**Q: GPU memory error**
```bash
# Reduce batch size or frame samples
--batch_pairs 8 \
--max_frames_per_scene 20 \
--distance_device cpu  # Use CPU instead
```

**Q: TransNetV2 model not found**
```bash
# Ensure model_dir exists and has weights/
ls src/models/TransNetV2/weights/
# Should contain: transnetv2.pth or similar
```

**Q: Slow evaluation**
```bash
# Reduce evaluation frames
--sample_stride 15 \
--max_frames_eval 100
```

**Q: Pipeline outputs not in expected location**
```bash
# Pipeline auto-appends video name to --out_dir
# If --out_dir is "outputs/test"
# Actual output is "outputs/test_videoname"
```

---

## 📚 References

- **TransNetV2**: [Paper](https://arxiv.org/abs/2003.13678)
- **LPIPS**: [Paper](https://arxiv.org/abs/1801.03924)
- **DISTS**: [Paper](https://arxiv.org/abs/2104.02935)
- **PySceneDetect**: [GitHub](https://github.com/Breakthrough/PySceneDetect)

---

## 📄 License

MIT License

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

## 📞 Contact & Support

For issues, questions, or suggestions, please open an issue on GitHub.
