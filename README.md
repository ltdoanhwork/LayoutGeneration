# Anime Layout Generation 🎬✨

A unified framework for **Anime Video Summarization** and **Manga-Style Layout Generation**. This project implements a sophisticated pipeline that detects scenes, selects keyframes using Reinforcement Learning (RL) or LLM-based methods, and composes them into aesthetically pleasing layouts (Voronoi SoftCollage or Rectangular Grids).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![TransNetV2](https://img.shields.io/badge/Scene_Detection-TransNetV2-green)

---

## 🌟 Key Features

- **Advanced Scene Detection**: Utilizes **TransNetV2** for state-of-the-art shot boundary detection in anime videos.
- **Intelligent Keyframe Selection**:
  - **V11 (Proposed)**: Deep Summarization Network (DSN) trained with **RL and Lagrangian Constraints** to balance Representativeness, Diversity, and Aesthetic Quality.
  - **LLMVS**: LLM-based Video Summarization using Llama-2 embeddings for semantic understanding.
  - **VSUMM**: Standard visual summarization baseline.
- **Dynamic Layout Generation**:
  - **SoftCollage**: Organic, Voronoi-based layouts that adapt to image content.
  - **Grid Layout**: Clean, rectangular panel layouts optimizing screen space.
  - **Temporal Coherence**: Maintains story flow by clustering temporally adjacent frames.
- **Aesthetic Quality Assessment**: Integrated "Anime Attributes" scoring (Sharpness, Noise, Exposure, Colorfulness).

---

## 🛠️ Installation

### Prerequisites
- Linux / macOS
- Python 3.8+
- CUDA-compatible GPU (Recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/ltdoanh2004/LayoutGeneration.git
cd LayoutGeneration

# Create a virtual environment (optional but recommended)
conda create -n layout_gen python=3.10
conda activate layout_gen

# Install dependencies
pip install -r requirements.txt
```

> **Note**: This project relies on `repos/Colla` for the SoftCollage layout engine. Ensure the submodule is initialized if applicable, or the code is present in `repos/Colla`.

---

## 🚀 Usage

### 1. Single Video Comparison (V11 vs VSUMM)
Run a quick demo to compare the proposed V11 model against the VSUMM baseline on a single video.

```bash
python scripts/demo_single_video.py \
  --video data/samples/Sakuga/13926.mp4 \
  --budget_ratio 0.1 \
  --output demo_results.json \
  --output_dir demo_vis \
  --stride 5
```

### 2. Full Layout Generation Pipeline
Generate a complete layout (Keyframes -> Layout) for a video.

```bash
# Standard Voronoi Layout
python layout_decomposer_final.py \
  --video data/samples/Sakuga/13926.mp4 \
  --out_dir outputs/layout_result \
  --backend transnetv2 \
  --no_grid_layout

# Rectangular Grid Layout
python layout_decomposer_final.py \
  --video data/samples/Sakuga/13926.mp4 \
  --out_dir outputs/grid_result \
  --use_grid_layout
```

### 3. Training V11 (RL with Constraints)
Train the V11 model using the Lagrangian Constrained Optimization approach.

```bash
# Updates model to satisfy RecErr < 0.35 while maximizing aesthetic reward
./scripts/run_v11_constrained.sh 0  # 0 is the GPU ID
```

---

## 📂 Project Structure

```
LayoutGeneration/
├── src/                        # Core source code
│   ├── models/                 # V11, DSN, TransNetV2 definitions
│   ├── scene_detection/        # Detection backends (TransNetV2, PySceneDetect)
│   ├── scoring/                # Scoring metrics (CLIP, IQA, Temporal)
│   └── pipeline/               # Training pipelines (RL, Lagrangian)
├── scripts/                    # Utility scripts and demos
│   ├── demo_single_video.py    # V11 vs VSUMM comparison
│   └── run_v11_constrained.sh  # Training script for V11
├── layout_decomposer_final.py  # Main entry point for layout generation
├── temporal_layout_composer_unified.py # Unified layout logic
├── ablation/                   # Ablation studies and baselines
│   ├── LLMVS/                  # LLM-based summarization model
│   └── pytorch-vsumm-reinforce # VSUMM baseline
├── repos/                      # External repositories
│   └── Colla/                  # SoftCollage implementation
└── data/                       # Datasets and samples
```

---

## � Models & Methods

### V11: Constrained RL Summarization
Proposed method that treats video summarization as a **Constrained Markov Decision Process (CMDP)**.
- **Objective**: Maximize Aesthetic Quality ($Q_{anm}$) and Diversity.
- **Constraints**: Reconstruction Error ($RecErr < \epsilon$) and Coverage.
- **Training**: Uses PPO with Lagrangian multipliers to dynamically adjust reward weights.

### LLMVS (Ablation)
Uses **Llama-2** embeddings to capture high-level semantic info.
- **Architecture**: Transformer Encoder + MLP Head on top of Llama embeddings.
- **Input**: User prompt embedding + Video generation embedding.

### Layout Engine
- **Hierarchical Slicing**: Decomposes the canvas into a binary tree of panels.
- **Optimization**: Uses Simulated Annealing (SAS) to optimize panel shapes and positions based on image content and saliency.

---

## � Evaluation & Metrics

The project uses a comprehensive set of metrics:
- **RecErr**: Reconstruction Error (pixel-level representativeness).
- **Frechet Distance**: Feature-level distribution similarity.
- **Coverage**: Temporal coverage of scenes.
- **Diversity**: Dissimilarity among selected keyframes.
- **MPR (Mean Percentile Rank)**: Aesthetic quality ranking relative to the full video.

---

## � License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.

---

## 📞 Contact

For questions or support, please open an issue in the repository.