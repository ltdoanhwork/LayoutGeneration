# Anime Layout Generation 🎬✨

A unified framework for **Anime Video Summarization** and **Manga-Style Layout Generation**. This project detects scenes, selects keyframes with the V11 DSN (RL + constraints), and composes **Voronoi** layouts via **CAST**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![TransNetV2](https://img.shields.io/badge/Scene_Detection-TransNetV2-green)

---

## 🌟 Key Features

- **Scene detection** with TransNetV2; **V11 keyframe selection** (representativeness, diversity, aesthetics).
- **CAST** layout: Voronoi SoftCollage, optional ISNet masking (`--voronoi-layout --run-isnet`).

---

## 🛠️ Installation

### Prerequisites

- Linux / macOS
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
git clone https://github.com/ltdoanh2004/LayoutGeneration.git
cd LayoutGeneration

conda create -n layout_gen python=3.10
conda activate layout_gen

pip install -r requirements.txt
```

Install **CAST** dependencies from `CAST/requirements.txt` when running layout (phase 2).

---

## 🚀 Usage — two-phase pipeline

Run **phase 1** from the repo root (`LayoutGeneration/`). Run **phase 2** after `cd CAST`.

### Phase 1 — Extract keyframes from video

```bash
python3 -m scripts.run_inference_v11 \
  --video_path /home/serverai/ltdoanh/LayoutGeneration/data/FINAL/t1.mp4 \
  --checkpoint runs/training_v11_final_new/best.pt \
  --output_dir /home/serverai/ltdoanh/LayoutGeneration/data/FINAL/keyframe/t1 \
  --budget_ratio 0.1 \
  --stride 10 \
  --save_images \
  --scene_threshold 0.9 \
  --min_scene_len 100
```

### Phase 2 — Create layout (CAST)

```bash
cd CAST

python run.py \
  /home/serverai/ltdoanh/LayoutGeneration/data/FINAL/layout/input/nobody_layout.jpg \
  /home/serverai/ltdoanh/LayoutGeneration/data/FINAL/keyframe/Nobody/Nobody/keyframes_filter \
  ./tests/output_nobody_new 2 --voronoi-layout --run-isnet
```

---

## 📂 Project structure (high level)

```
LayoutGeneration/
├── scripts/
│   └── run_inference_v11.py   # Phase 1: keyframes + V11 inference
├── CAST/
│   └── run.py                 # Phase 2: Voronoi layout
├── src/                       # Models, scene detection, training code
├── data/                      # Your videos, keyframes, layout inputs
└── requirements.txt
```

---

## 📚 Models & methods

- **V11**: Constrained RL summarization (aesthetic + diversity, reconstruction-style constraints).
- **CAST**: Hierarchical slicing + optimization for panel layout; Voronoi / ISNet integration via CLI flags.

---

## 🧠 CAST Optimize Objective (paper-style summary)

In Voronoi mode, CAST optimizes a weighted multi-objective loss:

$$
\mathcal{L}_{total} = \lambda_{cap}\mathcal{L}_{cap} + \lambda_{asp}\mathcal{L}_{asp} + \lambda_{ov}\mathcal{L}_{ov}
$$

Current default weights:

$$
(\lambda_{cap},\lambda_{asp},\lambda_{ov}) = (400, 600, 1500)
$$

### 1) Capacity loss $\mathcal{L}_{cap}$

Aligns each cell area with target area ratio derived from per-frame bbox area in `summary.json`:

$$
\mathcal{L}_{cap} = \frac{1}{N}\sum_i (A_i - \hat{A}_i)^2
$$

This stabilizes global size allocation (large-object frames get larger cells).

### 2) Aspect loss $\mathcal{L}_{asp}$

Aligns cell elongation with target bbox aspect ratio:

$$
\mathcal{L}_{asp} = \frac{1}{N}\sum_i (\log r_i - \log r_i^*)^2
$$

Log-space error makes penalties symmetric for over-wide and over-tall distortions.

### 3) Overlap-retention loss $\mathcal{L}_{ov}$

Maximizes normalized overlap between soft cell assignment and bbox prior:

$$
\mathcal{L}_{ov} = \frac{1}{N}\sum_i (1-o_i)^2
$$

This is the core content-preservation term, therefore assigned the largest weight.

### Why these lambda values?

1. Semantic retention first: $\lambda_{ov}$ is highest to reduce foreground/object truncation.
2. Geometric readability second: $\lambda_{asp} > \lambda_{cap}$ because aspect distortion is visually more harmful than mild area drift.
3. Gradient scale balancing: early iterations are tuned so the three terms are comparable in gradient magnitude, then biased toward retention.

### Practical retuning ranges

- $\lambda_{ov} \in [1000, 2500]$
- $\lambda_{asp} \in [300, 900]$
- $\lambda_{cap} \in [200, 800]$

Quick heuristic:
- Increase $\lambda_{ov}$ if objects are frequently cut.
- Increase $\lambda_{asp}$ if cells are shape-distorted.
- Increase $\lambda_{cap}$ if large/small frame size allocation drifts.

For full derivations, implementation context, and caveats (soft-to-hard gap, post-optimization spatial reassignment), see `CAST/PIPELINE_ANALYSIS_VI.md`.

---

## 📊 Evaluation & metrics

Representativeness (e.g. reconstruction error), diversity, temporal coverage, and aesthetic-style scores are used during training and analysis; see paper / code under `src/` for definitions.

---

## 📄 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome — open an issue or a pull request.

---

## 📞 Contact

For questions, open an issue in the repository.
