# V11 "Sakuga" Selection Pipeline: Detailed Implementation Report

This report provides a granular analysis of the V11 implementation, specifying exact prompts, mathematical formulas, and architectural choices that define the pipeline.

## 1. Multi-Prompt CLIP-IQA System

To evaluate aesthetic quality, we move beyond single-prompt CLIP scoring to a **Multi-Prompt Ensemble**. For each attribute, we define 3 pairs of (Positive, Negative) synonyms.

### 1.1. Exact Prompts Used (`scripts/prepare_anime_attrs_v11.py`)

| Attribute | Prompt Group 1 (Pos / Neg) | Prompt Group 2 (Pos / Neg) | Prompt Group 3 (Pos / Neg) |
| :--- | :--- | :--- | :--- |
| **Sharpness** | "Sharp anime frame." / "Blurry anime frame." | "Crisp anime artwork." / "Fuzzy anime artwork." | "Clear anime image." / "Unclear anime image." |
| **Colorfulness** | "Vibrant anime colors." / "Dull anime colors." | "Colorful anime scene." / "Desaturated anime scene." | "Rich anime palette." / "Muted anime palette." |
| **Brightness** | "Well-lit anime scene." / "Dark anime scene." | "Bright anime frame." / "Dim anime frame." | "Good exposure anime." / "Underexposed anime." |
| **Sakuga** | "High sakuga animation frame." / "Low sakuga animation frame." | "Key animation frame." / "In-between animation frame." | "Fluid motion anime." / "Static anime frame." |
| **Cinematic** | "Cinematic anime shot." / "Plain anime shot." | "Well-composed anime." / "Poorly-composed anime." | "Professional anime framing." / "Amateur anime framing." |
| **Expression** | "Expressive anime face." / "Bland anime face." | "Emotional anime character." / "Neutral anime character." | "Dynamic anime expression." / "Static anime expression." |

### 1.2. Scoring Mechanism
For a given image $I$ and an attribute $A$ with $N=3$ prompt pairs $\{(p_i^+, p_i^-)\}_{i=1}^3$:
1.  **Embedding**: Encode image $I$ and all textual prompts using CLIP ViT-B/32.
2.  **Probability**: For each pair $i$, calculate softmax probability of positive prompt:
    $$ S_i = \frac{\exp(I \cdot p_i^+ \times 100)}{\exp(I \cdot p_i^+ \times 100) + \exp(I \cdot p_i^- \times 100)} $$
3.  **Ensemble**: Final score is the arithmetic mean: $S_{final} = \frac{1}{3} \sum_{i=1}^3 S_i$.

## 2. Neural Architecture: DSN V8 (Hybrid)

The backbone (`src/models/dsn_v8.py`) is a hybrid model designed to capture both global semantic context and local temporal smoothness.

### 2.1. Feature Input
*   **Visual Vector**: 512-dim CLIP ViT-B/32 embeddings (normalized).
*   **Attribute Vector**: 6-dim scores from the ensemble above.
*   **Total Input**: $x_t \in \mathbb{R}^{518}$.

### 2.2. Shared Encoder Stack
1.  **Input Projection**: Linear(518 $\to$ 256) + LayerNorm + GELU + Dropout(0.1).
2.  **Position Encoding**: Sinusoidal encoding added to features.
3.  **Transformer Encoder**:
    *   2 Layers of `TransformerEncoderLayer`.
    *   `d_model=256`, `nhead=4`, `dim_feedforward=1024`.
    *   *Role*: Global attention mechanism allowing each frame to attend to any other frame in the scene.
4.  **BiLSTM**:
    *   1 Layer `LSTM`, `input_size=256`, `hidden_size=128`, `bidirectional=True`.
    *   *Role*: Modeling local temporal transitions and motion continuity.
    *   **Output**: $h_t \in \mathbb{R}^{256}$ (concatenated forward [128] + backward [128]).

### 2.3. Dual-Head with Gating
Instead of a single scalar output, the model has two "brains" that it learns to balance:
*   **Rec Head**: `Linear(256->64) -> GELU -> Linear(64->1)`. Focus: Content representation.
*   **Anime Head**: Same architecture. Focus: Aesthetic quality.
*   **Gating Network**: `Linear(256->64) -> GELU -> Linear(64->1) -> Sigmoid`.
    *   Outputs $\alpha_t \in [0, 1]$ per frame.
*   **Final Fusion**:
    $$ \text{Logits}_t = \alpha_t \cdot \text{Rec}_t + (1 - \alpha_t) \cdot \text{Anime}_t $$

## 3. Training Loop: Simplified PPO (`train_rl_dsn_v11_simple.py`)

We utilize a simplified PPO approach to ensure stability.

### 3.1. Reward Function
The reward maximizes Mean Percentile Rank (MPR) while ensuring range diversity.

$$ R_{quality} = (\text{MPR} - 0.5) \times 6.0 $$
$$ R_{diversity} = (\text{DiversityScore} - 0.5) \times 2.0 $$
$$ R_{total} = R_{quality} + 0.3 \times R_{diversity} $$

*   **MPR**: For selected frames $\{t_1, \dots, t_K\}$, we calculate their percentile ranks in the scene's quality distribution and take the mean.
*   **Diversity Score**: $\min(1.0, \frac{\text{min\_gap}}{\text{expected\_gap}})$. If frames are clustered (small gap), this score drops.

### 3.2. PPO Loss
$$ L = L_{policy}^{clip} + 0.5 \cdot L_{value} - 0.02 \cdot H_{entropy} $$

*   **Clip Range**: $\epsilon = 0.2$ (standard stable value).
*   **Entropy Coeff**: 0.02 (encourages exploration early on).
*   **Optimizer**: AdamW with `lr=1e-4`, `weight_decay=1e-4`.
