# DSN Model Evaluation Scripts

This folder contains batch evaluation scripts for all trained DSN model variants. Each evaluation script corresponds to a specific training configuration and uses the unified **transnetv2** backend for scene detection.

## 📋 Overview

All evaluation scripts follow the same structure:
1. Load pre-trained checkpoint from the corresponding training run
2. Run batch evaluation on test videos using `eval/batch_eval.py`
3. Generate comprehensive metrics including:
   - Reconstruction Error (RecErr)
   - Frechet Distance
   - Scene Coverage
   - Temporal Coverage
   - LPIPS Perceptual Gap & Diversity
   - MS-SWD Color Distance

## 🎯 Model Tracks

### **Track A: Anime-CLIP-IQA Features** (`eval_track_a_features.sh`)
- **Training Script**: `train_track_a.sh`
- **Key Features**: Concatenates Anime-CLIP-IQA attributes (6-dim) to CLIP features
- **Total Feature Dim**: 518 (512 CLIP + 6 Anime)
- **Checkpoint**: `runs/dsn_track_a_features/dsn_checkpoint_ep20.pt`
- **Output**: `runs/eval_track_a_features/`

### **Track B: Anime-CLIP-IQA Rewards** (`eval_track_b_rewards.sh`)
- **Training Script**: `train_track_b.sh`
- **Key Features**: Uses Anime-CLIP-IQA for reward computation only (not as features)
- **Feature Dim**: 512 (CLIP only)
- **Reward Weights**: Look=0.5, Sakuga=0.7, Story=0.0
- **Checkpoint**: `runs/dsn_track_b_rewards/dsn_checkpoint_ep20.pt`
- **Output**: `runs/eval_track_b_rewards/`

### **Track C: Combined Anime-CLIP-IQA** (`eval_track_c_combined.sh`)
- **Training Script**: `train_track_c.sh`
- **Key Features**: Both Anime-CLIP-IQA features AND rewards
- **Total Feature Dim**: 518 (512 CLIP + 6 Anime)
- **Reward Weights**: Look=0.5, Sakuga=0.7, Story=0.2
- **Checkpoint**: `runs/dsn_track_c_combined/dsn_checkpoint_ep20.pt`
- **Output**: `runs/eval_track_c_combined/`

### **Track D: Anime Emphasis + Prob Separation** (`eval_track_d_anime.sh`)
- **Training Script**: `train_track_d_anime.sh`
- **Key Features**: Advanced anime emphasis rewards + probability separation
- **Total Feature Dim**: 518 (512 CLIP + 6 Anime)
- **New Rewards**: W_ANIME=0.2, W_PROBSEP=0.1
- **Entropy Coef**: 0.005 (reduced for sharper probabilities)
- **Checkpoint**: `runs/dsn_track_d_anime/dsn_checkpoint_ep20.pt`
- **Output**: `runs/eval_track_d_anime/`

### **Baseline DSN** (`eval_baseline.sh`)
- **Training Script**: `train_baseline.sh`
- **Key Features**: Simple baseline without advanced features
- **Feature Dim**: 512 (CLIP only)
- **No motion, no anime attributes**
- **Checkpoint**: `runs/dsn_runs_baseline_100_samples/baseline_v1/dsn_checkpoint_ep20.pt`
- **Output**: `runs/eval_baseline/`

### **Advanced DSN (no motion)** (`eval_advanced.sh`)
- **Training Script**: `train_advanced_dsn.sh`
- **Key Features**: Advanced model with attention mechanisms but no motion
- **Feature Dim**: 512 (CLIP only)
- **Advanced Architecture**: Multi-head attention, multi-scale features, positional encoding
- **Checkpoint**: `runs/dsn_advanced_v1_no_motion_100_samples_test_sakura/dsn_checkpoint_ep20.pt`
- **Output**: `runs/eval_advanced/`

### **DSN with RAFT Motion** (`eval_raft_motion.sh`)
- **Training Script**: `train_dsn_with_raft_motion.sh`
- **Key Features**: Advanced DSN + RAFT optical flow motion features (128-dim)
- **Feature Dim**: 512 (CLIP) + 128 (RAFT motion) via cross-attention fusion
- **Checkpoint**: `runs/dsn_raft_motion/dsn_checkpoint_ep20.pt`
- **Output**: `runs/eval_raft_motion/`

## ⚙️ Unified Configuration

All evaluation scripts use the following **standardized settings**:

### Scene Detection
- **Backend**: `transnetv2` (unified across all scripts)
- **Sample Stride**: 5
- **Resize**: 320x180

### Feature Extraction
- **Embedder**: `clip_vitb32`
- **Device**: CUDA

### Evaluation Metrics
- **Evaluation Backbone**: ResNet50
- **Sample Stride**: 1
- **Max Frames**: 200
- **Tau (Temporal Coverage)**: 0.5
- **With Baselines**: Enabled

### Budget Constraints
- **Budget Ratio**: 0.06
- **Bmin**: 3 frames
- **Bmax**: 15 frames

## 🚀 Usage

### Run Single Track Evaluation

```bash
# Track A: Anime-CLIP-IQA Features
bash scripts/eval_bash_script/eval_track_a_features.sh

# Track B: Anime-CLIP-IQA Rewards
bash scripts/eval_bash_script/eval_track_b_rewards.sh

# Track C: Combined
bash scripts/eval_bash_script/eval_track_c_combined.sh

# Track D: Anime Emphasis + Prob Sep
bash scripts/eval_bash_script/eval_track_d_anime.sh

# Baseline
bash scripts/eval_bash_script/eval_baseline.sh

# Advanced (no motion)
bash scripts/eval_bash_script/eval_advanced.sh

# RAFT Motion
bash scripts/eval_bash_script/eval_raft_motion.sh
```

### Run All Evaluations

```bash
# Evaluate all 7 model variants
for script in scripts/eval_bash_script/*.sh; do
    echo "Running $script"
    bash "$script"
done
```

### Customize Checkpoint Path

Edit the checkpoint path in each script before running:

```bash
# Example: Evaluate a specific epoch
CHECKPOINT_PATH="${CHECKPOINT_DIR}/dsn_checkpoint_ep15.pt"  # Change from ep20 to ep15
```

### Limit Number of Test Videos

By default, scripts evaluate up to 30 videos. To change this:

```bash
# Edit MAX_VIDEOS in the script
MAX_VIDEOS=10  # Evaluate only 10 videos for faster testing
```

## 📊 Output Structure

Each evaluation generates:

```
runs/eval_<track_name>/
├── pipeline_results/          # DSN keyframe extraction results
│   ├── video1/
│   │   ├── scenes.json       # Scene boundaries
│   │   └── keyframes.csv     # Selected keyframes
│   └── ...
├── eval_results/             # Evaluation metrics
│   ├── video1/
│   │   ├── eval_results.json # Base metrics
│   │   └── extra_metrics.json # LPIPS & MS-SWD
│   └── ...
└── summary_results.json      # Aggregated metrics across all videos
```

## 📈 View Results

### View Aggregated Metrics

```bash
# Pretty print summary metrics
cat runs/eval_track_a_features/summary_results.json | jq '.aggregate_metrics'
```

Example output:
```json
{
  "RecErr_mean": 0.234,
  "Frechet_mean": 1.456,
  "SceneCoverage_mean": 0.892,
  "TemporalCoverage@tau_mean": 0.745,
  "LPIPS_PerceptualGap_mean": 0.123,
  "LPIPS_DiversitySel_mean": 0.345,
  "MS_SWD_Color_mean": 2.678
}
```

### Compare Tracks

```bash
# Quick comparison across all tracks
for dir in runs/eval_*/; do
    echo "=== $(basename $dir) ==="
    jq '.aggregate_metrics.RecErr_mean' "$dir/summary_results.json"
done
```

## 🔧 Troubleshooting

### Checkpoint Not Found
If you see "WARNING: Checkpoint not found", either:
1. Train the model first using the corresponding training script
2. Adjust the `CHECKPOINT_PATH` variable in the eval script

### Anime Attributes Required
For Tracks A, C, and D, ensure Anime-CLIP-IQA attributes are precomputed:

```bash
python scripts/prepare_anime_attrs.py \
  --dataset_root data/sakuga_dataset_100_samples \
  --device cuda
```

### RAFT Motion Features Required
For the RAFT Motion track, ensure motion features are precomputed:

```bash
python scripts/precompute_raft_motion.py \
  --dataset_root data/sakuga_dataset_100_samples \
  --raft_model repos/RAFT/models/raft-small.pth \
  --device cuda
```

## 📝 Notes

1. **Backend Unification**: All scripts now use `transnetv2` for consistent scene detection across training and evaluation
2. **Checkpoint Compatibility**: Ensure the checkpoint was trained with matching hyperparameters (feat_dim, enc_hidden, etc.)
3. **GPU Memory**: Evaluation requires ~4-8GB GPU memory depending on video length
4. **Parallel Evaluation**: For faster evaluation, consider running multiple scripts in parallel on different GPUs by modifying the `DEVICE` variable

## 🔗 Related Files

- **Training Scripts**: `scripts/dsn_bash_script/train_*.sh`
- **Batch Eval Pipeline**: `eval/batch_eval.py`
- **DSN Pipeline**: `eval/run_dsn_pipeline.py`
- **Metrics Computation**: `scripts/eval_keyframes.py`, `eval/extra_metrics.py`
