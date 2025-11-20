# Training Advanced DSN Model - Quick Start Guide

## ✅ Setup Complete

Bạn đã có đầy đủ mọi thứ để train Advanced DSN model!

---

## 📁 Files Created

### Model Implementation
- ✅ `src/models/dsn_advanced.py` - Advanced DSN model (3.1M parameters)
- ✅ `src/models/README_DSN_ADVANCED.md` - Comprehensive documentation

### Training Pipeline
- ✅ `src/pipeline/train_rl_dsn.py` - Updated to support both baseline and advanced models

### Training Scripts
- ✅ `scripts/train_advanced_dsn.sh` - Full training script
- ✅ `scripts/ablation_study.sh` - Ablation study (5 configurations)
- ✅ `scripts/test_advanced_dsn.sh` - Quick test script

---

## 🚀 How to Train

### Option 1: Quick Test (1 epoch)
```bash
cd /home/serverai/ltdoanh/LayoutGeneration
./scripts/test_advanced_dsn.sh
```

### Option 2: Full Training (20 epochs)
```bash
cd /home/serverai/ltdoanh/LayoutGeneration
./scripts/train_advanced_dsn.sh
```

### Option 3: Ablation Study (5 configurations)
```bash
cd /home/serverai/ltdoanh/LayoutGeneration
./scripts/ablation_study.sh
```

### Option 4: Manual Command
```bash
conda activate sam

python -m src.pipeline.train_rl_dsn \
  --model_type advanced \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/dsn_advanced_v1 \
  --epochs 20 \
  --device cuda:0 \
  --num_attn_heads 4 \
  --num_attn_layers 2 \
  --num_scales 3 \
  --use_cache 1 \
  --cache_size 1000
```

---

## 🎛️ Key Arguments

### Model Selection
- `--model_type` - Choose `baseline` or `advanced`

### Advanced Model Hyperparameters
- `--num_attn_heads 4` - Number of attention heads (2, 4, 8)
- `--num_attn_layers 2` - Number of attention layers (1, 2, 3)
- `--num_scales 3` - Temporal scales (1, 2, 3, 4)
- `--use_cache 1` - Enable caching (0 or 1)
- `--cache_size 1000` - Max cache size
- `--pos_encoding_type sinusoidal` - Positional encoding type
- `--use_lstm_in_advanced 1` - Use LSTM in advanced model (for ablation)

### Standard Training
- `--epochs 20` - Number of training epochs
- `--lr 1e-4` - Learning rate
- `--device cuda:0` - GPU device
- `--budget_ratio 0.06` - Target selection ratio

---

## 📊 Monitor Training

### TensorBoard
```bash
# For full training
tensorboard --logdir runs/dsn_advanced_v1/

# For ablation study
tensorboard --logdir runs/ablation/
```

### Metrics to Watch

**Training Metrics:**
- `train/mean_reward` - Average reward (higher is better)
- `train/sel_ratio` - Selection ratio (should be ~0.06)
- `train/entropy` - Policy entropy (diversity)
- `train/mean_prob` - Average selection probability

**Cache Metrics (Advanced only):**
- `cache/hit_rate` - Cache hit rate (higher = faster)
- `cache/hits` - Number of cache hits
- `cache/size` - Current cache size

**Validation Metrics:**
- `val/RecErr_mean` - Reconstruction error (lower is better)
- `val/Diversity_mean` - Diversity score (higher is better)
- `val/MS_SWD_mean` - Multi-scale SWD (lower is better)

---

## 🔬 Ablation Study Configurations

The ablation study script runs 5 experiments:

1. **Baseline** - BiLSTM only (for comparison)
2. **Attention Only** - Self-attention, no LSTM, no multi-scale
3. **Multi-Scale Only** - Multi-scale, no attention
4. **Attention + Multi-Scale** - Both, no cache
5. **Full Advanced** - All features (attention + multi-scale + cache)

Results will be saved in `runs/ablation/` and can be compared in TensorBoard.

---

## 📈 Expected Results

### Model Comparison

| Model | Parameters | Training Time | Cache Hit Rate | Expected Performance |
|-------|------------|---------------|----------------|---------------------|
| Baseline | ~200K | 1x | N/A | Baseline |
| Advanced (no cache) | ~3.1M | ~1.5x | N/A | +5-10% better |
| Advanced (with cache) | ~3.1M | ~1.2x | 30-50% | +5-10% better |

### Performance Improvements (Expected)

- **Diversity**: +5-10% (better temporal coverage)
- **Representativeness**: +3-7% (smarter selection)
- **MS-SWD**: -10-20% (better perceptual quality)
- **Training Speed**: 1.2-1.5x with caching

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce model size
--enc_hidden 128 --num_attn_heads 2 --num_attn_layers 1

# Or disable cache
--use_cache 0
```

### Slow Training
```bash
# Enable cache
--use_cache 1 --cache_size 1000

# Reduce scales
--num_scales 2
```

### Cache Not Helping
```bash
# Check cache stats in TensorBoard (cache/hit_rate)
# If hit rate < 10%, dataset may not have repeated scenes
```

---

## 📝 Checkpoints

### Checkpoint Format

**Baseline:**
```python
{
    "encoder": encoder_state_dict,
    "policy": policy_state_dict
}
```

**Advanced:**
```python
{
    "model": model_state_dict,
    "config": DSNConfig(...),
    "model_type": "advanced"
}
```

### Loading Checkpoints

```python
# Load advanced model checkpoint
checkpoint = torch.load("runs/dsn_advanced_v1/dsn_checkpoint_ep10.pt")
config = checkpoint["config"]
model = DSNAdvanced(config)
model.load_state_dict(checkpoint["model"])
```

---

## 🎯 Next Steps

1. **Run Quick Test**
   ```bash
   ./scripts/test_advanced_dsn.sh
   ```

2. **Start Full Training**
   ```bash
   ./scripts/train_advanced_dsn.sh
   ```

3. **Monitor in TensorBoard**
   ```bash
   tensorboard --logdir runs/dsn_advanced_v1/
   ```

4. **Run Ablation Study** (for paper)
   ```bash
   ./scripts/ablation_study.sh
   ```

5. **Analyze Results**
   - Compare metrics in TensorBoard
   - Visualize selected keyframes
   - Compute cache speedup

---

## 📚 Documentation

- **Model Details**: `src/models/README_DSN_ADVANCED.md`
- **Implementation Plan**: `.gemini/antigravity/brain/.../implementation_plan.md`
- **Walkthrough**: `.gemini/antigravity/brain/.../walkthrough.md`

---

## ✨ Summary

Bạn đã có:
- ✅ Advanced DSN model với 3.1M parameters
- ✅ Temporal self-attention (4 heads, 2 layers)
- ✅ Multi-scale temporal modeling (3 scales)
- ✅ Feature caching (LRU, thread-safe)
- ✅ Training pipeline hoàn chỉnh
- ✅ Ablation study scripts
- ✅ Comprehensive documentation

**Sẵn sàng để train và publish paper! 🎉**
