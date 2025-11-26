# Model Type Compatibility Update Summary

## ✅ Files Updated

### 1. **src/pipeline/train_rl_dsn.py**
**Changes:**
- ✅ Added `--model_type` argument (baseline/advanced)
- ✅ Added advanced model hyperparameters:
  - `--num_attn_heads` (default: 4)
  - `--num_attn_layers` (default: 2)
  - `--num_scales` (default: 3)
  - `--use_cache` (default: 1)
  - `--cache_size` (default: 1000)
  - `--pos_encoding_type` (default: sinusoidal)
  - `--use_lstm_in_advanced` (default: 1)
- ✅ Model initialization supports both types
- ✅ Forward pass handles both architectures
- ✅ Gradient clipping works for both
- ✅ Checkpoint saving includes model_type metadata
- ✅ Cache statistics logging for advanced model

**Checkpoint Format:**

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

### 2. **eval/run_dsn_pipeline.py**
**Changes:**
- ✅ Import `DSNAdvanced` and `DSNConfig`
- ✅ Auto-detect model type from checkpoint
- ✅ Load baseline or advanced model based on checkpoint
- ✅ Forward pass handles both model types
- ✅ Print model info (config, parameters) for advanced

**Detection Logic:**
```python
if "model_type" in ckpt and ckpt["model_type"] == "advanced":
    # Load advanced model
    model = DSNAdvanced(config).to(dev).eval()
    model.load_state_dict(ckpt["model"])
else:
    # Load baseline model
    enc = EncoderFC(...).to(dev).eval()
    pol = DSNPolicy(...).to(dev).eval()
```

### 3. **eval/batch_eval.py**
**Status:** ✅ Already compatible
- Calls `run_dsn_pipeline.py` which now handles both types
- No changes needed

---

## 🔄 Backward Compatibility

### Loading Old Checkpoints
Old baseline checkpoints will work automatically:
```python
# Old checkpoint format (no model_type key)
{
    "encoder": ...,
    "policy": ...
}
# → Detected as baseline ✅
```

### Loading New Checkpoints
New advanced checkpoints have explicit metadata:
```python
{
    "model": ...,
    "config": ...,
    "model_type": "advanced"
}
# → Detected as advanced ✅
```

---

## 🧪 Testing

### Test Baseline Model
```bash
# Train baseline
python -m src.pipeline.train_rl_dsn \
  --model_type baseline \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/test_baseline \
  --epochs 1

# Evaluate baseline
python -m eval.run_dsn_pipeline \
  --video data/samples/Sakuga/14652.mp4 \
  --out_dir outputs/test_baseline_eval \
  --checkpoint runs/test_baseline/dsn_checkpoint_ep1.pt
```

### Test Advanced Model
```bash
# Train advanced
python -m src.pipeline.train_rl_dsn \
  --model_type advanced \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/test_advanced \
  --epochs 1 \
  --num_attn_heads 4 \
  --num_attn_layers 2 \
  --use_cache 1

# Evaluate advanced
python -m eval.run_dsn_pipeline \
  --video data/samples/Sakuga/14652.mp4 \
  --out_dir outputs/test_advanced_eval \
  --checkpoint runs/test_advanced/dsn_checkpoint_ep1.pt
```

---

## 📊 GPU Usage Verification

All components properly use GPU:

### Model
```python
# Baseline
enc = EncoderFC(...).to(device)  # ✅
pol = DSNPolicy(...).to(device)  # ✅

# Advanced
model = DSNAdvanced(config).to(device)  # ✅
```

### Input Data
```python
x = torch.from_numpy(feats).unsqueeze(0).to(device)  # ✅
```

### Forward Pass
```python
# Baseline
h = enc(x)      # ✅ on GPU
probs = pol(h)  # ✅ on GPU

# Advanced
probs = model(x, scene_id=scene_id)  # ✅ on GPU
```

### Gradients
```python
loss.backward()  # ✅ All gradients on GPU
```

---

## 🎯 Usage Examples

### Training

**Baseline:**
```bash
python -m src.pipeline.train_rl_dsn \
  --model_type baseline \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/dsn_baseline \
  --epochs 20 \
  --device cuda:0
```

**Advanced:**
```bash
python -m src.pipeline.train_rl_dsn \
  --model_type advanced \
  --dataset_root outputs/sakuga_dataset \
  --save_dir runs/dsn_advanced \
  --epochs 20 \
  --device cuda:0 \
  --num_attn_heads 4 \
  --num_attn_layers 2 \
  --num_scales 3 \
  --use_cache 1
```

### Evaluation

**Single Video:**
```bash
python -m eval.run_dsn_pipeline \
  --video data/samples/Sakuga/14652.mp4 \
  --out_dir outputs/eval_14652 \
  --checkpoint runs/dsn_advanced/dsn_checkpoint_ep10.pt \
  --device cuda:0
```

**Batch Evaluation:**
```bash
python -m eval.batch_eval \
  --videos_dir data/samples/Sakuga \
  --output_dir outputs/batch_eval \
  --checkpoint runs/dsn_advanced/dsn_checkpoint_ep10.pt \
  --device cuda:0 \
  --with_baselines
```

---

## 🔍 Verification Checklist

- [x] Training pipeline supports both model types
- [x] Checkpoint saving includes model_type metadata
- [x] Checkpoint loading auto-detects model type
- [x] Evaluation pipeline works with both types
- [x] Batch evaluation compatible
- [x] GPU usage verified for all components
- [x] Backward compatible with old checkpoints
- [x] Cache statistics logged for advanced model
- [x] Forward pass correct for both types
- [x] Gradient flow correct for both types

---

## 📝 Notes

1. **Model Type Detection:** Automatic based on checkpoint content
2. **No Breaking Changes:** Old code still works with old checkpoints
3. **GPU Compatibility:** All tensors properly moved to device
4. **Cache Support:** Only available in advanced model
5. **Ablation Ready:** Can disable LSTM, attention, or multi-scale in advanced model

---

## ✅ Status: COMPLETE

All files have been updated to support both baseline and advanced DSN models with full backward compatibility.
