# DSN Evaluation Scripts Summary

## 📁 Folder Structure

```
scripts/eval_bash_script/
├── eval_track_a_features.sh    # Track A: Anime-CLIP-IQA Features
├── eval_track_b_rewards.sh     # Track B: Anime-CLIP-IQA Rewards
├── eval_track_c_combined.sh    # Track C: Combined (Features + Rewards)
├── eval_track_d_anime.sh       # Track D: Anime Emphasis + Prob Sep
├── eval_baseline.sh            # Baseline DSN
├── eval_advanced.sh            # Advanced DSN (no motion)
├── eval_raft_motion.sh         # DSN with RAFT Motion
└── README.md                   # Full documentation
```

## 🔗 Training ↔ Evaluation Mapping

| Training Script | Evaluation Script | Description |
|----------------|-------------------|-------------|
| `train_track_a.sh` | `eval_track_a_features.sh` | Anime-CLIP-IQA as input features (518-dim) |
| `train_track_b.sh` | `eval_track_b_rewards.sh` | Anime-CLIP-IQA as rewards only |
| `train_track_c.sh` | `eval_track_c_combined.sh` | Both features AND rewards |
| `train_track_d_anime.sh` | `eval_track_d_anime.sh` | Advanced anime emphasis rewards |
| `train_baseline.sh` | `eval_baseline.sh` | Simple baseline DSN |
| `train_advanced_dsn.sh` | `eval_advanced.sh` | Advanced architecture, no motion |
| `train_dsn_with_raft_motion.sh` | `eval_raft_motion.sh` | Advanced + RAFT motion features |

## ⚙️ Unified Configuration (All Scripts)

- **Scene Detection Backend**: `transnetv2` ✅
- **Embedder**: `clip_vitb32`
- **Budget**: ratio=0.06, Bmin=3, Bmax=15
- **Resolution**: 320x180
- **Sample Stride**: 5
- **Max Videos**: 30

## 🚀 Quick Start

```bash
# Run any track evaluation
bash scripts/eval_bash_script/eval_track_a_features.sh

# Run all evaluations
for script in scripts/eval_bash_script/eval_*.sh; do
    echo "Running $(basename $script)"
    bash "$script"
done
```

## 📊 View Results

```bash
# Compare all tracks
for dir in runs/eval_*/; do
    echo "=== $(basename $dir) ==="
    jq '.aggregate_metrics' "$dir/summary_results.json"
done
```

See `README.md` for full documentation.
