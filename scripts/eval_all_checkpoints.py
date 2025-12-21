#!/usr/bin/env python3
"""
Evaluate all checkpoints in a directory and compare their performance.
Processes checkpoints from newest to oldest (highest epoch to lowest).
"""
import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

def find_checkpoints(checkpoint_dir: str) -> List[Tuple[int, Path]]:
    """
    Find all checkpoint files and sort by epoch number (descending).
    Returns list of (epoch_num, checkpoint_path) tuples.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    checkpoints = []
    pattern = re.compile(r'dsn_checkpoint_ep(\d+)\.pt')
    
    for ckpt_file in ckpt_dir.glob("dsn_checkpoint_ep*.pt"):
        match = pattern.match(ckpt_file.name)
        if match:
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, ckpt_file))
    
    # Sort by epoch number descending (newest first)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    
    return checkpoints

def run_evaluation(
    checkpoint_path: Path,
    epoch_num: int,
    videos_dir: str,
    output_base: str,
    feat_dim: int = 512,
    enc_hidden: int = 1024,
    lstm_hidden: int = 512,
    budget_ratio: float = 0.06,
    device: str = "cuda",
    backend: str = "transnetv2",
    embedder: str = "clip_vitb32",
    max_videos: int = None,
    use_anime_attrs: int = 1,
    min_scene_len: int = 48
) -> dict:
    """
    Run batch evaluation for a single checkpoint.
    Returns the summary results dict.
    """
    output_dir = Path(output_base) / f"ep{epoch_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "-m", "eval.batch_eval",
        "--videos_dir", videos_dir,
        "--output_dir", str(output_dir),
        "--checkpoint", str(checkpoint_path),
        "--device", device,
        "--feat_dim", str(feat_dim),
        "--enc_hidden", str(enc_hidden),
        "--lstm_hidden", str(lstm_hidden),
        "--budget_ratio", str(budget_ratio),
        "--embedder", embedder,
        "--backend", backend,
        "--model_dir", "src/models/TransNetV2",
        "--prob_threshold", "0.5",
        "--scene_device", device,
        "--use_anime_attrs", str(use_anime_attrs),
        "--min_scene_len", str(min_scene_len),
        "--with_baselines"
    ]
    
    if max_videos is not None:
        cmd += ["--max_videos", str(max_videos)]
    
    print(f"\n{'='*60}")
    print(f"Evaluating Checkpoint: Epoch {epoch_num}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Evaluation failed for epoch {epoch_num}")
        print(f"Error: {result.stderr[:500]}")
        return None
    
    # Load results
    summary_path = output_dir / "summary_results.json"
    if not summary_path.exists():
        print(f"⚠️  No summary file found for epoch {epoch_num}")
        return None
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate all checkpoints")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing checkpoints")
    parser.add_argument("--videos_dir", required=True, help="Directory containing videos to evaluate")
    parser.add_argument("--output_dir", required=True, help="Base output directory for evaluation results")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--enc_hidden", type=int, default=1024)
    parser.add_argument("--lstm_hidden", type=int, default=512)
    parser.add_argument("--budget_ratio", type=float, default=0.06)
    parser.add_argument("--backend", default="transnetv2")
    parser.add_argument("--embedder", default="clip_vitb32")
    parser.add_argument("--max_videos", type=int, default=None, help="Limit number of videos per eval")
    parser.add_argument("--use_anime_attrs", type=int, default=1)
    parser.add_argument("--min_scene_len", type=int, default=48)
    
    args = parser.parse_args()
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir)
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Epochs: {[ep for ep, _ in checkpoints]}")
    
    # Evaluate each checkpoint
    all_results = []
    
    for epoch_num, ckpt_path in tqdm(checkpoints, desc="Evaluating Checkpoints"):
        summary = run_evaluation(
            checkpoint_path=ckpt_path,
            epoch_num=epoch_num,
            videos_dir=args.videos_dir,
            output_base=args.output_dir,
            feat_dim=args.feat_dim,
            enc_hidden=args.enc_hidden,
            lstm_hidden=args.lstm_hidden,
            budget_ratio=args.budget_ratio,
            device=args.device,
            backend=args.backend,
            embedder=args.embedder,
            max_videos=args.max_videos,
            use_anime_attrs=args.use_anime_attrs,
            min_scene_len=args.min_scene_len
        )
        
        if summary:
            agg = summary.get("aggregate_metrics", {})
            result = {
                "epoch": epoch_num,
                "checkpoint": str(ckpt_path),
                "metrics": agg,
                "stats": summary.get("statistics", {})
            }
            all_results.append(result)
            
            # Print summary
            print(f"\n📊 Epoch {epoch_num} Results:")
            print(f"  RecErr: {agg.get('RecErr_mean', 'N/A'):.4f}" if agg.get('RecErr_mean') else "  RecErr: N/A")
            print(f"  Frechet: {agg.get('Frechet_mean', 'N/A'):.4f}" if agg.get('Frechet_mean') else "  Frechet: N/A")
            print(f"  SceneCoverage: {agg.get('SceneCoverage_mean', 'N/A'):.4f}" if agg.get('SceneCoverage_mean') else "  SceneCoverage: N/A")
    
    # Save comparison results
    comparison_file = Path(args.output_dir) / "checkpoint_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Comparison saved to: {comparison_file}")
    
    # Print best checkpoints
    if all_results:
        print(f"\n🏆 Best Checkpoints:")
        
        # Best RecErr (lower is better)
        rec_results = [(r["epoch"], r["metrics"].get("RecErr_mean")) 
                       for r in all_results if r["metrics"].get("RecErr_mean") is not None]
        if rec_results:
            rec_results.sort(key=lambda x: x[1])
            print(f"  RecErr: Epoch {rec_results[0][0]} = {rec_results[0][1]:.4f}")
        
        # Best Frechet (lower is better)
        fre_results = [(r["epoch"], r["metrics"].get("Frechet_mean")) 
                       for r in all_results if r["metrics"].get("Frechet_mean") is not None]
        if fre_results:
            fre_results.sort(key=lambda x: x[1])
            print(f"  Frechet: Epoch {fre_results[0][0]} = {fre_results[0][1]:.4f}")
        
        # Best SceneCoverage (higher is better)
        scov_results = [(r["epoch"], r["metrics"].get("SceneCoverage_mean")) 
                        for r in all_results if r["metrics"].get("SceneCoverage_mean") is not None]
        if scov_results:
            scov_results.sort(key=lambda x: x[1], reverse=True)
            print(f"  SceneCoverage: Epoch {scov_results[0][0]} = {scov_results[0][1]:.4f}")

if __name__ == "__main__":
    main()
