#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_anime_attrs(scene_dir: Path):
    path = scene_dir / "anime_attrs.npy"
    if path.exists():
        return np.load(path)
    return None

def main():
    parser = argparse.ArgumentParser(description="Analyze Anime-CLIP-IQA scores for selected keyframes.")
    parser.add_argument("--summary_json", type=str, required=True, help="Path to summary_results.json from batch_eval")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save analysis plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.summary_json, "r") as f:
        summary = json.load(f)

    dataset_root = Path(args.dataset_root)
    
    # Accumulators
    scores_selected = []
    scores_rejected = []
    
    # Attribute names (must match prepare_anime_attrs.py)
    attr_names = ["Sharpness", "Colorfulness", "Brightness", "Sakuga", "Cinematic", "Expression"]

    print(f"Analyzing {len(summary['videos'])} videos...")

    for vid_key, vid_data in tqdm(summary["videos"].items()):
        # vid_key might be full path or relative, let's try to find it
        # Assuming vid_key is the video name
        
        # In batch_eval, keys are video paths. 
        # But we need to map back to the training dataset structure to find anime_attrs.npy
        # The training dataset is split into scenes.
        # This script assumes we are analyzing validation results on the SAME dataset or a dataset with same structure.
        # If validation was done on raw videos, we might not have anime_attrs.npy for them unless we ran preprocess on them too.
        
        # Let's assume the user ran preprocess on the validation set too, or we are analyzing training set results.
        # If not, we can't do much.
        
        # Actually, let's try to find the scene directory.
        # If vid_data has 'scene_dir' or similar? No, usually just keyframes.
        
        # Wait, the prompt said: "Đọc summary JSON / keyframe index."
        # And "Dùng Anime-CLIP-IQA scores để tính".
        
        # If the validation set is different from training set, we need anime_attrs for validation set.
        # The user should run prepare_anime_attrs.py on validation set as well.
        
        # Let's assume we can find the scene by name.
        # Video name -> Scene dir?
        # If the dataset is split into scenes, we need to know which scene corresponds to which result.
        
        # If batch_eval was run on "videos", it produces one entry per video.
        # If the input to batch_eval was a folder of videos, it treats each video as a unit.
        
        # If we are evaluating on the training set (which is split into scenes), 
        # we can match by scene name.
        
        # Let's try to match by filename.
        vid_path = Path(vid_key)
        video_name = vid_path.stem
        
        # Search for this video/scene in dataset_root
        # dataset_root structure: <root>/<video_stem>/scene_xxxx/
        # or <root>/scene_xxxx/ if flat.
        
        # Let's search recursively
        found_scenes = list(dataset_root.glob(f"**/{video_name}*"))
        
        # This is tricky because one video might be split into multiple scenes in the dataset.
        # But batch_eval usually runs on full videos.
        
        # If we can't map easily, we skip.
        # BUT, for the sake of the plan, let's assume we are analyzing the training set or a preprocessed validation set
        # where we can find the attributes.
        
        # Let's assume the user provides a dataset_root that contains the scenes corresponding to the evaluated videos.
        # And that the video name in summary json matches the scene folder name or parent.
        
        pass

    # REVISED STRATEGY:
    # Instead of complex mapping, let's just iterate through the dataset_root, 
    # and if we find a scene that matches an entry in summary, we process it.
    
    # Actually, the summary_results.json from batch_eval contains "keyframes" (list of indices).
    # If we ran batch_eval on the *training scenes* (which is possible), then keys are scene paths.
    
    # Let's assume keys in summary are paths that we can resolve.
    
    count = 0
    for vid_path_str, data in tqdm(summary["videos"].items()):
        # Try to find anime_attrs.npy at this path
        # If vid_path_str is absolute, check it.
        # If relative, check relative to dataset_root? No, batch_eval usually uses absolute paths.
        
        scene_path = Path(vid_path_str)
        attrs_path = scene_path / "anime_attrs.npy"
        
        if not attrs_path.exists():
            # Try appending to dataset_root if it's just a name
            scene_path = dataset_root / vid_path_str
            attrs_path = scene_path / "anime_attrs.npy"
            
        if not attrs_path.exists():
            # Try searching
            candidates = list(dataset_root.glob(f"**/{Path(vid_path_str).name}"))
            if candidates:
                scene_path = candidates[0]
                attrs_path = scene_path / "anime_attrs.npy"
        
        if not attrs_path.exists():
            continue
            
        attrs = np.load(attrs_path) # (T, K)
        keyframes = data["keyframes"] # list of indices
        
        # Get selected attributes
        # keyframes are 0-indexed? usually yes.
        
        # Check bounds
        T = len(attrs)
        valid_kf = [k for k in keyframes if k < T]
        
        if not valid_kf:
            continue
            
        sel_attrs = attrs[valid_kf]
        scores_selected.append(sel_attrs)
        
        # Get rejected attributes
        all_indices = set(range(T))
        sel_set = set(valid_kf)
        rej_indices = list(all_indices - sel_set)
        
        if rej_indices:
            rej_attrs = attrs[rej_indices]
            scores_rejected.append(rej_attrs)
            
        count += 1

    print(f"Matched {count} scenes/videos with attributes.")
    
    if not scores_selected:
        print("No data found.")
        return

    scores_selected = np.concatenate(scores_selected, axis=0) # (N_sel, K)
    scores_rejected = np.concatenate(scores_rejected, axis=0) # (N_rej, K)
    
    print(f"Selected frames: {len(scores_selected)}")
    print(f"Rejected frames: {len(scores_rejected)}")
    
    # Compute statistics
    mean_sel = scores_selected.mean(axis=0)
    mean_rej = scores_rejected.mean(axis=0)
    
    print("\n--- Mean Scores ---")
    print(f"{'Attribute':<15} | {'Selected':<10} | {'Rejected':<10} | {'Diff':<10}")
    print("-" * 55)
    for i, name in enumerate(attr_names):
        diff = mean_sel[i] - mean_rej[i]
        print(f"{name:<15} | {mean_sel[i]:.4f}     | {mean_rej[i]:.4f}     | {diff:+.4f}")
        
    # Plot
    x = np.arange(len(attr_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, mean_sel, width, label='Selected')
    rects2 = ax.bar(x + width/2, mean_rej, width, label='Rejected')
    
    ax.set_ylabel('Mean Score')
    ax.set_title('Anime-CLIP-IQA Scores: Selected vs Rejected')
    ax.set_xticks(x)
    ax.set_xticklabels(attr_names)
    ax.legend()
    
    plt.savefig(output_dir / "anime_scores_comparison.png")
    print(f"\nPlot saved to {output_dir / 'anime_scores_comparison.png'}")
    
    # Save raw stats
    stats = {
        "selected_mean": mean_sel.tolist(),
        "rejected_mean": mean_rej.tolist(),
        "attr_names": attr_names
    }
    with open(output_dir / "anime_scores_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
