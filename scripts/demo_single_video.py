#!/usr/bin/env python3
"""
Demo V11 vs VSUMM on single video with TransNetV2 scene detection.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dsn_v8 import create_dsn_v8
from src.scene_detection.registry import create_detector
from src.scene_detection.interface import Scene
from scripts.precompute_script.precompute_all_v11 import normalize_and_merge_scenes
from src.models.anime_clipiqa_v3 import create_anime_clipiqa
import clip
from PIL import Image


def create_keyframe_grid(frames, selected_indices, title="", max_frames=8):
    """Create visualization grid of keyframes."""
    if not frames or not selected_indices:
        return None
    
    # Select frames to display
    display_indices = selected_indices[:max_frames]
    key_frames = [frames[i] for i in display_indices if i < len(frames)]
    
    if not key_frames:
        return None
    
    # Resize frames
    target_h, target_w = 180, 320
    resized = [cv2.resize(f, (target_w, target_h)) for f in key_frames]
    
    # Create grid
    n = len(resized)
    cols = 4
    rows = (n + cols - 1) // cols
    
    grid = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
    
    for idx, frame in enumerate(resized):
        r = idx // cols
        c = idx % cols
        grid[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = frame
    
    if rows < 2: # Ensure uniform height for stacking if row count differs, though minimal here
        padding = np.zeros(((2-rows) * target_h, cols * target_w, 3), dtype=np.uint8)
        grid = np.vstack([grid, padding])

    # Add title bar
    title_bar = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return np.vstack([title_bar, grid])



def extract_video_features(video_path, scenes, device="cuda", stride=1):
    """Extract CLIP features and anime attributes for each scene."""
    
    # Initialize models
    print(f"Loading CLIP (ViT-B/32)... Stride={stride}")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    anime_iqa = create_anime_clipiqa(device=device)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scene_data = []
    
    for scene_id, scene in enumerate(scenes):
        start, end = scene.start_frame, scene.end_frame
        print(f"\n  Scene {scene_id+1}: frames {start}-{end}")
        
        # Extract frames with stride
        frames = []
        frame_indices = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        # Optimization: Jump directly if stride > 1
        for i in range(start, min(end + 1, total_frames), stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_indices.append(i)
        
        if not frames:
            continue
        
        # Extract features
        # 1. CLIP Features
        images = []
        for f in frames:
            img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            images.append(preprocess(pil_img))
        
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            clip_feats = clip_model.encode_image(image_input)
            clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True)
            clip_feats = clip_feats.cpu().numpy()
            
        # 2. Anime Attributes
        anime_attrs = anime_iqa.get_legacy_format_scores(frames)  # (T, 6)
        
        scene_data.append({
            "scene_id": scene_id,
            "start": start,
            "end": end,
            "frames": frames,
            "frame_indices": frame_indices,
            "clip_feats": clip_feats,
            "anime_attrs": anime_attrs,
        })
    
    cap.release()
    return scene_data


def run_comparison_on_video(video_path, v11_model, vsumm_model, device="cuda", budget_ratio=0.1, output_dir=".", stride=1, scene_threshold=0.5, min_scene_len=15):
    """Run V11 vs VSUMM comparison on a video."""
    
    print(f"\n{'='*80}")
    print(f"Processing: {Path(video_path).name}")
    print(f"{'='*80}")
    
    # Scene detection
    print(f"\n[1] Scene Detection (TransNetV2, threshold={scene_threshold}, min_len={min_scene_len})...")
    detector = create_detector("transnetv2", model_dir="/home/serverai/ltdoanh/LayoutGeneration/src/models/TransNetV2", prob_threshold=scene_threshold)
    scenes_raw = detector.detect(str(video_path))
    scenes = normalize_and_merge_scenes(scenes_raw, min_len_frames=min_scene_len)
    print(f"  Detected {len(scenes)} scenes (after merge)")
    
    # Feature extraction
    print("\n[2] Feature Extraction...")
    scene_data = extract_video_features(video_path, scenes, device=device, stride=stride)
    
    # Run models on each scene
    print("\n[3] Model Inference...")
    results = []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for scene_info in scene_data:
        T = len(scene_info["frames"])
        budget = max(1, int(T * budget_ratio))
        
        print(f"\n  Scene {scene_info['scene_id']+1}: {T} frames (stride={stride}), budget={budget}")
        
        # V11
        feats_v11 = np.concatenate([scene_info["clip_feats"], scene_info["anime_attrs"]], axis=1)
        feats_t_v11 = torch.from_numpy(feats_v11).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs_v11, _ = v11_model(feats_t_v11)
            probs_v11 = probs_v11.squeeze(0).cpu().numpy()
        
        sel_v11 = sorted(np.argsort(probs_v11)[-budget:].tolist())
        
        # VSUMM  
        feats_vsumm = torch.from_numpy(scene_info["clip_feats"]).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs_vsumm = vsumm_model(feats_vsumm)
            probs_vsumm = probs_vsumm.squeeze().cpu().numpy()
        
        sel_vsumm = sorted(np.argsort(probs_vsumm)[-budget:].tolist())
        
        # Metrics
        quality = scene_info["anime_attrs"].mean(axis=1)
        ranks = np.argsort(np.argsort(quality))
        percentiles = ranks / max(1, T - 1)
        
        mpr_v11 = float(np.mean(percentiles[sel_v11]))
        mpr_vsumm = float(np.mean(percentiles[sel_vsumm]))
        
        overlap = len(set(sel_v11) & set(sel_vsumm))
        
        # Map back to global frame indices for report
        v11_global_indices = [scene_info["frame_indices"][i] for i in sel_v11]
        vsumm_global_indices = [scene_info["frame_indices"][i] for i in sel_vsumm]
        
        results.append({
            "scene_id": scene_info["scene_id"],
            "start": scene_info["start"],
            "end": scene_info["end"],
            "total_frames_sampled": T,
            "stride": stride,
            "budget": budget,
            "v11_mpr": mpr_v11,
            "vsumm_mpr": mpr_vsumm,
            "overlap": overlap,
            "overlap_pct": overlap / budget * 100 if budget > 0 else 0,
            "v11_indices": v11_global_indices,
            "vsumm_indices": vsumm_global_indices,
        })
        
        print(f"    V11:   MPR={mpr_v11:.3f}, Selected (Global): {v11_global_indices[:5]}...")
        print(f"    VSUMM: MPR={mpr_vsumm:.3f}, Selected (Global): {vsumm_global_indices[:5]}...")
        print(f"    Overlap: {overlap}/{budget} ({overlap / budget * 100:.1f}%)")
        
        # Visualization
        grid_v11 = create_keyframe_grid(scene_info["frames"], sel_v11, 
                                       title=f"V11 (MPR={mpr_v11:.2f})")
        grid_vsumm = create_keyframe_grid(scene_info["frames"], sel_vsumm,
                                         title=f"VSUMM (MPR={mpr_vsumm:.2f})")
                                         
        if grid_v11 is not None and grid_vsumm is not None:
            # Resize if needed (width should match)
            if grid_v11.shape[1] != grid_vsumm.shape[1]:
                 grid_vsumm = cv2.resize(grid_vsumm, (grid_v11.shape[1], grid_vsumm.shape[0]))
                 
            combined = np.vstack([grid_v11, grid_vsumm])
            viz_path = output_path / f"scene_{scene_info['scene_id']+1:03d}_comparison.jpg"
            cv2.imwrite(str(viz_path), combined)
            print(f"    Saved comparison: {viz_path}")
    
    return results, scene_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--v11_checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/training_v11_final_new/best.pt")
    parser.add_argument("--vsumm_checkpoint", type=str,
                       default="/home/serverai/ltdoanh/LayoutGeneration/runs/ablation_vsumm/sakuga_train/model_epoch60.pth.tar")
    parser.add_argument("--budget_ratio", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="demo_single_video_results.json")
    parser.add_argument("--output_dir", type=str, default="demo_vis_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stride", type=int, default=5, help="Frame sampling stride")
    parser.add_argument("--scene_threshold", type=float, default=0.5, help="TransNetV2 boundary threshold (higher = fewer scenes)")
    parser.add_argument("--min_scene_len", type=int, default=15, help="Minimum scene length (frames) to merge")
    
    args = parser.parse_args()
    
    # Load V11
    print("\nLoading V11...")
    v11_ckpt = torch.load(args.v11_checkpoint, map_location="cpu", weights_only=False)
    v11_config = v11_ckpt.get("config", {})
    
    v11_model = create_dsn_v8(
        feat_dim=512 + 6,
        use_pcgrad=False,
        num_attn_layers=v11_config.get("num_attn_layers", 2),
        gating_hidden=v11_config.get("gating_hidden", 64),
        lstm_hidden=v11_config.get("lstm_hidden", 128),
    ).to(args.device)
    v11_model.load_state_dict(v11_ckpt["model_state_dict"])
    v11_model.eval()
    
    # Load VSUMM
    print("Loading VSUMM...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "ablation" / "pytorch-vsumm-reinforce"))
    from models import DSN
    
    vsumm_ckpt = torch.load(args.vsumm_checkpoint, map_location="cpu", weights_only=False)
    vsumm_model = DSN(in_dim=512, hid_dim=256, num_layers=1, cell='lstm').to(args.device)
    
    if "state_dict" in vsumm_ckpt:
        state_dict = vsumm_ckpt["state_dict"]
    elif "model_state_dict" in vsumm_ckpt:
        state_dict = vsumm_ckpt["model_state_dict"]
    else:
        state_dict = vsumm_ckpt
    
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    vsumm_model.load_state_dict(new_state_dict)
    vsumm_model.eval()
    
    # Run comparison
    results, scene_data = run_comparison_on_video(
        args.video, v11_model, vsumm_model, args.device, args.budget_ratio, args.output_dir, args.stride, args.scene_threshold, args.min_scene_len
    )
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    if not df.empty:
        print(df[["scene_id", "total_frames_sampled", "budget", "v11_mpr", "vsumm_mpr", "overlap_pct"]].to_string(index=False))
        
        print("\n" + "="*80)
        print("AVERAGES ACROSS ALL SCENES")
        print("="*80)
        print(f"V11 MPR:      {df['v11_mpr'].mean():.4f}")
        print(f"VSUMM MPR:    {df['vsumm_mpr'].mean():.4f}")
        print(f"Overlap:      {df['overlap_pct'].mean():.1f}%")
    else:
        print("No scenes processed.")
    
    print(f"\n✅ Results saved to: {args.output}")
    print(f"✅ Visualizations saved to: {args.output_dir}/*.jpg")




if __name__ == "__main__":
    main()


"""
python3 scripts/demo_single_video.py --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/13926.mp4 --budget_ratio 0.1 --output demo_13926_results.json --output_dir demo_13926_vis
python3 scripts/demo_single_video.py --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/7369.mp4 --budget_ratio 0.1 --output demo_7369_results.json --output_dir demo_7369_vis --stride 5
python3 scripts/demo_single_video.py --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/9046.mp4 --budget_ratio 0.1 --output demo_9046_results.json --output_dir demo_9046_vis --stride 5
python3 scripts/demo_single_video.py --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/115042.mp4 --budget_ratio 0.1 --output demo_115042_results.json --output_dir demo_115042_vis --stride 5
python3 scripts/demo_single_video.py --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/115042.mp4 --budget_ratio 0.1 --output outputs/demo_115042_results.json --output_dir demo_115042_vis --stride 7 --scene_threshold 0.8   --min_scene_len 100

python3 scripts/demo_single_video.py --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/15019.mp4 --budget_ratio 0.1 --output outputs/demo_15019_results.json --output_dir demo_15019_vis --stride 7 --scene_threshold 0.8   --min_scene_len 100
"""