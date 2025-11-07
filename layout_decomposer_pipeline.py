#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Scene→Keyframe Pipeline
- Scene detection via pluggable backends (registry): pyscenedetect, transnetv2, ...
- Keyframe selection via pluggable distance metrics (registry): lpips, dists, ...
- Outputs:
    * scenes.json / scenes.csv
    * keyframes.csv
    * keyframes/ (exported JPGs)
    * scene_previews/ (optional mid/start/end frame of each scene)
All code comments are in English (per user requirement).
"""

from __future__ import annotations
import os
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import sys

from tqdm import tqdm


# --- Registries (auto-register built-ins via package __init__) ---
from src.scene_detection import (
    create_detector,
    available_detectors,
    Scene,
)
from src.distance_selector import (
    create_metric,
    available_metrics,
)
from src.keyframe.medoid_selector import (
    MedoidSelector,
    Keyframe as KF,
)

from src.keyframe.random_selector import RandomSelector

sys.path.append('objectfree')
from objectfree.eval_objectfree import CompletePipeline
from utils.io import *

# Import Colla modules
sys.path.append('repos/Colla')
import repos.Colla.shape_decomposition as sd
import repos.Colla.sas_optimization as so
import repos.Colla.collage_assembly as ca
import repos.Colla.create_masks as cm
from repos.Colla import evaluation



# ------------------------------
# Object-Free Pipeline Integration
# ------------------------------
def run_complete_object_free_pipeline(keyframes_folder, output_base, device="cuda", config_path="objectfree/config.yaml", checkpoint_path="./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"):
    """Run complete object-free pipeline using CompletePipeline class"""
    
    # Initialize pipeline
    pipeline = CompletePipeline(device=device, output_dir=output_base, config_path=config_path)
    pipeline.initialize_detectors()
    
    # Override config and checkpoint paths
    pipeline.object_detector.config_path = config_path
    pipeline.object_detector.checkpoint_path = checkpoint_path
    
    # Process the folder
    result = pipeline.process_single_folder(keyframes_folder, output_base)
    
    return result
def normalize_and_merge_scenes(
    scenes: List[Scene],
    min_len_frames: int = 0,
) -> List[Scene]:
    """
    Normalize (ensure start<=end), sort by start, and optionally merge short scenes
    into the previous one if below `min_len_frames`.
    """
    if not scenes:
        return []

    # Normalize and sort
    norm: List[Scene] = []
    for s in scenes:
        a, b = int(s.start_frame), int(s.end_frame)
        if b < a:
            a, b = b, a
        norm.append(Scene(a, b))
    norm.sort(key=lambda x: (x.start_frame, x.end_frame))

    if min_len_frames <= 0:
        return norm

    merged: List[Scene] = []
    for sc in norm:
        if not merged:
            merged.append(sc)
            continue
        cur_len = sc.end_frame - sc.start_frame + 1
        if cur_len >= min_len_frames:
            merged.append(sc)
        else:
            prev = merged[-1]
            if sc.start_frame <= prev.end_frame + 1:
                # Contiguous → extend previous
                merged[-1] = Scene(prev.start_frame, max(prev.end_frame, sc.end_frame))
            else:
                # Non-contiguous but still merge into previous by extending end
                merged[-1] = Scene(prev.start_frame, sc.end_frame)
    return merged


# ------------------------------
# Argparse
# ------------------------------
def build_argparser() -> argparse.ArgumentParser:
    # Query available backends dynamically (packages import will auto-register)
    scene_choices = available_detectors()
    metric_choices = available_metrics()

    ap = argparse.ArgumentParser(
        description="Scene→Keyframe pipeline using pluggable scene detectors and distance metrics."
    )
    ap.add_argument("--video", type=str, required=True, help="Input video path.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")

    # Scene detection backend + params
    ap.add_argument("--backend", type=str, default="pyscenedetect", choices=scene_choices,
                    help="Scene detection backend.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="[pyscenedetect] ContentDetector threshold (default 27.0).")
    ap.add_argument("--model_dir", type=str, default=None,
                    help="[transnetv2] Directory containing weights/, or pass --weights_path.")
    ap.add_argument("--weights_path", type=str, default=None,
                    help="[transnetv2] Direct path to .pth weights (overrides model_dir).")
    ap.add_argument("--prob_threshold", type=float, default=None,
                    help="[transnetv2] Boundary probability threshold (default 0.5).")
    ap.add_argument("--scene_device", type=str, default=None,
                    help="[transnetv2] Device for model ('cuda'/'cpu').")

    # Scenes post-process & preview
    ap.add_argument("--min_scene_len", type=int, default=0,
                    help="Minimum scene length in frames for post-merge (0 = disabled).")
    ap.add_argument("--export_preview", action="store_true",
                    help="Export one preview image per scene.")
    ap.add_argument("--preview_which", type=str, default="mid",
                    choices=["start", "mid", "end"], help="Which frame to export as preview.")
    ap.add_argument("--preview_jpeg_quality", type=int, default=95)

    # Distance metric + selection params
    ap.add_argument("--distance_backend", type=str, default="lpips", choices=metric_choices,
                    help="Distance metric backend.")
    ap.add_argument("--distance_device", type=str, default=None,
                    help="Device for metric ('cuda'/'cpu').")
    ap.add_argument("--lpips_net", type=str, default="alex",
                    help="[lpips] Backbone: 'alex'|'vgg'|'squeeze'.")
    ap.add_argument("--dists_as_distance", type=int, default=1,
                    help="[dists] Use raw DISTS as distance (1) or negate as similarity (0).")

    ap.add_argument("--sample_stride", type=int, default=10,
                    help="Sample every N frames within a scene.")
    ap.add_argument("--max_frames_per_scene", type=int, default=30,
                    help="Cap sampled frames per scene (controls O(N^2) cost).")
    ap.add_argument("--keyframes_per_scene", type=int, default=1,
                    help="How many keyframes to pick per scene.")
    ap.add_argument("--nms_radius", type=int, default=3,
                    help="Greedy index-NMS radius when selecting multiple keyframes per scene.")
    ap.add_argument("--resize_w", type=int, default=320,
                    help="Resize width for distance computation (<=0 to disable).")
    ap.add_argument("--resize_h", type=int, default=180,
                    help="Resize height for distance computation (<=0 to disable).")
    ap.add_argument("--batch_pairs", type=int, default=16,
                    help="Mini-batch size of (i,j) pairs when computing pairwise distances.")

    # Keyframe selection
    ap.add_argument("--keyframe_selector", type=str, default="medoid", choices=["medoid", "random"],
                    help="Keyframe selection strategy.")
    ap.add_argument("--random_seed", type=int, default=None,
                    help="Random seed for reproducibility (only used with random selector).")

    # Keyframe export
    ap.add_argument("--key_jpeg_quality", type=int, default=95,
                    help="JPEG quality for exported keyframe images.")

    # Object-free pipeline
    ap.add_argument("--run_object_free_pipeline", action="store_true",
                    help="Run complete object-free evaluation pipeline after keyframe extraction.")
    ap.add_argument("--detection_config", type=str, default=None,
                    help="Path to Grounding DINO config file for object detection.")
    ap.add_argument("--detection_checkpoint", type=str, default=None,
                    help="Path to Grounding DINO checkpoint file.")
    ap.add_argument("--detection_device", type=str, default=None,
                    help="Device for object detection ('cuda'/'cpu').")
    
    # Colla layout decomposer pipeline args
    ap.add_argument("--input_shape_layout", type=str, default="repos/Colla/input_data/layout/baby.png",  help="Input shape layout image path.")
    ap.add_argument("--input_mask_folder", type=str, default="repos/Colla/input_data/image_collections/children_mask", help="Input mask folder path.")
    ap.add_argument("--scaling_factor", type=int, default = 2, help="Scaling factor for collage rendering.")

    return ap


# ------------------------------
# Main
# ------------------------------
def main():
    args = build_argparser().parse_args()
    
    # Create unique output directory with timestamp to avoid overwriting
    video_name = args.video.split('/')[-1].split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir = f"{args.out_dir}_{video_name}_{timestamp}"

    # Prepare output folders
    ensure_dir(args.out_dir)
    key_dir = os.path.join(args.out_dir, "keyframes")
    ensure_dir(key_dir)
    preview_dir = os.path.join(args.out_dir, "scene_previews")

    # Read basic video info
    total_frames, fps = read_video_basic_info(args.video)

    # Build scene-detector kwargs (only pass values that are actually set)
    det_kwargs: Dict[str, Any] = {
        "threshold": args.threshold,
        "model_dir": args.model_dir,
        "weights_path": args.weights_path,
        "prob_threshold": args.prob_threshold,
        "device": args.scene_device,
    }
    det_kwargs = {k: v for k, v in det_kwargs.items() if v not in (None, "", [])}

    # Run scene detection
    detector = create_detector(args.backend, **det_kwargs)
    scenes_raw: List[Scene] = detector.detect(args.video)
    detector.close()

    if not scenes_raw:
        print("[WARN] No scenes detected by backend. Fallback to the whole video as one scene.")
        scenes_raw = [Scene(0, max(0, total_frames - 1))]

    # Post-process scenes (optional)
    scenes = normalize_and_merge_scenes(scenes_raw, min_len_frames=args.min_scene_len)

    # Save scenes to JSON/CSV
    scene_rows: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenes):
        s, e = int(sc.start_frame), int(sc.end_frame)
        dur_frames = max(0, e - s + 1)
        scene_rows.append({
            "scene_id": i,
            "start_frame": s,
            "end_frame": e,
            "start_time": frames_to_timecode(s, fps),
            "end_time": frames_to_timecode(e, fps),
            "duration_frames": dur_frames,
            "duration_seconds": round(dur_frames / fps, 3) if fps > 0 else 0.0,
        })

    save_json(scene_rows, os.path.join(args.out_dir, "scenes.json"))
    save_csv(scene_rows, os.path.join(args.out_dir, "scenes.csv"))

    if args.export_preview:
        export_scene_previews(
            video_path=args.video,
            scenes=scenes,
            out_dir=preview_dir,
            which=args.preview_which,
            jpeg_quality=args.preview_jpeg_quality,
        )

    # Build distance metric
    dist_kwargs: Dict[str, Any] = {"device": args.distance_device}
    if args.distance_backend == "lpips":
        dist_kwargs.update({"net": args.lpips_net})
    elif args.distance_backend == "dists":
        dist_kwargs.update({"as_distance": bool(args.dists_as_distance)})

    metric = create_metric(args.distance_backend, **dist_kwargs)

    if args.keyframe_selector == "random":
        selector = RandomSelector(seed=args.random_seed)
    else:
        selector = MedoidSelector(metric=metric)

    # Prepare resize
    resize_to: Optional[Tuple[int, int]]
    if args.resize_w > 0 and args.resize_h > 0:
        resize_to = (args.resize_w, args.resize_h)
    else:
        resize_to = None

    # Select keyframes per scene
    keyframes: List[KF] = []
    for sid, sc in enumerate(tqdm(scenes, desc="Selecting keyframes")):
        kfs = selector.select_for_scene(
            video_path=args.video,
            scene_range=(sc.start_frame, sc.end_frame),
            sample_stride=args.sample_stride,
            max_frames_per_scene=args.max_frames_per_scene,
            keyframes_per_scene=args.keyframes_per_scene,
            nms_radius=args.nms_radius,
            resize_to=resize_to,
            scene_id=sid,
            batch_pairs=args.batch_pairs,
        )
        keyframes.extend(kfs)

    # Save keyframes CSV
    key_rows: List[Dict[str, Any]] = []
    for kf in keyframes:
        key_rows.append({
            "scene_id": kf.scene_id,
            "frame_idx": kf.frame_idx,
            "time": frames_to_timecode(kf.frame_idx, fps),
            "score": round(kf.score, 6),
            "distance_backend": args.distance_backend,
        })
    save_csv(key_rows, os.path.join(args.out_dir, "keyframes.csv"))

    # Export keyframe images
    export_keyframe_images(
        video_path=args.video,
        keyframes=keyframes,
        out_dir=key_dir,
        jpeg_quality=args.key_jpeg_quality,
    )

    # Run complete object-free pipeline (optional)
    object_free_results = None
    if args.run_object_free_pipeline:
        print("\n" + "="*80)
        print("RUNNING COMPLETE OBJECT-FREE PIPELINE")
        print("="*80)
        
        # Determine device for object-free pipeline
        of_device_str = args.detection_device or args.distance_device or "cuda"
        
        # Create base output directory for object-free results
        of_base_dir = os.path.join(args.out_dir, "object_free_evaluation")
        ensure_dir(of_base_dir)
        
        try:
            object_free_results = run_complete_object_free_pipeline(
                keyframes_folder=key_dir,
                output_base=of_base_dir,
                device=of_device_str,
                config_path=args.detection_config,
                checkpoint_path=args.detection_checkpoint
            )
            
            if object_free_results:
                print(f"\n[SUCCESS] Object-free pipeline completed!")
                print(f"  • Results: {object_free_results['output_dir']}")
                print(f"  • Final report: {os.path.join(object_free_results['output_dir'], 'final_report.json')}")
            else:
                print(f"[WARN] Object-free pipeline failed!")
                
        except Exception as e:
            print(f"[ERROR] Object-free pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"[DONE] Scenes: {len(scenes)} | Keyframes: {len(keyframes)}")
    print(f"  • Scenes JSON : {os.path.join(args.out_dir, 'scenes.json')}")
    print(f"  • Scenes CSV  : {os.path.join(args.out_dir, 'scenes.csv')}")
    print(f"  • Keyframes CSV: {os.path.join(args.out_dir, 'keyframes.csv')}")
    if args.export_preview:
        print(f"  • Scene previews: {preview_dir}")
    print(f"  • Keyframe images: {key_dir}")
    if args.run_object_free_pipeline and object_free_results:
        print(f"  • Object-free evaluation: {object_free_results['output_dir']}")
    print("="*60)

    # Colla pipeline
    print("\n" + "="*80)
    print("RUNNING COLLA LAYOUT DECOMPOSER PIPELINE")
    print("="*80)
    sd.generate_cuts(args.input_shape_layout, object_free_results['output_dir'])
    input_mask_folder = os.path.join(object_free_results['output_dir'], 'masked_objects')
    os.makedirs(input_mask_folder, exist_ok=True)
    image_pool = os.path.join(object_free_results['output_dir'], 'cropped_objects')
    cm.batch_create_masks(image_pool, input_mask_folder, mask_type='simple')
    print("Starting shape assembly optimization...")
    so.optimization(args.input_shape_layout, input_mask_folder, object_free_results['output_dir'])
    print("Rendering collage...")
    ca.render_collage(image_pool, object_free_results['output_dir'], args.scaling_factor)
    print("Evaluating pipeline output...")
    metrics = evaluation.evaluate_pipeline_output(object_free_results['output_dir'],
                                                   args.input_shape_layout)
    print("Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")
    print("="*60)

if __name__ == "__main__":
    main()
    


"""



# 1) PySceneDetect + LPIPS(Alex)
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 3 --max_frames_per_scene 100 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_psd_lpips \
  --export_preview

# 1) PySceneDetect + DISTS(Alex)
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend dists --lpips_net alex \
  --sample_stride 3 --max_frames_per_scene 100 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_psd_dists \
  --export_preview

# 2) TransNetV2 (PyTorch) + DISTS
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend transnetv2  \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --distance_backend dists --dists_as_distance 1 \
  --sample_stride 8 --max_frames_per_scene 40 \
  --keyframes_per_scene 2 --nms_radius 4 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_tv2_dists

python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 3 --max_frames_per_scene 100 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_psd_lpips \
  --export_preview \
  --keyframe_selector random --random_seed 42
rgs.out_dir, 'scenes.json')}")
    print(f"  • Scenes CSV  : {os.path.join(args.out_dir, 'scenes.csv')}")
    print(f"  • Keyframes CSV: {os.path.join(args.out_dir, 'keyframes.csv')}")
    if args.export_preview:
        print(f"  • Scene previews: {preview_dir}")
    print(f"  • Keyframe images: {key_dir}")
    if args.run_object_detection and detection_results:
        print(f"  • Object detections: {os.path.join(args.out_dir, 'object_detections')}")
    print("="*60)



# 1) PySceneDetect + LPIPS(Alex)
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 10 --max_frames_per_scene 30 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_psd_lpips \
  --export_preview

# 1) PySceneDetect + DISTS(Alex)
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend dists --lpips_net alex \
  --sample_stride 3 --max_frames_per_scene 100 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_psd_dists \
  --export_preview

# 2) TransNetV2 (PyTorch) + DISTS
python pipeline.py \
  --video samples/Sakuga/10736.mp4 \
  --backend transnetv2  \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --distance_backend dists --dists_as_distance 1 \
  --sample_stride 8 --max_frames_per_scene 40 \
  --keyframes_per_scene 2 --nms_radius 4 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_tv2_dists

# 3) With Object-Free Pipeline
python pipeline.py \
  --video ./data/samples/Sakuga/6261.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 10 --max_frames_per_scene 30 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_with_object_free \
  --export_preview \
  --run_object_free_pipeline \
  --detection_config objectfree/config.yaml \
  --detection_checkpoint ./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt

# 3) With layout pipeline
python layout_decomposer_pipeline.py \
  --video ./data/samples/Sakuga/6261.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 10 --max_frames_per_scene 30 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_with_object_free \
  --export_preview \
  --run_object_free_pipeline \
  --detection_config objectfree/config.yaml \
  --detection_checkpoint ./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt


# 3) With layout pipeline
python layout_decomposer_pipeline.py \
  --video ./data/samples/Sakuga/6261.mp4 \
  --backend transnetv2  \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --distance_backend dists --dists_as_distance 1 \
  --sample_stride 8 --max_frames_per_scene 40 \
  --keyframes_per_scene 2 --nms_radius 4 \
  --resize_w 320 --resize_h 180 \
  --out_dir data/outputs/run_collage \
  --export_preview \
  --run_object_free_pipeline \
  --detection_config objectfree/config.yaml \
  --detection_checkpoint ./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt \
  --input_shape_layout repos/Colla/input_data/layout/baby.png \
  --input_mask_folder repos/Colla/input_data/image_collections/children_mask \
  --shape repos/Colla/input_data/layout/baby.png
"""


