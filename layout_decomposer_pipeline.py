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
python temporal_layout_composer_unified.py \
  --mode run \
  --keyframes-dir /home/serverai/ltdoanh/LayoutGeneration/outputs/run_with_object_free_6261_20251101_133231/object_free_evaluation/keyframes \
  --shape-image optimal_layout.png \
  --min-len 3 \
  --max-len 4 \
  --w-clip 0.8 \
  --w-iqa 0.2from objectfree import complete_pipeline_advanced
from utils.io import *

# Import Colla modules
sys.path.append('repos/Colla')
import repos.Colla.shape_decomposition as sd
import repos.Colla.sas_optimization as so
import repos.Colla.collage_assembly as ca
import repos.Colla.create_masks as cm
from repos.Colla import evaluation
from repos.Colla.utils.get_mask import predict_mask, preprocess_image, refine_mask, net
import cv2



# ------------------------------
# Object-Free Pipeline Integration
# ------------------------------
def run_complete_object_free_pipeline(keyframes_folder, output_base, device="cuda", config_path="objectfree/config.yaml", checkpoint_path="./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"):
    """Run complete object-free pipeline using CompletePipeline class"""
    
    # Initialize pipeline
    pipeline = complete_pipeline_advanced.AdvancedBBoxPipeline(device=device, output_dir=output_base)
    pipeline.initialize_detectors()
    
    # Override config and checkpoint paths
    pipeline.object_detector.config_path = config_path
    pipeline.object_detector.checkpoint_path = checkpoint_path
    
    # Process the folder
    result = pipeline.process_single_folder(keyframes_folder, output_base)
    
    return result

def prepare_colla_pipeline():
    """Prepare system resources for Colla pipeline to avoid segmentation faults."""
    print("[prepare_colla_pipeline] Clearing resources...")
    
    # Clear GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  ✓ Cleared CUDA cache and synchronized")
    except:
        pass
    
    # Force garbage collection
    import gc
    gc.collect()
    print("  ✓ Forced garbage collection")
    
    print("[prepare_colla_pipeline] Ready to run Colla pipeline")

def get_mask_from_image(input_image_path, output_dir):
    """Generate mask from RGB image using U2NET.
    
    Args:
        input_image_path: Path to input RGB image (can be .jpg, .png, etc.)
        output_dir: Directory to save the refined mask
        
    Returns:
        Path to the refined mask file
    """
    print(f"[get_mask_from_image] Processing input image: {input_image_path}")
    
    # Load RGB image
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {input_image_path}")
    
    print(f"  Image shape: {image.shape}")
    
    # Preprocess image for U2NET
    inputs, orig_h, orig_w = preprocess_image(image)
    print(f"  Original size: {orig_w}x{orig_h}")
    
    # Predict mask using U2NET
    print("  Predicting mask with U2NET...")
    pred_mask = predict_mask(net, inputs)
    print(f"  Prediction shape: {pred_mask.shape}")
    
    # Refine mask (remove noise, smooth edges, keep largest component)
    print("  Refining mask...")
    mask_refined = refine_mask(pred_mask, orig_h, orig_w)
    print(f"  Refined mask shape: {mask_refined.shape}")
    
    # Save refined mask
    shape_mask_path = os.path.join(output_dir, "shape_mask_refined.png")
    cv2.imwrite(shape_mask_path, mask_refined)
    print(f"  Saved refined mask to: {shape_mask_path}")
    
    return shape_mask_path

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
                    help="How many keyframes to pick per scene (default 1 to avoid segfault with many images).")
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
    ap.add_argument("--scaling_factor", type=int, default=1, help="Scaling factor for collage rendering (default 1 to avoid segfault with many images).")

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
    
    # Setup paths
    colla_output_dir = object_free_results['output_dir']
    input_shape = args.input_shape_layout
    input_mask_folder = os.path.join(colla_output_dir, 'masked_objects')
    input_image_collection_folder = os.path.join(colla_output_dir, 'cropped_objects')
    
    print(f"\n[Colla Input Verification]")
    print(f"  input_shape: {input_shape}")
    print(f"  input_mask_folder: {input_mask_folder}")
    print(f"  input_image_collection: {input_image_collection_folder}")
    print(f"  output_dir: {colla_output_dir}")
    print(f"  scaling_factor: {args.scaling_factor}")
    
    # Verify cropped objects exist
    if not os.path.exists(input_image_collection_folder):
        raise FileNotFoundError(f"Cropped objects folder not found: {input_image_collection_folder}")
    
    cropped_files = [f for f in os.listdir(input_image_collection_folder) if f.endswith(('.png', '.jpg'))]
    print(f"  Found {len(cropped_files)} cropped images")
    
    if len(cropped_files) == 0:
        raise FileNotFoundError(f"No cropped images in {input_image_collection_folder}")
    
    if len(cropped_files) > 12:
        print(f"  [WARN] Many images ({len(cropped_files)}), may cause segfault")
    
    # CRITICAL: Free all previous models before Colla
    print("\n[Freeing Memory Before Colla Pipeline]")
    try:
        # Delete all heavy objects
        if 'metric' in locals():
            del metric
        if 'selector' in locals():
            del selector
        if 'detector' in locals():
            del detector
        
        # Force garbage collection
        import gc
        gc.collect()
        print("  ✓ Freed Python objects")
        
        # Clear CUDA memory completely
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"  ✓ Cleared CUDA cache (freed {torch.cuda.memory_allocated() / 1e9:.2f} GB)")
        
        # Wait a bit for system to stabilize
        import time
        time.sleep(2)
        print("  ✓ Memory cleanup completed")
        
    except Exception as e:
        print(f"  [WARN] Memory cleanup had issues: {e}")
    
    # Prepare resources
    prepare_colla_pipeline()
    
    # ============================================
    # STEP 0: Generate Mask from Input RGB Image
    # ============================================
    print(f"\n[STEP 0] Generating mask from input shape image")
    shape_mask_path = get_mask_from_image(input_shape, colla_output_dir)
    
    # ============================================
    # STEP 1: Shape Decomposition
    # ============================================
    print(f"\n[STEP 1] Shape decomposition")
    try:
        sd.generate_cuts(shape_mask_path, colla_output_dir)
        print("  ✓ Shape decomposition completed")
        
        # Verify slicing_result.json was created
        slicing_result_path = os.path.join(colla_output_dir, 'slicing_result.json')
        if not os.path.exists(slicing_result_path):
            print(f"  [WARN] slicing_result.json not created at {slicing_result_path}")
            print(f"  [DEBUG] Checking output_dir contents...")
            if os.path.exists(colla_output_dir):
                files = [f for f in os.listdir(colla_output_dir) if f.endswith('.json')]
                print(f"  [DEBUG] JSON files in output_dir: {files}")
        else:
            print(f"  ✓ slicing_result.json created successfully")
    except Exception as e:
        print(f"[ERROR] Shape decomposition failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ============================================
    # STEP 2: Create Masks from Cropped Objects
    # ============================================
    print(f"\n[STEP 2] Creating masks from cropped objects")
    os.makedirs(input_mask_folder, exist_ok=True)
    
    print(f"  Creating masks for {len(cropped_files)} cropped images...")
    cm.batch_create_masks(input_image_collection_folder, input_mask_folder, mask_type='simple')
    
    # Verify masks were created
    mask_files = [f for f in os.listdir(input_mask_folder) if f.endswith('.png')]
    print(f"  Created {len(mask_files)} masks")
    
    if len(mask_files) == 0:
        raise FileNotFoundError(f"Failed to create masks in {input_mask_folder}")
    
    if len(mask_files) != len(cropped_files):
        print(f"  [WARN] Mask count ({len(mask_files)}) != image count ({len(cropped_files)})")
    
    # ============================================
    # STEP 3: Spatial Assignment Optimization
    # ============================================
    print(f"\n[STEP 3] Spatial assignment optimization")
    print(f"  Processing {len(mask_files)} masks")
    
    if len(mask_files) > 12:
        print(f"  [WARN] Many masks, high risk of segfault")
    
    # Validate tree structure
    try:
        import json
        # slicing_result.json is created by sd.generate_cuts in colla_output_dir
        # But we need to check where it actually gets created
        # Based on STEP 1: sd.generate_cuts(shape_mask_path, colla_output_dir)
        # It should create slicing_result.json in colla_output_dir
        slicing_result_path = os.path.join(colla_output_dir, 'slicing_result.json')
        
        # Debug: check what files were actually created
        if not os.path.exists(slicing_result_path):
            print(f"  [DEBUG] Looking for slicing_result.json in: {colla_output_dir}")
            if os.path.exists(colla_output_dir):
                files = os.listdir(colla_output_dir)
                print(f"  [DEBUG] Files in output_dir: {files}")
            raise FileNotFoundError(f"slicing_result.json not found at {slicing_result_path}")
        
        with open(slicing_result_path, 'r') as f:
            slicing_data = json.load(f)
        
        def count_leaves(node):
            if 'children' not in node or not node['children']:
                return 1
            return sum(count_leaves(child) for child in node['children'])
        
        def get_tree_height(node):
            if 'children' not in node or not node['children']:
                return 0
            return 1 + max(get_tree_height(child) for child in node['children'])
        
        if 'tree' in slicing_data:
            tree_leaves = count_leaves(slicing_data['tree'])
            tree_height = get_tree_height(slicing_data['tree'])
            print(f"  Tree structure: height={tree_height}, leaves={tree_leaves}")
            print(f"  Available masks: {len(mask_files)}")
            
            # Critical validation
            if tree_height == 0:
                raise ValueError(f"Tree height is 0 - shape decomposition failed to create proper tree structure. This will cause segfault.")
            
            if tree_leaves == 0:
                raise ValueError(f"Tree has no leaves - cannot assign images. This will cause segfault.")
            
            if tree_leaves > len(mask_files):
                print(f"  [WARN] Tree needs {tree_leaves} images but only {len(mask_files)} available")
                print(f"  [SUGGESTION] Increase --keyframes_per_scene to get more images")
            
            if tree_leaves < len(mask_files):
                print(f"  [INFO] Tree has {tree_leaves} leaves but {len(mask_files)} images available")
                print(f"  [INFO] Optimization will select best {tree_leaves} images")
                
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise
    except ValueError as e:
        print(f"[ERROR] Tree validation failed: {e}")
        print(f"[SOLUTION] The shape layout image may be too simple or too complex.")
        print(f"[SUGGESTION] Try a different shape layout image with clearer structure.")
        raise
    except Exception as e:
        print(f"  [WARN] Could not validate tree: {e}")
    
    # Run optimization with validation
    try:
        so.optimization(shape_mask_path, input_mask_folder, colla_output_dir)
        print("  ✓ Optimization completed")
    except Exception as e:
        print(f"[ERROR] Optimization failed: {e}")
        print(f"[TIP] This is likely caused by:")
        print(f"  1. Tree structure mismatch (tree leaves != available images)")
        print(f"  2. Invalid shape layout image")
        print(f"  3. Memory overflow with too many images")
        import traceback
        traceback.print_exc()
        raise
    
    # ============================================
    # STEP 4: Collage Assembly & Rendering
    # ============================================
    print(f"\n[STEP 4] Collage assembly & rendering")
    
    # Verify slicing result exists
    import json
    slicing_result_path = os.path.join(colla_output_dir, 'slicing_result.json')
    if not os.path.exists(slicing_result_path):
        raise FileNotFoundError(f"slicing_result.json not found at {slicing_result_path}")
    
    # Check canvas size
    with open(slicing_result_path, 'r') as f:
        layout = json.load(f)
    
    canvas_w = layout['width'] * args.scaling_factor
    canvas_h = layout['height'] * args.scaling_factor
    canvas_size_mb = (canvas_w * canvas_h * 4) / 1e6
    
    print(f"  Canvas: {canvas_w}x{canvas_h} ({canvas_size_mb:.1f} MB)")
    print(f"  Images: {len(layout.get('images', []))}, Parts: {len(layout.get('parts', []))}")
    
    if canvas_size_mb > 500:
        print(f"  [WARN] Large canvas ({canvas_size_mb:.1f} MB), may be slow")
    
    try:
        ca.render_collage(input_image_collection_folder, colla_output_dir, args.scaling_factor)
        print("  ✓ Rendering completed")
    except Exception as e:
        print(f"[ERROR] Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ============================================
    # STEP 5: Evaluation (Optional)
    # ============================================
    print(f"\n[STEP 5] Evaluating results")
    try:
        metrics = evaluation.evaluate_pipeline_output(colla_output_dir, shape_mask_path)
        print("  Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"    {metric_name}: {metric_value}")
    except Exception as e:
        print(f"  [WARN] Evaluation failed: {e}")
    
    print("\n" + "="*80)
    print("COLLA PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"  Output directory: {colla_output_dir}")
    print(f"  Final collage: {os.path.join(colla_output_dir, 'final_collage.png')}")
    print("="*80)
    
    print("\n[DONE] Full pipeline completed!")

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
  --input_shape_layout repos/Colla/input_data/image_collections/cars/01.jpg \
  --scaling_factor 2


test case chạy đc
python layout_decomposer_pipeline.py \
  --video ./data/samples/Sakuga/14652.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 10 --max_frames_per_scene 30 \
  --keyframes_per_scene 2 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir data/outputs/run_collage \
  --export_preview \
  --run_object_free_pipeline \
  --detection_config objectfree/config.yaml \
  --detection_checkpoint ./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt \
  --input_shape_layout repos/Colla/input_data/image_collections/cars/01.jpg \
  --scaling_factor 1
"""


