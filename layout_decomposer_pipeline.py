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
import json
import gc
import time
import traceback
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
# Cartoon Detection Pipeline Integration (YOLOE)
# ------------------------------
def run_cartoon_detection_pipeline(keyframes_folder, output_base, device="cuda", config_path="objectfree/detector_config.yaml"):
    """Run cartoon character detection using DetectorCartoon class"""
    
    from objectfree.detector_cartoon import DetectorCartoon
    import yaml
    import tempfile
    
    # Load config and override paths with absolute paths
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override paths with absolute paths to avoid relative path issues
    config['input_path'] = os.path.abspath(keyframes_folder)
    config['type_content'] = 'image'
    config['save_path'] = os.path.abspath(output_base)
    
    # Override model paths to use absolute paths from objectfree/weight_model
    base_weight_dir = os.path.abspath("objectfree/weight_model")
    # Use new trained weights from train3
    config['model_path'] = os.path.abspath("objectfree/yoloe/runs/detect/train3/weights/best.pt")
    config['pe_path'] = os.path.join(base_weight_dir, "character-pe.pt")
    config['mobileclip_model_path'] = os.path.join(base_weight_dir, "mobileclip_blt.pt")
    
    print(f"[Cartoon Detection] Input: {config['input_path']}")
    print(f"[Cartoon Detection] Output: {config['save_path']}")
    print(f"[Cartoon Detection] Model: {config['model_path']}")
    
    # Save modified config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(config, tmp)
        temp_config_path = tmp.name
    
    try:
        # Initialize detector with modified config
        detector = DetectorCartoon(config_path=temp_config_path)
        
        # Run detection
        results = detector.forward(save_results=True)
    finally:
        # Cleanup temp config
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
    
    # Return results summary
    if isinstance(results, list):
        if not results:
            return {
                "output_dir": output_base,
                "total_images": 0,
                "total_detections": 0,
                "results": []
            }
        total_detections = sum(len(r.boxes) for r in results if hasattr(r, 'boxes'))
        return {
            "output_dir": output_base,
            "total_images": len(results),
            "total_detections": total_detections,
            "results": results
        }
    else:
        total_detections = len(results.boxes) if hasattr(results, 'boxes') else 0
        return {
            "output_dir": output_base,
            "total_images": 1,
            "total_detections": total_detections,
            "results": results
        }

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

    # Cartoon character detection
    ap.add_argument("--run_object_free_pipeline", action="store_true",
                    help="Run cartoon character detection on extracted keyframes.")
    ap.add_argument("--detection_config", type=str, default=None,
                    help="Path to cartoon detection config file (default: objectfree/detector_config.yaml).")
    ap.add_argument("--detection_device", type=str, default=None,
                    help="Device for cartoon detection ('cuda'/'cpu').")
    
    # Colla layout decomposer pipeline args
    ap.add_argument("--input_shape_layout", type=str, default="repos/Colla/input_data/layout/baby.png",  help="Input shape layout image path.")
    ap.add_argument("--input_mask_folder", type=str, default="repos/Colla/input_data/image_collections/children_mask", help="Input mask folder path.")
    ap.add_argument("--scaling_factor", type=int, default=1, help="Scaling factor for collage rendering (default 1 to avoid segfault with many images).")
    ap.add_argument("--center_salient", action="store_true", default=True,
                    help="Use U2-Net to center salient region at shape centroid (default: True).")
    ap.add_argument("--no_center_salient", action="store_false", dest="center_salient",
                    help="Disable U2-Net salient centering, use original foreground positioning.")
    ap.add_argument("--salient_fit_ratio", type=float, default=0.65,
                    help="How much of shape the salient region should occupy (0.0-1.0, default: 0.65).")
    ap.add_argument("--use_smart_crop", action="store_true", default=True,
                    help="Use fast smart crop instead of mesh warp (default: True, ~10x faster).")
    ap.add_argument("--use_mesh_warp", action="store_false", dest="use_smart_crop",
                    help="Use mesh warp instead of smart crop (slower but more flexible).")
    ap.add_argument("--use_fast_saliency", action="store_true", default=True,
                    help="Use fast heuristic saliency instead of U2-Net (default: True, ~100x faster).")
    ap.add_argument("--use_u2net", action="store_false", dest="use_fast_saliency",
                    help="Use U2-Net saliency instead of fast heuristic (slower but more accurate).")
    
    # Object detection mode (for cartoon characters)
    ap.add_argument("--use_object_detection", action="store_true", default=False,
                    help="Use YOLOE object detection instead of saliency (for cartoon characters).")
    ap.add_argument("--detection_threshold", type=float, default=0.25,
                    help="Detection confidence threshold (default: 0.25).")
    ap.add_argument("--enable_seam_carving", action="store_true", default=True,
                    help="Apply seam carving when objects are spread out (default: True).")
    ap.add_argument("--disable_seam_carving", action="store_false", dest="enable_seam_carving",
                    help="Disable seam carving for spread-out objects.")

    return ap


# ------------------------------
# Main
# ------------------------------
def main():
    args = build_argparser().parse_args()
    
    # Use output directory as specified (no timestamp modification)
    # args.out_dir remains as user specified

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

    # Run cartoon character detection pipeline (optional)
    detection_results = None
    if args.run_object_free_pipeline:
        print("\n" + "="*80)
        print("RUNNING CARTOON CHARACTER DETECTION")
        print("="*80)
        
        # Determine device for detection
        detection_device_str = args.detection_device or args.distance_device or "cuda"
        
        # Create base output directory for detection results
        detection_base_dir = os.path.join(args.out_dir, "cartoon_detection")
        ensure_dir(detection_base_dir)
        
        try:
            detection_results = run_cartoon_detection_pipeline(
                keyframes_folder=key_dir,
                output_base=detection_base_dir,
                device=detection_device_str,
                config_path=args.detection_config or "objectfree/detector_config.yaml"
            )
            
            if detection_results:
                print(f"\n[SUCCESS] Cartoon detection completed!")
                print(f"  • Results: {detection_results['output_dir']}")
                print(f"  • Images processed: {detection_results['total_images']}")
                print(f"  • Total detections: {detection_results['total_detections']}")
            else:
                print(f"[WARN] Cartoon detection failed!")
                
        except Exception as e:
            print(f"[ERROR] Cartoon detection failed: {e}")
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
    if args.run_object_free_pipeline and detection_results:
        print(f"  • Cartoon detection: {detection_results['output_dir']}")
    print("="*60)

    # Colla pipeline
    print("\n" + "="*80)
    print("RUNNING COLLA LAYOUT DECOMPOSER PIPELINE")
    print("="*80)
    
    # === FIX: Use keyframes folder as input instead of object_free output ===
    # Setup paths - USE KEYFRAMES FOLDER DIRECTLY
    colla_output_dir = os.path.join(args.out_dir, "colla_layout")
    ensure_dir(colla_output_dir)
    
    input_shape = args.input_shape_layout
    # Use exported keyframes as input images
    input_image_collection_folder = key_dir  # This is outputs/.../keyframes/
    # Create masks folder for these keyframes
    input_mask_folder = os.path.join(colla_output_dir, 'keyframe_masks')
    
    print(f"\n[Colla Input Verification]")
    print(f"  input_shape: {input_shape}")
    print(f"  input_image_collection: {input_image_collection_folder}")
    print(f"  output_dir: {colla_output_dir}")
    print(f"  scaling_factor: {args.scaling_factor}")
    print(f"  use_object_detection: {args.use_object_detection}")
    if not args.use_object_detection:
        print(f"  input_mask_folder: {input_mask_folder}")
    
    # Verify keyframe images exist
    if not os.path.exists(input_image_collection_folder):
        raise FileNotFoundError(f"Keyframes folder not found: {input_image_collection_folder}")
    
    keyframe_files = [f for f in os.listdir(input_image_collection_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"  Found {len(keyframe_files)} keyframe images")
    
    if len(keyframe_files) == 0:
        raise FileNotFoundError(f"No keyframe images in {input_image_collection_folder}")
    
    if len(keyframe_files) > 12:
        print(f"  [WARN] Many images ({len(keyframe_files)}), may cause segfault with Colla pipeline")
        print(f"  [SUGGESTION] Reduce --keyframes_per_scene to 1 or use fewer scenes")
    
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
        
        # Verify final_cut.json was created (this is what generate_cuts produces)
        final_cut_path = os.path.join(colla_output_dir, 'final_cut.json')
        if not os.path.exists(final_cut_path):
            print(f"  [ERROR] final_cut.json not created at {final_cut_path}")
            print(f"  [DEBUG] Checking output_dir contents...")
            if os.path.exists(colla_output_dir):
                files = [f for f in os.listdir(colla_output_dir) if f.endswith('.json')]
                print(f"  [DEBUG] JSON files in output_dir: {files}")
            raise FileNotFoundError(f"final_cut.json not found - shape decomposition failed")
        else:
            print(f"  ✓ final_cut.json created successfully")
            # Read and show cut count
            import json
            with open(final_cut_path, 'r') as f:
                cuts = json.load(f)
            print(f"  Number of cuts: {len(cuts)}")
    except Exception as e:
        print(f"[ERROR] Shape decomposition failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ============================================
    # STEP 2: Create Masks from Keyframe Images (skip if using object detection)
    # ============================================
    if args.use_object_detection:
        print(f"\n[STEP 2] SKIPPED - Using object detection mode (no mask folder needed)")
        input_mask_folder = None  # Object detection mode doesn't need masks
        mask_files = []
    else:
        print(f"\n[STEP 2] Creating masks from keyframe images")
        os.makedirs(input_mask_folder, exist_ok=True)
        
        print(f"  Creating masks for {len(keyframe_files)} keyframe images...")
        cm.batch_create_masks(input_image_collection_folder, input_mask_folder, mask_type='simple')
        
        # Verify masks were created
        mask_files = [f for f in os.listdir(input_mask_folder) if f.endswith('.png')]
        print(f"  Created {len(mask_files)} masks")
        
        if len(mask_files) == 0:
            raise FileNotFoundError(f"Failed to create masks in {input_mask_folder}")
        
        if len(mask_files) != len(keyframe_files):
            print(f"  [WARN] Mask count ({len(mask_files)}) != keyframe count ({len(keyframe_files)})")
            print(f"  [INFO] This may happen if some keyframes failed mask generation")
    
    # ============================================
    # STEP 3: Spatial Assignment Optimization
    # ============================================
    print(f"\n[STEP 3] Spatial assignment optimization")
    if args.use_object_detection:
        print(f"  Mode: Object Detection (YOLOE)")
        print(f"  Detection threshold: {args.detection_threshold}")
    else:
        print(f"  Processing {len(mask_files)} masks")
        print(f"  Input mask folder: {input_mask_folder}")
    print(f"  Input shape mask: {shape_mask_path}")
    print(f"  Output dir: {colla_output_dir}")
    
    if not args.use_object_detection and len(mask_files) > 12:
        print(f"  [WARN] Many masks ({len(mask_files)}), high risk of segfault")
    
    # Verify final_cut.json exists (created by STEP 1)
    final_cut_path = os.path.join(colla_output_dir, 'final_cut.json')
    if not os.path.exists(final_cut_path):
        raise FileNotFoundError(f"final_cut.json not found at {final_cut_path}. STEP 1 may have failed.")
    
    # Run optimization - this will create slicing_result.json
    try:
        # Pass image folder and object detection parameters
        so.optimization(
            shape_mask_path, 
            input_mask_folder,  # Can be None in object detection mode
            colla_output_dir, 
            image_folder=input_image_collection_folder,
            use_object_detection=args.use_object_detection,
            detection_threshold=args.detection_threshold
        )
        print("  ✓ Optimization completed")
        
        # Verify slicing_result.json was created
        slicing_result_path = os.path.join(colla_output_dir, 'slicing_result.json')
        if not os.path.exists(slicing_result_path):
            raise FileNotFoundError(f"slicing_result.json not created after optimization")
        print(f"  ✓ slicing_result.json created successfully")
        
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
    
    # Determine mode string for logging
    if args.use_object_detection:
        mode_str = f"ObjDet+MeshWarp (thresh={args.detection_threshold}, seam_carving={args.enable_seam_carving})"
    elif args.use_smart_crop and args.use_fast_saliency:
        mode_str = "SmartCrop+FastSaliency"
    elif args.use_smart_crop:
        mode_str = "SmartCrop+U2Net"
    else:
        mode_str = "MeshWarp"
    print(f"  Mode: {mode_str}")
    print(f"  Center salient: {args.center_salient}, Fit ratio: {args.salient_fit_ratio}")
    
    try:
        ca.render_collage(
            input_image_collection_folder, colla_output_dir, args.scaling_factor,
            enable_debug=True,  # Enable debug visualization
            center_salient=args.center_salient, salient_fit_ratio=args.salient_fit_ratio,
            use_smart_crop=args.use_smart_crop, use_fast_saliency=args.use_fast_saliency,
            use_object_detection=args.use_object_detection,
            detection_threshold=args.detection_threshold,
            enable_seam_carving=args.enable_seam_carving
        )
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
    print(f"  Collage: {os.path.join(colla_output_dir, 'collage.png')}")
    print(f"  Collage with borders: {os.path.join(colla_output_dir, 'collage_white_space.png')}")
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


