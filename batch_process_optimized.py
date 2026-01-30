#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processing Optimization for Layout Decomposer DSN Pipeline
- Batch video processing with parallel scene detection
- Batch image processing with vectorized embedding extraction
- Memory-efficient multiprocessing for Colla pipeline
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import shutil


# =========================================================
# VIDEO BATCH PROCESSOR
# =========================================================

def process_video_batch(
    video_paths: List[str],
    output_base: str,
    checkpoint: str,
    num_workers: int = 2,
    **kwargs
) -> List[Tuple[str, bool, str]]:
    """
    Process multiple videos in parallel.
    
    Args:
        video_paths: List of video file paths
        output_base: Output base directory
        checkpoint: DSN checkpoint path
        num_workers: Number of parallel workers
        **kwargs: Additional arguments for pipeline
    
    Returns:
        List of (video_path, success, output_dir) tuples
    """
    results = []
    
    def process_single_video(video_path):
        try:
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = f"{output_base}_{video_name}_{timestamp}"
            
            # Build command
            cmd = [
                "python", "layout_decomposer_dsn_pipeline.py",
                "--video", video_path,
                "--out_dir", out_dir,
                "--checkpoint", checkpoint,
            ]
            
            # Add kwargs
            for key, value in kwargs.items():
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(value)])
            
            # Run pipeline
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                return (video_path, True, out_dir)
            else:
                print(f"[ERROR] {video_path}: {result.stderr[:200]}")
                return (video_path, False, "")
        
        except Exception as e:
            print(f"[ERROR] Processing {video_path}: {str(e)[:200]}")
            return (video_path, False, "")

    # Process videos in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_video, vp): vp 
            for vp in video_paths
        }
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return results


# =========================================================
# SCENE DETECTION BATCH PROCESSOR
# =========================================================

def batch_detect_scenes(
    video_paths: List[str],
    backend: str = "transnetv2",
    num_workers: int = 2,
    **det_kwargs
) -> dict:
    """
    Detect scenes in multiple videos in parallel.
    
    Returns dict: {video_path: List[Scene]}
    """
    from src.scene_detection import detect_scenes_generic
    
    scenes_dict = {}
    
    def detect_video_scenes(video_path):
        try:
            scenes = detect_scenes_generic(
                video_path,
                backend,
                **det_kwargs
            )
            return (video_path, scenes)
        except Exception as e:
            print(f"[ERROR] Scene detection for {video_path}: {e}")
            return (video_path, [])

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(detect_video_scenes, vp): vp 
            for vp in video_paths
        }
        
        for future in as_completed(futures):
            video_path, scenes = future.result()
            scenes_dict[video_path] = scenes
    
    return scenes_dict


# =========================================================
# IMAGE BATCH PROCESSOR
# =========================================================

def batch_embed_images(
    image_paths: List[str],
    embedder_name: str = "clip_vitb32",
    device: str = "cuda",
    resize_w: int = 320,
    resize_h: int = 180,
    batch_size: int = 32,
    num_workers: int = 4
) -> dict:
    """
    Extract embeddings for multiple images in batches.
    
    Returns dict: {image_path: np.ndarray(embedding)}
    """
    import numpy as np
    import cv2
    import torch
    from tqdm import tqdm
    
    # Import embedder
    from layout_decomposer_dsn_pipeline import build_embedder
    
    embeddings = {}
    encode, emb_dim = build_embedder(embedder_name, device=device)
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch embedding"):
        batch_paths = image_paths[i:i+batch_size]
        batch_frames = []
        
        for img_path in batch_paths:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (resize_w, resize_h))
                batch_frames.append(img)
        
        if batch_frames:
            # Batch encode
            feats = encode(batch_frames)
            
            for img_path, feat in zip(batch_paths, feats):
                embeddings[img_path] = feat
    
    return embeddings


# =========================================================
# COLLA PIPELINE BATCH PROCESSOR
# =========================================================

def batch_process_colla(
    keyframe_folders: List[str],
    input_shape_layout: str,
    output_base: str,
    num_workers: int = 1,
    **colla_kwargs
) -> List[Tuple[str, bool, str]]:
    """
    Process Colla layout decomposition for multiple keyframe folders in parallel.
    """
    results = []
    
    def process_colla_single(keyframe_folder):
        try:
            folder_name = Path(keyframe_folder).name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(output_base, f"colla_{folder_name}_{timestamp}")
            
            # Prepare paths
            keyframe_masks_dir = os.path.join(out_dir, "keyframe_masks")
            os.makedirs(keyframe_masks_dir, exist_ok=True)
            
            # Import here to avoid circular imports
            import repos.Colla.shape_decomposition as sd
            import repos.Colla.sas_optimization as so
            import repos.Colla.collage_assembly as ca
            import repos.Colla.create_masks as cm
            from repos.Colla.utils.get_mask import predict_mask, preprocess_image, refine_mask, net
            
            # 1. Get shape mask
            # (Assuming input_shape_layout is provided)
            shape_mask = preprocess_image(input_shape_layout)
            pred_mask = predict_mask(shape_mask, net, device='cuda')
            mask_refined = refine_mask(pred_mask, shape_mask.shape[0], shape_mask.shape[1])
            shape_mask_path = os.path.join(out_dir, "shape_mask_refined.png")
            import cv2
            cv2.imwrite(shape_mask_path, mask_refined)
            
            # 2. Shape decomposition
            sd.decompose_shape(shape_mask_path, out_dir)
            
            # 3. Create masks for keyframes
            cm.batch_create_masks(keyframe_folder, keyframe_masks_dir, mask_type='center')
            
            # 4. Optimization
            so.optimization(shape_mask_path, keyframe_masks_dir, out_dir, image_folder=keyframe_folder)
            
            # 5. Render
            ca.render_collage(
                keyframe_folder, out_dir,
                scaling_factor=colla_kwargs.get('scaling_factor', 1),
                **{k: v for k, v in colla_kwargs.items() if k != 'scaling_factor'}
            )
            
            return (keyframe_folder, True, out_dir)
        
        except Exception as e:
            print(f"[ERROR] Colla processing {keyframe_folder}: {str(e)[:200]}")
            return (keyframe_folder, False, "")

    # Process in parallel (limit to 1-2 workers for memory)
    with ProcessPoolExecutor(max_workers=min(num_workers, 2)) as executor:
        futures = {
            executor.submit(process_colla_single, kf): kf 
            for kf in keyframe_folders
        }
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return results


# =========================================================
# MAIN BATCH RUNNER
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos/images with DSN pipeline"
    )
    parser.add_argument("--mode", choices=["video", "image", "colla"], 
                       default="video", help="Processing mode")
    parser.add_argument("--input", required=True, 
                       help="Input file or folder")
    parser.add_argument("--output_base", default="outputs", 
                       help="Output base directory")
    parser.add_argument("--checkpoint", default="runs/dsn_advanced_v1/dsn_checkpoint_ep8.pt",
                       help="DSN checkpoint path")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of parallel workers")
    parser.add_argument("--device", default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for embeddings")
    
    # Additional options
    parser.add_argument("--embedder", default="clip_vitb32",
                       help="Embedding model")
    parser.add_argument("--input_shape_layout", default="repos/Colla/input_data/layout/baby.png",
                       help="Shape layout for Colla")
    parser.add_argument("--scaling_factor", type=int, default=1,
                       help="Scaling factor for collage")
    
    args = parser.parse_args()

    input_path = Path(args.input)
    
    if args.mode == "video":
        if input_path.is_dir():
            videos = sorted(input_path.glob("*.mp4"))
        else:
            videos = [input_path]
        
        print(f"[INFO] Found {len(videos)} videos")
        
        results = batch_process_video_batch(
            [str(v) for v in videos],
            args.output_base,
            args.checkpoint,
            num_workers=args.num_workers,
            device=args.device,
            embedder=args.embedder
        )
        
        print("\n========== RESULTS ==========")
        success_count = sum(1 for _, success, _ in results if success)
        print(f"Successfully processed: {success_count}/{len(results)}")
        for video, success, out_dir in results:
            status = "✅" if success else "❌"
            print(f"{status} {Path(video).name}: {out_dir}")
    
    elif args.mode == "image":
        if input_path.is_dir():
            images = sorted(list(input_path.glob("*.jpg")) + 
                           list(input_path.glob("*.png")))
        else:
            images = [input_path]
        
        print(f"[INFO] Found {len(images)} images")
        
        embeddings = batch_embed_images(
            [str(img) for img in images],
            embedder_name=args.embedder,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        print(f"[INFO] Extracted {len(embeddings)} embeddings")
    
    elif args.mode == "colla":
        if input_path.is_dir():
            keyframe_folders = sorted(input_path.glob("*/keyframes"))
        else:
            keyframe_folders = [input_path]
        
        print(f"[INFO] Processing {len(keyframe_folders)} folders")
        
        results = batch_process_colla(
            [str(kf) for kf in keyframe_folders],
            args.input_shape_layout,
            args.output_base,
            num_workers=args.num_workers,
            scaling_factor=args.scaling_factor
        )
        
        print("\n========== RESULTS ==========")
        success_count = sum(1 for _, success, _ in results if success)
        print(f"Successfully processed: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
