#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layout Decomposer Final Pipeline with DSN-based Keyframe Extraction
- Scene detection via pluggable backends (registry): pyscenedetect, transnetv2
- Keyframe selection via DSN (Deep Summarization Network)
- Cartoon character detection (optional)
- Colla layout via run.py with --grid-layout mode for better rectangular layouts

Key differences from layout_decomposer_dsn_pipeline.py:
- Uses repos/Colla/run.py instead of calling modules directly
- Supports --grid-layout for even rectangular grid partitioning
- Simpler Colla integration via subprocess

Outputs:
    * scenes.json / keyframes.csv / all_probs.csv
    * keyframes/ (exported JPGs)
    * cartoon_detection/ (optional)
    * colla_layout/ (shape decomposition, optimization, final collage)
"""

from __future__ import annotations
import os
import csv
import json
import glob
import argparse
import sys
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Scene detection
from src.scene_detection import (
    create_detector,
    available_detectors,
    Scene,
)

# DSN models
from src.models.dsn import EncoderFC, DSNPolicy
from src.models.dsn_advanced import DSNAdvanced, DSNConfig


# ------------------------------
# Utilities
# ------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timecode_from_frame(i: int, fps: float) -> str:
    """Convert frame index to HH:MM:SS.mmm timecode."""
    if fps <= 0:
        return "00:00:00.000"
    sec = i / fps
    ms = int(round((sec - int(sec)) * 1000))
    m = int(sec // 60)
    s = int(sec % 60)
    h = m // 60
    m = m % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


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
            # Merge into previous
            merged[-1] = Scene(merged[-1].start_frame, sc.end_frame)
    return merged


def detect_scenes_generic(video_path: str, backend: str, min_scene_len: int = 0, **det_kwargs) -> List[Scene]:
    """Wrap scene detector into a simple helper that always returns >=1 scene."""
    det = create_detector(backend, **det_kwargs)
    try:
        scenes = det.detect(video_path)
    finally:
        det.close()

    if not scenes:
        # Fallback: single scene covering the whole video
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        scenes = [Scene(0, max(0, n - 1))]
    
    # Merge short scenes if min_scene_len is specified
    if min_scene_len > 0:
        scenes = normalize_and_merge_scenes(scenes, min_len_frames=min_scene_len)

    return scenes


def grab_frames(
    video_path: str,
    start: int,
    end: int,
    stride: int,
    resize: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Grab frames [start, end] with given stride from the video.
    Optionally resize to (W, H).
    Returns:
      frames: list of BGR uint8 images
      idxs  : list of global frame indices
    """
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    idxs: List[int] = []
    w, h = resize
    for i in range(start, end + 1, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frm = cap.read()
        if not ok:
            continue
        if w > 0 and h > 0:
            frm = cv2.resize(frm, (w, h))
        frames.append(frm)
        idxs.append(i)
    cap.release()
    return frames, idxs


def select_by_budget(
    probs: np.ndarray,
    T: int,
    budget_ratio: float,
    Bmin: int,
    Bmax: int,
) -> List[int]:
    """
    Select indices according to a frame budget:
      B = clip(ceil(budget_ratio * T), [Bmin, Bmax])
    Then pick the top-B frames with highest probability.
    """
    B = int(np.clip(int(np.ceil(budget_ratio * T)), Bmin, Bmax))
    if B <= 0 or T == 0:
        return []
    order = np.argsort(-probs)[:B]  # descending
    return sorted(order.tolist())


def export_keyframe_images(
    video_path: str,
    keyframe_indices: List[int],
    scene_ids: List[int],
    out_dir: str,
    jpeg_quality: int = 95,
) -> None:
    """Export JPG image for each selected keyframe."""
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    for idx, (frame_idx, scene_id) in enumerate(zip(keyframe_indices, scene_ids)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        fn = f"scene_{scene_id:04d}_frame_{frame_idx:08d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fn), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

    cap.release()


# ------------------------------
# Embedding backends
# ------------------------------
def build_embedder(name: str, device: str):
    """
    Returns:
      encode(frames: List[np.ndarray]) -> np.ndarray of shape (T, D)
      D: feature dimension
    All frames are BGR uint8 (OpenCV convention).
    """
    name = name.lower()

    if name == "clip_vitb32":
        import clip
        from PIL import Image

        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()

        def encode(frames: List[np.ndarray]) -> np.ndarray:
            tensors = []
            for f in frames:
                img = Image.fromarray(f[..., ::-1])  # BGR -> RGB
                tensors.append(preprocess(img))
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                feat = model.encode_image(batch)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()

        return encode, 512

    elif name == "resnet50":
        import torchvision.models as tvm
        import torchvision.transforms as T

        model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT).to(device).eval()
        trunk = torch.nn.Sequential(*(list(model.children())[:-1]))

        tfm = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        def encode(frames: List[np.ndarray]) -> np.ndarray:
            tensors = []
            for f in frames:
                rgb = f[..., ::-1].copy()  # BGR -> RGB
                tensors.append(tfm(rgb))
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                feat = trunk(batch).squeeze(-1).squeeze(-1)
            return feat.cpu().numpy()

        return encode, 2048

    elif name == "classic":
        hog = cv2.HOGDescriptor()

        def encode(frames: List[np.ndarray]) -> np.ndarray:
            feats = []
            for f in frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (128, 128))
                h = hog.compute(gray)
                
                # Color histogram
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
                s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
                v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
                
                color_feat = np.concatenate([
                    h_hist.flatten(),
                    s_hist.flatten(),
                    v_hist.flatten()
                ])
                color_feat = color_feat / (color_feat.sum() + 1e-6)
                
                feat = np.concatenate([h.flatten()[:512], color_feat])
                feats.append(feat)
            return np.array(feats, dtype=np.float32)

        return encode, 608  # 512 HOG + 96 color

    else:
        raise ValueError(f"Unsupported embedder '{name}'. Use: clip_vitb32 | resnet50 | classic")


# ------------------------------
# Cartoon Detection Pipeline
# ------------------------------
def run_cartoon_detection_pipeline(keyframes_folder, output_base, device="cuda", config_path="objectfree/detector_config.yaml"):
    """Run cartoon character detection using DetectorCartoon class"""
    import yaml
    import tempfile
    
    from objectfree.detector_cartoon import DetectorCartoon
    
    # Load config and override paths with absolute paths
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
            os.remove(temp_config_path)
    
    # Return results summary
    if isinstance(results, list):
        if not results:
            return {
                "output_dir": output_base,
                "total_images": 0,
                "total_detections": 0,
                "results": results
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


# ------------------------------
# Colla Pipeline via run.py
# ------------------------------
def run_colla_pipeline(
    input_shape: str,
    keyframes_folder: str,
    output_dir: str,
    scaling_factor: int = 2,
    use_object_detection: bool = True,
    use_grid_layout: bool = True,
    debug: bool = True,
    detection_threshold: float = 0.25,
) -> Dict:
    """
    Run Colla pipeline via repos/Colla/run.py subprocess.
    
    Args:
        input_shape: Path to input shape image (e.g., car silhouette)
        keyframes_folder: Path to folder containing keyframe images
        output_dir: Output directory for Colla results
        scaling_factor: Scaling factor for canvas size
        use_object_detection: Use YOLOE object detection mode
        use_grid_layout: Use grid-based layout (rectangular cells)
        debug: Enable debug visualizations
        detection_threshold: Confidence threshold for object detection
    
    Returns:
        Dict with results info
    """
    # Build command
    colla_dir = os.path.join(os.path.dirname(__file__), "repos", "Colla")
    run_script = os.path.join(colla_dir, "run.py")
    
    if not os.path.exists(run_script):
        raise FileNotFoundError(f"Colla run.py not found at: {run_script}")
    
    cmd = [
        sys.executable, run_script,
        os.path.abspath(input_shape),
        os.path.abspath(keyframes_folder),
        os.path.abspath(output_dir),
        str(scaling_factor),
    ]
    
    if use_object_detection:
        cmd.append("--object-detection")
    
    if use_grid_layout:
        cmd.append("--grid-layout")
    
    if debug:
        cmd.append("--debug")
    
    print(f"\n[Colla Pipeline] Running command:")
    print(f"  {' '.join(cmd)}")
    print()
    
    # Run subprocess
    try:
        result = subprocess.run(
            cmd,
            cwd=colla_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"[ERROR] Colla pipeline failed with return code {result.returncode}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            return {"success": False, "error": result.stderr}
        
        # Check output files
        collage_path = os.path.join(output_dir, "collage.png")
        slicing_path = os.path.join(output_dir, "slicing_result.json")
        
        results = {
            "success": True,
            "output_dir": output_dir,
            "collage_path": collage_path if os.path.exists(collage_path) else None,
            "slicing_result": slicing_path if os.path.exists(slicing_path) else None,
        }
        
        # Read slicing result for stats
        if os.path.exists(slicing_path):
            with open(slicing_path, 'r') as f:
                slicing_data = json.load(f)
            results["num_parts"] = len(slicing_data.get("parts", []))
            results["num_images"] = len(slicing_data.get("images", []))
            results["canvas_size"] = (slicing_data.get("width", 0), slicing_data.get("height", 0))
        
        return results
        
    except subprocess.TimeoutExpired:
        print("[ERROR] Colla pipeline timed out after 600 seconds")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"[ERROR] Colla pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ------------------------------
# Argparse
# ------------------------------
def build_argparser() -> argparse.ArgumentParser:
    scene_choices = available_detectors()

    ap = argparse.ArgumentParser(
        description="Layout Decomposer Final Pipeline with DSN-based keyframe extraction and grid layout."
    )
    
    # Input mode: either video or image list
    ap.add_argument("--video", type=str, default=None, help="Input video path.")
    ap.add_argument("--image_list", type=str, default=None, 
                    help="Path to text file containing image paths (one per line). "
                         "Use this instead of --video for processing existing images.")
    ap.add_argument("--image_folder", type=str, default=None,
                    help="Path to folder containing images. Use this instead of --video.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")

    # Scene detection backend + params
    ap.add_argument("--backend", type=str, default="transnetv2", choices=scene_choices,
                    help="Scene detection backend.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="[pyscenedetect] ContentDetector threshold (default 27.0).")
    ap.add_argument("--model_dir", type=str, default='./src/models/TransNetV2',
                    help="[transnetv2] Directory containing weights/, or pass --weights_path.")
    ap.add_argument("--weights_path", type=str, default=None,
                    help="[transnetv2] Direct path to .pth weights (overrides model_dir).")
    ap.add_argument("--prob_threshold", type=float, default=0.5,
                    help="[transnetv2] Boundary probability threshold (default 0.5).")
    ap.add_argument("--scene_device", type=str, default="cuda",
                    help="[transnetv2] Device for model ('cuda'/'cpu').")
    ap.add_argument("--min_scene_len", type=int, default=0,
                    help="Minimum scene length in frames for post-merge (0 = disabled).")

    # DSN / policy
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DSN checkpoint (.pt). If None, use randomly initialized DSN."
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--feat_dim",
        type=int,
        default=-1,
        help="Feature dimension expected by DSN. If <=0, will be inferred from embedder."
    )
    ap.add_argument("--enc_hidden", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=128)

    # Budget
    ap.add_argument("--budget_ratio", type=float, default=0.06)
    ap.add_argument("--Bmin", type=int, default=3)
    ap.add_argument("--Bmax", type=int, default=15)

    # Sampling / resize
    ap.add_argument("--sample_stride", type=int, default=1,
                    help="Sample every N frames within a scene.")
    ap.add_argument("--resize_w", type=int, default=320,
                    help="Resize width for feature extraction.")
    ap.add_argument("--resize_h", type=int, default=180,
                    help="Resize height for feature extraction.")

    # Embedder
    ap.add_argument(
        "--embedder",
        type=str,
        default="clip_vitb32",
        help="Embedding backend: clip_vitb32 | resnet50 | classic",
    )

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
    ap.add_argument("--input_shape_layout", type=str, default="repos/Colla/input_data/image_collections/cars/27.jpg",
                    help="Input shape layout image path (RGB image, mask will be auto-generated).")
    ap.add_argument("--scaling_factor", type=int, default=2,
                    help="Scaling factor for collage canvas (default 2).")
    
    # Grid layout mode (NEW - main feature)
    ap.add_argument("--use_grid_layout", action="store_true", default=True,
                    help="Use grid-based rectangular layout (default: True).")
    ap.add_argument("--no_grid_layout", action="store_false", dest="use_grid_layout",
                    help="Disable grid layout, use traditional medial axis decomposition.")
    
    # Object detection mode
    ap.add_argument("--use_object_detection", action="store_true", default=True,
                    help="Use YOLOE object detection for frame placement (default: True).")
    ap.add_argument("--no_object_detection", action="store_false", dest="use_object_detection",
                    help="Disable object detection, use saliency-based placement.")
    ap.add_argument("--detection_threshold", type=float, default=0.25,
                    help="Detection confidence threshold (default: 0.25).")
    
    # Debug
    ap.add_argument("--debug", action="store_true", default=True,
                    help="Enable debug visualizations (default: True).")
    ap.add_argument("--no_debug", action="store_false", dest="debug",
                    help="Disable debug visualizations.")

    return ap


# ------------------------------
# Helper: Load images from list or folder
# ------------------------------
def load_images_from_source(image_list_path: Optional[str], image_folder: Optional[str]) -> Tuple[List[str], str]:
    """
    Load image paths from either a text file or folder.
    Returns (list of image paths, source name for output folder naming).
    """
    images = []
    source_name = "images"
    
    if image_list_path:
        if not os.path.isfile(image_list_path):
            raise FileNotFoundError(f"Image list file not found: {image_list_path}")
        
        with open(image_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if os.path.isfile(line):
                        images.append(line)
                    else:
                        print(f"  [WARN] Image not found, skipping: {line}")
        
        source_name = Path(image_list_path).stem
        print(f"Loaded {len(images)} images from list file: {image_list_path}")
    
    elif image_folder:
        if not os.path.isdir(image_folder):
            raise NotADirectoryError(f"Image folder not found: {image_folder}")
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif']
        for ext in extensions:
            images.extend(glob.glob(os.path.join(image_folder, f'*{ext}')))
            images.extend(glob.glob(os.path.join(image_folder, f'*{ext.upper()}')))
        
        images = sorted(images)
        source_name = Path(image_folder).name
        print(f"Found {len(images)} images in folder: {image_folder}")
    
    return images, source_name


# ------------------------------
# Main
# ------------------------------
def main():
    args = build_argparser().parse_args()
    
    # Determine input mode: video or images
    is_image_mode = args.image_list is not None or args.image_folder is not None
    
    if not is_image_mode and args.video is None:
        print("Error: Must provide either --video, --image_list, or --image_folder")
        return
    
    # Create unique output directory with timestamp to avoid overwriting
    if is_image_mode:
        _, source_name = load_images_from_source(args.image_list, args.image_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"{args.out_dir}_{source_name}_{timestamp}"
    else:
        video_name = Path(args.video).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"{args.out_dir}_{video_name}_{timestamp}"

    # Prepare output folders
    ensure_dir(args.out_dir)
    key_dir = os.path.join(args.out_dir, "keyframes")
    ensure_dir(key_dir)

    # Resolve device
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("[layout_decomposer_final] CUDA not available, falling back to CPU.")
        dev = "cpu"

    out_dir = Path(args.out_dir)
    
    print("\n" + "="*80)
    print("LAYOUT DECOMPOSER FINAL PIPELINE")
    print("="*80)
    print(f"  Output directory: {args.out_dir}")
    print(f"  Grid layout: {args.use_grid_layout}")
    print(f"  Object detection: {args.use_object_detection}")
    print(f"  Scaling factor: {args.scaling_factor}")
    print("="*80)
    
    # Variables to track
    key_rows = []
    detection_results = None
    
    # =========================================================================
    # IMAGE MODE: Process list of images directly (no video, no scene detection)
    # =========================================================================
    if is_image_mode:
        print("\n" + "="*80)
        print("IMAGE MODE: Processing image list/folder")
        print("="*80)
        
        images, source_name = load_images_from_source(args.image_list, args.image_folder)
        
        if not images:
            print("Error: No valid images found")
            return
        
        # Copy images to keyframes folder
        print(f"\nCopying {len(images)} images to keyframes folder...")
        for i, img_path in enumerate(images):
            dst = os.path.join(key_dir, f"scene_{0:04d}_frame_{i:08d}.jpg")
            shutil.copy(img_path, dst)
        
        # Create dummy scene and keyframe data
        scene_rows = [{"scene_id": 0, "start_frame": 0, "end_frame": len(images)-1, "num_images": len(images)}]
        key_rows = [{"scene_id": 0, "frame_global": i, "prob": 1.0 / (i + 1)} for i in range(len(images))]
        
        print(f"  Copied {len(images)} images to {key_dir}")
        
    # =========================================================================
    # VIDEO MODE: Full pipeline with scene detection and DSN
    # =========================================================================
    else:
        print("\n" + "="*80)
        print("VIDEO MODE: Scene Detection + DSN Keyframe Selection")
        print("="*80)
        
        video_path = args.video
        
        # 1) Scene detection
        print(f"\n[STEP 1] Scene Detection (backend={args.backend})")
        det_kwargs = {}
        if args.backend == "pyscenedetect" and args.threshold is not None:
            det_kwargs["threshold"] = args.threshold
        elif args.backend == "transnetv2":
            det_kwargs["model_dir"] = args.model_dir
            if args.weights_path:
                det_kwargs["weights_path"] = args.weights_path
            det_kwargs["prob_threshold"] = args.prob_threshold
            det_kwargs["device"] = args.scene_device
        
        scenes = detect_scenes_generic(video_path, args.backend, args.min_scene_len, **det_kwargs)
        print(f"  Detected {len(scenes)} scenes")
        
        # Save scenes
        scene_rows = []
        for i, sc in enumerate(scenes):
            scene_rows.append({
                "scene_id": i,
                "start_frame": sc.start_frame,
                "end_frame": sc.end_frame,
                "num_frames": sc.end_frame - sc.start_frame + 1
            })
        
        with open(os.path.join(args.out_dir, "scenes.json"), "w") as f:
            json.dump(scene_rows, f, indent=2)
        
        # 2) Build embedder
        print(f"\n[STEP 2] Building embedder: {args.embedder}")
        encode, emb_dim = build_embedder(args.embedder, device=dev)
        if args.feat_dim <= 0:
            args.feat_dim = emb_dim
        print(f"  Feature dimension: {args.feat_dim}")
        
        # 3) Build DSN model
        print(f"\n[STEP 3] Building DSN model")
        
        # Check checkpoint format first to decide model type
        use_advanced_model = False
        dsn_advanced = None
        enc = None
        pol = None
        
        if args.checkpoint and os.path.isfile(args.checkpoint):
            print(f"  Loading checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
            
            # Check if this is DSNAdvanced format (has 'model', 'config', 'model_type')
            if "model" in ckpt and "config" in ckpt:
                print(f"  Detected DSNAdvanced checkpoint format")
                use_advanced_model = True
                config = ckpt["config"]
                dsn_advanced = DSNAdvanced(config).to(dev)
                dsn_advanced.load_state_dict(ckpt["model"])
                dsn_advanced.eval()
                print(f"  Loaded DSNAdvanced model successfully")
            # Handle old format (encoder/policy or enc/pol)
            elif "encoder" in ckpt:
                enc = EncoderFC(args.feat_dim, args.enc_hidden)
                pol = DSNPolicy(args.enc_hidden, args.lstm_hidden)
                enc.load_state_dict(ckpt["encoder"])
                pol.load_state_dict(ckpt["policy"])
                enc = enc.to(dev).eval()
                pol = pol.to(dev).eval()
            elif "enc" in ckpt:
                enc = EncoderFC(args.feat_dim, args.enc_hidden)
                pol = DSNPolicy(args.enc_hidden, args.lstm_hidden)
                enc.load_state_dict(ckpt["enc"])
                pol.load_state_dict(ckpt["pol"])
                enc = enc.to(dev).eval()
                pol = pol.to(dev).eval()
            else:
                print(f"  [WARN] Unknown checkpoint format, using random initialization")
                enc = EncoderFC(args.feat_dim, args.enc_hidden).to(dev).eval()
                pol = DSNPolicy(args.enc_hidden, args.lstm_hidden).to(dev).eval()
        else:
            print(f"  No checkpoint provided, using random initialization")
            enc = EncoderFC(args.feat_dim, args.enc_hidden).to(dev).eval()
            pol = DSNPolicy(args.enc_hidden, args.lstm_hidden).to(dev).eval()
        
        # 4) Process each scene
        print(f"\n[STEP 4] Processing {len(scenes)} scenes")
        
        resize_tuple = (args.resize_w, args.resize_h)
        all_keyframes = []
        all_scene_ids = []
        
        for scene_idx, sc in enumerate(tqdm(scenes, desc="Processing scenes")):
            start, end = sc.start_frame, sc.end_frame
            
            # Grab frames
            frames, idxs = grab_frames(video_path, start, end, args.sample_stride, resize_tuple)
            if len(frames) == 0:
                continue
            
            T = len(frames)
            
            # Encode
            feats = encode(frames)
            feats_t = torch.from_numpy(feats).float().to(dev)
            
            # Run DSN - support both advanced and basic models
            with torch.no_grad():
                if use_advanced_model and dsn_advanced is not None:
                    # DSNAdvanced model - returns (probs, values) or just probs
                    output = dsn_advanced(feats_t.unsqueeze(0))  # (1, T, D) -> probs
                    if isinstance(output, tuple):
                        probs = output[0].squeeze(0).cpu().numpy()
                    else:
                        probs = output.squeeze(0).cpu().numpy()
                else:
                    # Basic EncoderFC + DSNPolicy
                    h = enc(feats_t)  # (T, enc_hidden)
                    probs, _ = pol(h.unsqueeze(0))  # (1, T)
                    probs = probs.squeeze(0).cpu().numpy()
            
            # Select keyframes
            selected_local = select_by_budget(probs, T, args.budget_ratio, args.Bmin, args.Bmax)
            
            for local_idx in selected_local:
                global_idx = idxs[local_idx]
                all_keyframes.append(global_idx)
                all_scene_ids.append(scene_idx)
                key_rows.append({
                    "scene_id": scene_idx,
                    "frame_global": global_idx,
                    "prob": float(probs[local_idx])
                })
        
        print(f"  Selected {len(all_keyframes)} keyframes total")
        
        # 5) Export keyframes
        print(f"\n[STEP 5] Exporting keyframes to {key_dir}")
        export_keyframe_images(video_path, all_keyframes, all_scene_ids, key_dir, args.key_jpeg_quality)
        
        # Save keyframes CSV
        with open(os.path.join(args.out_dir, "keyframes.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scene_id", "frame_global", "prob"])
            writer.writeheader()
            writer.writerows(key_rows)
        
        # Save scenes CSV
        with open(os.path.join(args.out_dir, "scenes.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scene_id", "start_frame", "end_frame", "num_frames"])
            writer.writeheader()
            writer.writerows(scene_rows)
    
    # =========================================================================
    # 6) Optional: Cartoon character detection
    # =========================================================================
    if args.run_object_free_pipeline:
        print("\n" + "="*80)
        print("STEP 6: CARTOON CHARACTER DETECTION")
        print("="*80)
        
        detection_device_str = args.detection_device or args.device
        
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

    # =========================================================================
    # 7) Colla Layout Pipeline via run.py
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 7: COLLA LAYOUT PIPELINE (via run.py)")
    print("="*80)
    
    colla_output_dir = os.path.join(args.out_dir, "colla_layout")
    ensure_dir(colla_output_dir)
    
    # Verify keyframes exist
    keyframe_files = [f for f in os.listdir(key_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"  Found {len(keyframe_files)} keyframe images in {key_dir}")
    
    if len(keyframe_files) == 0:
        print(f"[ERROR] No keyframe images found!")
        return
    
    # Free memory before Colla
    print("\n[Freeing Memory Before Colla Pipeline]")
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  ✓ Memory cleanup completed")
    except Exception as e:
        print(f"  [WARN] Memory cleanup had issues: {e}")
    
    # Run Colla pipeline
    print(f"\n[Running Colla Pipeline]")
    print(f"  Input shape: {args.input_shape_layout}")
    print(f"  Keyframes: {key_dir}")
    print(f"  Output: {colla_output_dir}")
    print(f"  Grid layout: {args.use_grid_layout}")
    print(f"  Object detection: {args.use_object_detection}")
    
    colla_result = run_colla_pipeline(
        input_shape=args.input_shape_layout,
        keyframes_folder=key_dir,
        output_dir=colla_output_dir,
        scaling_factor=args.scaling_factor,
        use_object_detection=args.use_object_detection,
        use_grid_layout=args.use_grid_layout,
        debug=args.debug,
        detection_threshold=args.detection_threshold,
    )
    
    if colla_result.get("success"):
        print(f"\n[SUCCESS] Colla pipeline completed!")
        print(f"  • Output dir: {colla_result['output_dir']}")
        print(f"  • Collage: {colla_result.get('collage_path', 'N/A')}")
        print(f"  • Parts: {colla_result.get('num_parts', 'N/A')}")
        print(f"  • Canvas size: {colla_result.get('canvas_size', 'N/A')}")
    else:
        print(f"\n[ERROR] Colla pipeline failed: {colla_result.get('error', 'Unknown error')}")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETED!")
    print("="*80)
    print(f"  • Output directory: {args.out_dir}")
    if not is_image_mode:
        print(f"  • Scenes JSON: {os.path.join(args.out_dir, 'scenes.json')}")
        print(f"  • Keyframes CSV: {os.path.join(args.out_dir, 'keyframes.csv')}")
    print(f"  • Keyframe images: {key_dir}")
    if detection_results:
        print(f"  • Cartoon detection: {detection_results['output_dir']}")
    print(f"  • Colla layout: {colla_output_dir}")
    if colla_result.get("collage_path"):
        print(f"  • Final collage: {colla_result['collage_path']}")
    print("="*80)


if __name__ == "__main__":
    main()


"""
Example usage:

# Video mode with grid layout:
python layout_decomposer_final.py \
  --video /path/to/video.mp4 \
  --out_dir outputs/my_layout \
  --checkpoint runs/dsn_advanced_v1/dsn_checkpoint_ep8.pt \
  --device cuda \
  --embedder clip_vitb32 \
  --backend transnetv2 \
  --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
  --input_shape_layout repos/Colla/input_data/image_collections/cars/27.jpg \
  --scaling_factor 2 \
  --use_grid_layout \
  --use_object_detection

# Image folder mode:
python layout_decomposer_final.py \
  --image_folder /path/to/keyframes \
  --out_dir outputs/my_layout \
  --input_shape_layout repos/Colla/input_data/image_collections/cars/27.jpg \
  --scaling_factor 2 \
  --use_grid_layout \
  --use_object_detection

"""
