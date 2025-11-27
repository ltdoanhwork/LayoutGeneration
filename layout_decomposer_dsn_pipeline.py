#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layout Decomposer Pipeline with DSN-based Keyframe Extraction
- Scene detection via pluggable backends (registry): pyscenedetect, transnetv2
- Keyframe selection via DSN (Deep Summarization Network)
- Cartoon character detection (optional)
- Colla layout decomposition and collage assembly
- Outputs:
    * scenes.json / keyframes.csv / all_probs.csv
    * keyframes/ (exported JPGs)
    * cartoon_detection/ (optional)
    * colla_layout/ (shape decomposition, optimization, final collage)
"""

from __future__ import annotations
import os
import csv
import json
import argparse
import sys
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

# Import Colla modules
sys.path.append('objectfree')
from utils.io import *

sys.path.append('repos/Colla')
import repos.Colla.shape_decomposition as sd
import repos.Colla.sas_optimization as so
import repos.Colla.collage_assembly as ca
import repos.Colla.create_masks as cm
from repos.Colla import evaluation
from repos.Colla.utils.get_mask import predict_mask, preprocess_image, refine_mask, net


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
            prev = merged[-1]
            if sc.start_frame <= prev.end_frame + 1:
                # Contiguous → extend previous
                merged[-1] = Scene(prev.start_frame, max(prev.end_frame, sc.end_frame))
            else:
                # Non-contiguous but still merge into previous by extending end
                merged[-1] = Scene(prev.start_frame, sc.end_frame)
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
            break
        if w > 0 and h > 0:
            frm = cv2.resize(frm, (w, h), interpolation=cv2.INTER_AREA)
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
            if len(frames) == 0:
                return np.zeros((0, 512), dtype=np.float32)
            batch = torch.stack(
                [preprocess(Image.fromarray(f[..., ::-1])) for f in frames],  # BGR->RGB
                dim=0,
            ).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch).float()  # (T, 512)
            feats = F.normalize(feats, dim=1)
            return feats.cpu().numpy().astype(np.float32)

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
            if len(frames) == 0:
                return np.zeros((0, 2048), dtype=np.float32)
            batch = torch.stack(
                [tfm(f[..., ::-1].astype(np.float32) / 255.0) for f in frames],  # BGR->RGB
                dim=0,
            ).to(device)
            with torch.no_grad():
                x = trunk(batch).flatten(1)  # (T, 2048)
            x = F.normalize(x, dim=1)
            return x.cpu().numpy().astype(np.float32)

        return encode, 2048

    elif name == "classic":
        hog = cv2.HOGDescriptor()

        def encode(frames: List[np.ndarray]) -> np.ndarray:
            feats = []
            for img in frames:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist(
                    [hsv],
                    [0, 1],
                    None,
                    [32, 32],
                    [0, 180, 0, 256],
                )
                h = cv2.normalize(h, None).flatten()  # 1024

                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, (128, 128))
                hvec = hog.compute(g).reshape(-1)
                pool = np.mean(np.array_split(hvec, 64), axis=1)  # 64

                feat = np.concatenate([h, pool], axis=0).astype(np.float32)  # 1088
                feats.append(feat)

            if not feats:
                return np.zeros((0, 1088), dtype=np.float32)

            X = np.stack(feats, axis=0)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            return X

        return encode, 1088

    else:
        raise ValueError(f"Unsupported embedder '{name}'. Use: clip_vitb32 | resnet50 | classic")


def compute_anime_attrs(frames: List[np.ndarray], device: str = "cuda") -> np.ndarray:
    """
    Compute Anime-CLIP-IQA attributes on-the-fly for a list of frames.
    Returns (T, 6) array of scores.
    """
    import clip
    from PIL import Image
    
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    
    # Define prompt pairs
    prompt_pairs = [
        ("A sharp anime frame.", "A blurry anime frame."),
        ("A colorful anime frame.", "A dull anime frame."),
        ("A bright anime frame.", "A dark anime frame."),
        ("A dynamic sakuga action frame.", "A calm talking anime frame."),
        ("A cinematic impactful anime frame.", "An unremarkable anime frame."),
        ("An anime frame with strong facial expression.", "A neutral anime frame."),
    ]
    
    # Prepare text embeddings
    text_tokens = []
    for p_pos, p_neg in prompt_pairs:
        text_tokens.append(clip.tokenize(p_pos))
        text_tokens.append(clip.tokenize(p_neg))
    
    text_tokens = torch.cat(text_tokens).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        K = len(prompt_pairs)
        D = text_features.shape[-1]
        text_features_pairs = text_features.view(K, 2, D)
    
    # Process frames
    all_scores = []
    for frame in frames:
        img = Image.fromarray(frame[..., ::-1])  # BGR -> RGB
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_features = model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            scores = []
            for k in range(K):
                pair_feats = text_features_pairs[k]
                logits = (100.0 * img_features @ pair_feats.T)
                probs = logits.softmax(dim=-1)
                score_pos = probs[0, 0].item()
                scores.append(score_pos)
            
            all_scores.append(scores)
    
    return np.array(all_scores, dtype=np.float32)


# ------------------------------
# Cartoon Detection Pipeline
# ------------------------------
def run_cartoon_detection_pipeline(keyframes_folder, output_base, device="cuda", config_path="objectfree/detector_config.yaml"):
    """Run cartoon character detection using DetectorCartoon class"""
    
    from objectfree.detector_cartoon import DetectorCartoon
    import yaml
    import tempfile
    
    # Load config and override paths with absolute paths
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['input_path'] = os.path.abspath(keyframes_folder)
    config['type_content'] = 'image'
    config['save_path'] = os.path.abspath(output_base)
    
    # Override model paths to use absolute paths from objectfree/weight_model
    base_weight_dir = os.path.abspath("objectfree/weight_model")
    config['model_path'] = os.path.join(base_weight_dir, "yoloe/weights/best_general.pt")
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


# ------------------------------
# Colla Pipeline Functions
# ------------------------------
def prepare_colla_pipeline():
    """Prepare system resources for Colla pipeline to avoid segmentation faults."""
    print("[prepare_colla_pipeline] Clearing resources...")
    
    # Clear GPU memory
    try:
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


# ------------------------------
# Argparse
# ------------------------------
def build_argparser() -> argparse.ArgumentParser:
    scene_choices = available_detectors()

    ap = argparse.ArgumentParser(
        description="Layout Decomposer Pipeline with DSN-based keyframe extraction."
    )
    ap.add_argument("--video", type=str, required=True, help="Input video path.")
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
    
    # Anime-CLIP-IQA
    ap.add_argument("--use_anime_attrs", type=int, default=0, help="Use Anime-CLIP-IQA attributes (0 or 1)")
    ap.add_argument("--anime_attrs_dim", type=int, default=6, help="Dimension of anime attributes")

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
    ap.add_argument("--input_shape_layout", type=str, default="repos/Colla/input_data/layout/baby.png",
                    help="Input shape layout image path.")
    ap.add_argument("--scaling_factor", type=int, default=1,
                    help="Scaling factor for collage rendering (default 1 to avoid segfault with many images).")

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

    # Resolve device
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("[layout_decomposer_dsn_pipeline] CUDA not available, falling back to CPU.")
        dev = "cpu"

    video_path = args.video
    out_dir = Path(args.out_dir)

    # Read basic video info
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    # 1) Scene detection
    print("\n" + "="*80)
    print("STEP 1: SCENE DETECTION")
    print("="*80)
    
    det_kwargs: Dict[str, Any] = {
        "threshold": args.threshold,
        "model_dir": args.model_dir,
        "weights_path": args.weights_path,
        "prob_threshold": args.prob_threshold,
        "device": args.scene_device,
    }
    det_kwargs = {k: v for k, v in det_kwargs.items() if v not in (None, "", [])}
    
    scenes = detect_scenes_generic(video_path, args.backend, min_scene_len=args.min_scene_len, **det_kwargs)
    print(f"Detected {len(scenes)} scenes")

    # 2) Build embedder
    print("\n" + "="*80)
    print("STEP 2: BUILD EMBEDDER")
    print("="*80)
    
    encode, emb_dim = build_embedder(args.embedder, device=dev)
    
    # If feat_dim not specified, infer from embedder
    if args.feat_dim <= 0:
        args.feat_dim = emb_dim
        print(f"feat_dim not set, using emb_dim={emb_dim} from embedder '{args.embedder}'.")
    
    # 3) Load DSN model
    print("\n" + "="*80)
    print("STEP 3: LOAD DSN MODEL")
    print("="*80)
    
    model_type = "baseline"
    enc, pol, model = None, None, None
    use_anime_attrs_auto = False
    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=dev)
        
        # Check if it's an advanced model checkpoint
        if "model_type" in ckpt and ckpt["model_type"] == "advanced":
            model_type = "advanced"
            print("Detected advanced DSN model")
            config = ckpt["config"]
            
            # Auto-detect if anime_attrs were used in training
            if config.feat_dim > emb_dim:
                use_anime_attrs_auto = True
                print(f"Auto-detected Anime-CLIP-IQA (feat_dim={config.feat_dim} > emb_dim={emb_dim})")
                args.use_anime_attrs = 1
                args.anime_attrs_dim = config.feat_dim - emb_dim
            
            model = DSNAdvanced(config).to(dev).eval()
            model.load_state_dict(ckpt["model"])
            print(f"  Config: {config}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        else:
            # Baseline model
            model_type = "baseline"
            print("Detected baseline DSN model")
            enc = EncoderFC(args.feat_dim, args.enc_hidden).to(dev).eval()
            pol = DSNPolicy(args.enc_hidden, args.lstm_hidden).to(dev).eval()
            enc.load_state_dict(ckpt["encoder"])
            pol.load_state_dict(ckpt["policy"])
    else:
        # No checkpoint: use baseline
        print("No valid checkpoint provided → using randomly initialized baseline DSN (untrained).")
        model_type = "baseline"
        enc = EncoderFC(args.feat_dim, args.enc_hidden).to(dev).eval()
        pol = DSNPolicy(args.enc_hidden, args.lstm_hidden).to(dev).eval()

    # 4) Per-scene DSN inference
    print("\n" + "="*80)
    print("STEP 4: DSN KEYFRAME SELECTION")
    print("="*80)
    
    scene_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    all_prob_rows: List[Dict[str, Any]] = []
    
    # Track all selected keyframe indices and scene IDs for export
    selected_keyframe_indices = []
    selected_scene_ids = []

    resize_tuple = (args.resize_w, args.resize_h)

    for sid, sc in enumerate(tqdm(scenes, desc="Processing scenes")):
        s = int(sc.start_frame)
        e = int(sc.end_frame)
        if e < s:
            continue

        frames, gidx = grab_frames(video_path, s, e, args.sample_stride, resize_tuple)
        if not frames:
            continue

        feats = encode(frames)  # (T, D)
        
        # Compute and concatenate anime_attrs if needed
        if args.use_anime_attrs:
            try:
                anime_attrs = compute_anime_attrs(frames, device=dev)  # (T, K)
                # Align T
                T_min = min(len(feats), len(anime_attrs))
                feats = np.concatenate([feats[:T_min], anime_attrs[:T_min]], axis=1)
            except Exception as e:
                print(f"  [Warning] Failed to compute anime attrs: {e}")
        
        T = feats.shape[0]
        if T == 0:
            continue

        # Convert to torch, run DSN
        x = torch.from_numpy(feats).unsqueeze(0).to(dev)  # (1, T, D)
        with torch.no_grad():
            if model_type == "baseline":
                h = enc(x)                  # (1, T, H)
                probs = pol(h).squeeze(0)   # (T,)
            else:  # advanced
                scene_id = f"scene_{sid}"
                probs = model(x, scene_id=scene_id).squeeze(0)  # (T,)
            probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        probs_np = probs.cpu().numpy().astype(np.float32)

        sel_local = select_by_budget(
            probs_np,
            T=len(frames),
            budget_ratio=args.budget_ratio,
            Bmin=args.Bmin,
            Bmax=args.Bmax,
        )
        
        # Save ALL frame probabilities (for visualization)
        for li in range(len(frames)):
            gi = gidx[li]
            all_prob_rows.append(
                {
                    "scene_id": sid,
                    "frame_global": int(gi),
                    "frame_in_scene": int(li),
                    "time": timecode_from_frame(gi, fps),
                    "prob": float(probs_np[li]),
                    "selected": int(li in sel_local),
                }
            )

        # Save only selected keyframes
        for li in sel_local:
            gi = gidx[li]
            key_rows.append(
                {
                    "scene_id": sid,
                    "frame_global": int(gi),
                    "frame_in_scene": int(li),
                    "time": timecode_from_frame(gi, fps),
                    "prob": float(probs_np[li]),
                }
            )
            selected_keyframe_indices.append(gi)
            selected_scene_ids.append(sid)

        scene_rows.append(
            {
                "scene_id": sid,
                "start_frame": int(s),
                "end_frame": int(e),
                "start_time": timecode_from_frame(int(s), fps),
                "end_time": timecode_from_frame(int(e), fps),
                "duration_frames": int(e - s + 1),
                "duration_seconds": round((e - s + 1) / fps, 3) if fps > 0 else 0.0,
            }
        )

    # 5) Save outputs
    print("\n" + "="*80)
    print("STEP 5: SAVE OUTPUTS")
    print("="*80)
    
    with open(out_dir / "scenes.json", "w", encoding="utf-8") as f:
        json.dump(scene_rows, f, indent=2, ensure_ascii=False)

    with open(out_dir / "keyframes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_id", "frame_global", "frame_in_scene", "time", "prob"],
        )
        writer.writeheader()
        for r in key_rows:
            writer.writerow(r)
    
    # Save ALL frame probabilities for visualization
    with open(out_dir / "all_probs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_id", "frame_global", "frame_in_scene", "time", "prob", "selected"],
        )
        writer.writeheader()
        for r in all_prob_rows:
            writer.writerow(r)

    print(f"Scenes: {len(scene_rows)}, Keyframes: {len(key_rows)}, All frames: {len(all_prob_rows)}")

    # 6) Export keyframe images
    print("\n" + "="*80)
    print("STEP 6: EXPORT KEYFRAME IMAGES")
    print("="*80)
    
    export_keyframe_images(
        video_path=video_path,
        keyframe_indices=selected_keyframe_indices,
        scene_ids=selected_scene_ids,
        out_dir=key_dir,
        jpeg_quality=args.key_jpeg_quality,
    )
    print(f"Exported {len(selected_keyframe_indices)} keyframe images to {key_dir}")

    # 7) Run cartoon character detection pipeline (optional)
    detection_results = None
    if args.run_object_free_pipeline:
        print("\n" + "="*80)
        print("STEP 7: CARTOON CHARACTER DETECTION")
        print("="*80)
        
        # Determine device for detection
        detection_device_str = args.detection_device or args.device or "cuda"
        
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

    # 8) Colla pipeline
    print("\n" + "="*80)
    print("STEP 8: COLLA LAYOUT DECOMPOSER PIPELINE")
    print("="*80)
    
    # Setup paths
    colla_output_dir = os.path.join(args.out_dir, "colla_layout")
    ensure_dir(colla_output_dir)
    
    input_shape = args.input_shape_layout
    # Use exported keyframes as input images
    input_image_collection_folder = key_dir
    # Create masks folder for these keyframes
    input_mask_folder = os.path.join(colla_output_dir, 'keyframe_masks')
    
    print(f"\n[Colla Input Verification]")
    print(f"  input_shape: {input_shape}")
    print(f"  input_mask_folder: {input_mask_folder}")
    print(f"  input_image_collection: {input_image_collection_folder}")
    print(f"  output_dir: {colla_output_dir}")
    print(f"  scaling_factor: {args.scaling_factor}")
    
    # Verify keyframe images exist
    if not os.path.exists(input_image_collection_folder):
        raise FileNotFoundError(f"Keyframes folder not found: {input_image_collection_folder}")
    
    keyframe_files = [f for f in os.listdir(input_image_collection_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"  Found {len(keyframe_files)} keyframe images")
    
    if len(keyframe_files) == 0:
        raise FileNotFoundError(f"No keyframe images in {input_image_collection_folder}")
    
    if len(keyframe_files) > 12:
        print(f"  [WARN] Many images ({len(keyframe_files)}), may cause segfault with Colla pipeline")
        print(f"  [SUGGESTION] Reduce budget_ratio or Bmax to get fewer keyframes")
    
    # CRITICAL: Free all previous models before Colla
    print("\n[Freeing Memory Before Colla Pipeline]")
    try:
        # Delete all heavy objects
        if 'model' in locals() and model is not None:
            del model
        if 'enc' in locals() and enc is not None:
            del enc
        if 'pol' in locals() and pol is not None:
            del pol
        if 'encode' in locals():
            del encode
        
        # Force garbage collection
        import gc
        gc.collect()
        print("  ✓ Freed Python objects")
        
        # Clear CUDA memory completely
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"  ✓ Cleared CUDA cache")
        
        # Wait a bit for system to stabilize
        import time
        time.sleep(2)
        print("  ✓ Memory cleanup completed")
        
    except Exception as e:
        print(f"  [WARN] Memory cleanup had issues: {e}")
    
    # Prepare resources
    prepare_colla_pipeline()
    
    # ============================================
    # STEP 8.1: Generate Mask from Input RGB Image
    # ============================================
    print(f"\n[STEP 8.1] Generating mask from input shape image")
    shape_mask_path = get_mask_from_image(input_shape, colla_output_dir)
    
    # ============================================
    # STEP 8.2: Shape Decomposition
    # ============================================
    print(f"\n[STEP 8.2] Shape decomposition")
    try:
        sd.generate_cuts(shape_mask_path, colla_output_dir)
        print("  ✓ Shape decomposition completed")
        
        # Verify slicing_result.json was created
        slicing_result_path = os.path.join(colla_output_dir, 'slicing_result.json')
        if not os.path.exists(slicing_result_path):
            print(f"  [WARN] slicing_result.json not created at {slicing_result_path}")
        else:
            print(f"  ✓ slicing_result.json created successfully")
    except Exception as e:
        print(f"[ERROR] Shape decomposition failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ============================================
    # STEP 8.3: Create Masks from Keyframe Images
    # ============================================
    print(f"\n[STEP 8.3] Creating masks from keyframe images")
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
    
    # ============================================
    # STEP 8.4: Spatial Assignment Optimization
    # ============================================
    print(f"\n[STEP 8.4] Spatial assignment optimization")
    print(f"  Processing {len(mask_files)} masks")
    
    if len(mask_files) > 12:
        print(f"  [WARN] Many masks, high risk of segfault")
    
    # Validate tree structure
    try:
        slicing_result_path = os.path.join(colla_output_dir, 'slicing_result.json')
        
        if not os.path.exists(slicing_result_path):
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
                raise ValueError(f"Tree height is 0 - shape decomposition failed to create proper tree structure.")
            
            if tree_leaves == 0:
                raise ValueError(f"Tree has no leaves - cannot assign images.")
            
            if tree_leaves > len(mask_files):
                print(f"  [WARN] Tree needs {tree_leaves} images but only {len(mask_files)} available")
            
            if tree_leaves < len(mask_files):
                print(f"  [INFO] Tree has {tree_leaves} leaves but {len(mask_files)} images available")
                print(f"  [INFO] Optimization will select best {tree_leaves} images")
                
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise
    except ValueError as e:
        print(f"[ERROR] Tree validation failed: {e}")
        raise
    except Exception as e:
        print(f"  [WARN] Could not validate tree: {e}")
    
    # Run optimization
    try:
        so.optimization(shape_mask_path, input_mask_folder, colla_output_dir)
        print("  ✓ Optimization completed")
    except Exception as e:
        print(f"[ERROR] Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ============================================
    # STEP 8.5: Collage Assembly & Rendering
    # ============================================
    print(f"\n[STEP 8.5] Collage assembly & rendering")
    
    # Verify slicing result exists
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
    # STEP 8.6: Evaluation (Optional)
    # ============================================
    print(f"\n[STEP 8.6] Evaluating results")
    try:
        metrics = evaluation.evaluate_pipeline_output(colla_output_dir, shape_mask_path)
        print("  Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"    {metric_name}: {metric_value}")
    except Exception as e:
        print(f"  [WARN] Evaluation failed: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"  • Scenes JSON : {os.path.join(args.out_dir, 'scenes.json')}")
    print(f"  • Keyframes CSV: {os.path.join(args.out_dir, 'keyframes.csv')}")
    print(f"  • All Probs CSV: {os.path.join(args.out_dir, 'all_probs.csv')}")
    print(f"  • Keyframe images: {key_dir}")
    if args.run_object_free_pipeline and detection_results:
        print(f"  • Cartoon detection: {detection_results['output_dir']}")
    print(f"  • Colla output: {colla_output_dir}")
    print(f"  • Final collage: {os.path.join(colla_output_dir, 'final_collage.png')}")
    print("="*80)


if __name__ == "__main__":
    main()
"""
python layout_decomposer_dsn_pipeline.py \
  --video data/samples/Sakuga/14652.mp4 \
  --out_dir outputs/dsn_layout_test \
  --checkpoint runs/dsn_advanced_v1/dsn_checkpoint_ep8.pt \
  --device cuda \
  --embedder clip_vitb32 \
  --backend transnetv2 \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --scene_device cuda \
  --min_scene_len 48 \
  --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
  --sample_stride 1 \
  --resize_w 320 --resize_h 180 \
  --input_shape_layout repos/Colla/input_data/layout/baby.png \
  --scaling_factor 1 \
  --run_object_free_pipeline

"""