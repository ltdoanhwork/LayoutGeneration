#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dsn_pipeline.py

Run DSN-based keyframe extraction on a single video:
  1) Scene detection (PySceneDetect or TransNetV2 via create_detector)
  2) For each scene: sample frames (stride, resize) from the raw video
  3) Extract visual embeddings (CLIP / ResNet / classic)
  4) Run DSN (EncoderFC + DSNPolicy) to get per-frame selection probabilities
  5) Apply a budget (ratio + [Bmin, Bmax]) to select top-K frames per scene
  6) Export:
        out_dir / scenes.json
        out_dir / keyframes.csv
     which are compatible with eval_keyframes.py

IMPORTANT:
  - For your current DSN checkpoint trained with prepare_dataset_v2.py (CLIP ViT-B/32),
    you should use: --embedder clip_vitb32 and --feat_dim 512.

Example:
  python -m eval.run_dsn_pipeline \
    --video data/samples/Sakuga/14652.mp4 \
    --out_dir outputs/dsn_infer/14652 \
    --checkpoint /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_advanced_v1_no_motion_100_samples/dsn_checkpoint_ep8.pt \
    --device cuda \
    --feat_dim 512 \
    --enc_hidden 256 \
    --lstm_hidden 128 \
    --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
    --sample_stride 1 \
    --resize_w 320 --resize_h 180 \
    --backend transnetv2 --threshold 27 \
    --embedder clip_vitb32
"""

from __future__ import annotations
import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Import distribution metrics for evaluation
try:
    from src.rl.distribution_metrics import (
        DistributionAwareMetrics,
        compute_distribution_metrics_for_eval,
    )
    HAS_DISTRIBUTION_METRICS = True
except ImportError:
    HAS_DISTRIBUTION_METRICS = False

# RAFT core path (adjust if needed)
RAFT_PATH = Path(__file__).parent.parent / "repos" / "RAFT" / "core"
sys.path.insert(0, str(RAFT_PATH))

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scene_detection import create_detector, available_detectors, Scene
from src.models.dsn import EncoderFC, DSNPolicy
from src.models.dsn_advanced import DSNAdvanced, DSNConfig
from src.models.dsn_v8 import DSNMultiTaskV8, create_dsn_v8

try:
    from raft import RAFT
    from utils.utils import InputPadder
    RAFT_AVAILABLE = True
except ImportError:
    print("[run_dsn_pipeline] RAFT core not found. Motion features will be disabled.")
    RAFT_AVAILABLE = False


# -----------------------------
# RAFT / Motion Utilities
# -----------------------------
class MotionFeatureExtractor(nn.Module):
    """Extract compact motion features from RAFT flow fields. (V8 version)"""
    def __init__(self, motion_dim: int = 128):
        super().__init__()
        self.motion_dim = motion_dim
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, motion_dim)
        
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(flow))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

def load_raft_model(model_path: str, device: torch.device, small: bool = True):
    if not RAFT_AVAILABLE: return None
    class Args:
        def __init__(self):
            self.small = small
            self.mixed_precision = False
            self.alternate_corr = False
            self.dropout = 0
        def __contains__(self, key): return hasattr(self, key)
    
    args = Args()
    model = RAFT(args)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def compute_motion_features(
    raft_model,
    motion_extractor: MotionFeatureExtractor,
    frames: List[np.ndarray],
    device: torch.device
) -> np.ndarray:
    T = len(frames)
    if T < 2: return np.zeros((T, motion_extractor.motion_dim), dtype=np.float32)
    
    motion_feats = []
    with torch.no_grad():
        for t in range(T - 1):
            # img1, img2: (1, 3, H, W)
            img1 = torch.from_numpy(cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0).to(device)
            img2 = torch.from_numpy(cv2.cvtColor(frames[t + 1], cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            padder = InputPadder(img1.shape)
            img1_p, img2_p = padder.pad(img1, img2)
            _, flow_up = raft_model(img1_p, img2_p, iters=20, test_mode=True)
            flow = flow_up[:, :, :img1.shape[2], :img1.shape[3]]
            feat = motion_extractor(flow)
            motion_feats.append(feat.cpu().numpy())
            
        motion_feats.append(motion_feats[-1]) # same for last frame
    return np.concatenate(motion_feats, axis=0)
from src.models.dsn_v8 import DSNMultiTaskV8, create_dsn_v8


# -----------------------------
# Small utilities
# -----------------------------
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
    """Wrap your scene detector into a simple helper that always returns >=1 scene."""
    det = create_detector(backend, **det_kwargs)
    try:
        scenes = det.detect(video_path)  # List[Scene]
    finally:
        det.close()

    if not scenes:
        # Fallback: single scene covering the whole video (in original frame index space)
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


# -----------------------------
# Embedding backends
# -----------------------------
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
        # use everything except the classification head
        trunk = torch.nn.Sequential(*(list(model.children())[:-1]))  # global avg pool output

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
        # Same as in prepare_dataset_v2: HSV hist (32x32=1024) + pooled HOG (64)
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
    
    # Define prompt pairs (same as prepare_anime_attrs.py)
    prompt_pairs = [
        ("A sharp anime frame.", "A blurry anime frame."),
        ("A colorful anime frame.", "A dull anime frame."),
        ("A bright anime frame.", "A dark anime frame."),
        ("A dynamic sakuga action frame.", "A calm talking anime frame."),
        ("A cinematic impactful anime frame.", "An unremarkable anime frame."),
        ("An anime frame with strong facial expression.", "A neutral anime frame."),
    ]
    
    # Prepare text embeddings
    import torch
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
        # frame is BGR uint8
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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run DSN-based keyframe extraction on a single video.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)

    # DSN / policy
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DSN checkpoint (.pt). If None, use randomly initialized DSN."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=-1,
        help="Feature dimension expected by DSN. If <=0, will be inferred from embedder."
    )
    parser.add_argument("--enc_hidden", type=int, default=256)
    parser.add_argument("--lstm_hidden", type=int, default=128)

    # Budget
    parser.add_argument("--budget_ratio", type=float, default=0.05)
    parser.add_argument("--Bmin", type=int, default=2)
    parser.add_argument("--Bmax", type=int, default=5)

    # Sampling / resize
    parser.add_argument("--sample_stride", type=int, default=5)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--resize_h", type=int, default=180)

    # Scene detection
    parser.add_argument(
        "--backend",
        type=str,
        default="transnetv2",
        choices=available_detectors(),
    )
    parser.add_argument("--threshold", type=float, default=None, help="[pyscenedetect] ContentDetector threshold.")
    parser.add_argument("--model_dir", type=str, default='./src/models/TransNetV2', help="[transnetv2] directory with weights/")
    parser.add_argument("--weights_path", type=str, default=None, help="[transnetv2] direct .pth path (override model_dir)")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="[transnetv2] boundary probability threshold.")
    parser.add_argument("--scene_device", type=str, default="cuda", help="[transnetv2] device for model ('cuda'/'cpu').")
    parser.add_argument("--min_scene_len", type=int, default=80, help="Minimum scene length (frames). Shorter scenes are merged into previous one.")

    # Embedder
    parser.add_argument(
        "--embedder",
        type=str,
        default="clip_vitb32",
        help="Embedding backend: clip_vitb32 | resnet50 | classic",
    )
    
    # Anime-CLIP-IQA
    parser.add_argument("--use_anime_attrs", type=int, default=0, help="Use Anime-CLIP-IQA attributes (0 or 1)")
    parser.add_argument("--anime_attrs_dim", type=int, default=6, help="Dimension of anime attributes")

    # RAFT Motion
    parser.add_argument("--use_raft_motion", type=int, default=0, help="Use RAFT motion features (0 or 1)")
    parser.add_argument("--raft_model", type=str, default="repos/RAFT/models/raft-small.pth")
    parser.add_argument("--motion_dim", type=int, default=128)

    args = parser.parse_args()

    # Resolve device
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("[run_dsn_pipeline] CUDA not available, falling back to CPU.")
        dev = "cpu"

    video_path = args.video
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Scene detection
    det_kwargs: Dict[str, Any] = {
        "threshold": args.threshold,
        "model_dir": args.model_dir,
        "weights_path": args.weights_path,
        "prob_threshold": args.prob_threshold,
        "device": args.scene_device,
    }
    scenes = detect_scenes_generic(video_path, args.backend, min_scene_len=args.min_scene_len, **det_kwargs)

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    # 2) Build embedder
    encode, emb_dim = build_embedder(args.embedder, device=dev)
    
    # If feat_dim not specified, infer from embedder
    if args.feat_dim <= 0:
        args.feat_dim = emb_dim
        print(f"[run_dsn_pipeline] feat_dim not set, using emb_dim={emb_dim} from embedder '{args.embedder}'.")
    
    # If we have a checkpoint (trained DSN), emb_dim must match feat_dim.
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        if emb_dim != args.feat_dim:
            raise ValueError(
                f"[run_dsn_pipeline] Feature dim mismatch: embedder gives {emb_dim}, "
                f"but DSN expects feat_dim={args.feat_dim} (from training). "
                "Either retrain DSN with this embedder, or use the same embedder "
                "as training (e.g., clip_vitb32 with feat_dim=512)."
            )
    else:
        # No checkpoint: allow emb_dim != feat_dim and just force feat_dim = emb_dim
        if emb_dim != args.feat_dim:
            print(
                f"[run_dsn_pipeline] No checkpoint provided → "
                f"overriding feat_dim={args.feat_dim} with emb_dim={emb_dim}."
            )
            args.feat_dim = emb_dim

    # 3) Load DSN model
    # Detect model type from checkpoint
    model_type = "baseline"
    enc, pol, model = None, None, None
    use_anime_attrs_auto = False
    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        print(f"[run_dsn_pipeline] Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=dev)
        
        # Check for V7/Unified checkpoint (has model_state_dict and config)
        if "model_state_dict" in ckpt:
            config_dict = ckpt["config"]
            
            # V8/V9 Detection
            valid_versions = ["V8", "V9"]
            if config_dict.get("version") in valid_versions:
                print(f"[run_dsn_pipeline] Detected {config_dict.get('version')} Constrained Multi-Task DSN model")
                # Replicate training script feat_dim computation:
                # feat_dim = base_feat_dim + anime_attrs_dim (if use_anime_attrs) + motion_dim (if use_raft_motion)
                base_feat_dim = config_dict.get("feat_dim", 512)
                c_feat_dim = base_feat_dim
                
                if config_dict.get("use_anime_attrs", 0):
                    anime_dim = config_dict.get("anime_attrs_dim", 6)
                    c_feat_dim += anime_dim
                    args.use_anime_attrs = 1
                    args.anime_attrs_dim = anime_dim
                    print(f"[run_dsn_pipeline] Auto-enabled Anime attrs (dim={anime_dim})")
                
                if config_dict.get("use_raft_motion", 0):
                    motion_dim = config_dict.get("motion_dim", 128)
                    c_feat_dim += motion_dim
                    args.use_raft_motion = 1
                    args.motion_dim = motion_dim
                    print(f"[run_dsn_pipeline] Auto-enabled RAFT Motion (dim={motion_dim})")
                
                print(f"[run_dsn_pipeline] Final feat_dim={c_feat_dim}")

                model = create_dsn_v8(
                    feat_dim=c_feat_dim,
                    hidden_dim=config_dict.get("enc_hidden", args.enc_hidden),
                    lstm_hidden=config_dict.get("lstm_hidden", args.lstm_hidden),
                ).to(dev).eval()
                model.load_state_dict(ckpt["model_state_dict"])
                model_type = "v8"
            else:
                print("[run_dsn_pipeline] Detected V7/Unified DSN model")
                # Replicate training script feat_dim computation (same as V8)
                base_feat_dim = config_dict.get("feat_dim", 512)
                c_feat_dim = base_feat_dim
                
                if config_dict.get("use_anime_attrs", 0):
                    anime_dim = config_dict.get("anime_attrs_dim", 6)
                    c_feat_dim += anime_dim
                    args.use_anime_attrs = 1
                    args.anime_attrs_dim = anime_dim
                    print(f"[run_dsn_pipeline] Auto-enabled Anime attrs (dim={anime_dim})")
                
                if config_dict.get("use_raft_motion", 0):
                    motion_dim = config_dict.get("motion_dim", 128)
                    c_feat_dim += motion_dim
                    args.use_raft_motion = 1
                    args.motion_dim = motion_dim
                    print(f"[run_dsn_pipeline] Auto-enabled RAFT Motion (dim={motion_dim})")
                
                print(f"[run_dsn_pipeline] Final feat_dim={c_feat_dim}")
                
                from src.models.dsn_multitask import create_dsn_multitask
                model = create_dsn_multitask(
                    feat_dim=c_feat_dim,
                    hidden_dim=config_dict.get("enc_hidden", args.enc_hidden),
                    lstm_hidden=config_dict.get("lstm_hidden", args.lstm_hidden),
                ).to(dev).eval()
                model.load_state_dict(ckpt["model_state_dict"])
                model_type = "multitask_v7"

        # Check for V5 multi-task model
        elif "model_type" in ckpt and "multitask" in ckpt["model_type"]:
            model_type = "multitask_v5"
            print("[run_dsn_pipeline] Detected V5 Multi-Task DSN model")
            config = ckpt["config"]
            
            # Auto-detect if anime_attrs were used in training
            if config.feat_dim > emb_dim:
                use_anime_attrs_auto = True
                print(f"[run_dsn_pipeline] Auto-detected Anime-CLIP-IQA (feat_dim={config.feat_dim} > emb_dim={emb_dim})")
                args.use_anime_attrs = 1
                args.anime_attrs_dim = config.feat_dim - emb_dim
            
            from src.models.dsn_multitask import DSNMultiTask
            model = DSNMultiTask(config).to(dev).eval()
            model.load_state_dict(ckpt["model"])
            print(f"  Config: {config}")
            if "merge_weight" in ckpt:
                print(f"  Merge weight (α): {ckpt['merge_weight']:.3f}")
        
        # Check if it's an advanced model checkpoint (v3 or v4)
        elif "model_type" in ckpt and ("advanced" in ckpt["model_type"]):
            model_type = "advanced"
            print("[run_dsn_pipeline] Detected advanced DSN model")
            config = ckpt["config"]
            
            # Auto-detect if anime_attrs were used in training
            if config.feat_dim > emb_dim:
                use_anime_attrs_auto = True
                print(f"[run_dsn_pipeline] Auto-detected Anime-CLIP-IQA (feat_dim={config.feat_dim} > emb_dim={emb_dim})")
                args.use_anime_attrs = 1
                args.anime_attrs_dim = config.feat_dim - emb_dim
            
            model = DSNAdvanced(config).to(dev).eval()
            model.load_state_dict(ckpt["model"])
            print(f"  Config: {config}")
        else:
            # Baseline model
            model_type = "baseline"
            print("[run_dsn_pipeline] Detected baseline DSN model")
            enc = EncoderFC(args.feat_dim, args.enc_hidden).to(dev).eval()
            pol = DSNPolicy(args.enc_hidden, args.lstm_hidden).to(dev).eval()
            enc.load_state_dict(ckpt["encoder"])
            pol.load_state_dict(ckpt["policy"])
    else:
        # No checkpoint: use baseline
        print("[run_dsn_pipeline] No valid checkpoint provided → using randomly initialized baseline DSN (untrained).")
        model_type = "baseline"
        enc = EncoderFC(args.feat_dim, args.enc_hidden).to(dev).eval()
        pol = DSNPolicy(args.enc_hidden, args.lstm_hidden).to(dev).eval()

    # Load RAFT if needed
    raft_model, motion_extractor = None, None
    if args.use_raft_motion:
        print(f"[run_dsn_pipeline] Loading RAFT from {args.raft_model}")
        raft_model = load_raft_model(args.raft_model, dev)
        motion_extractor = MotionFeatureExtractor(args.motion_dim).to(dev).eval()

    # 4) Per-scene inference
    scene_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    all_prob_rows: List[Dict[str, Any]] = []  # Store ALL frame probabilities for visualization
    
    # For distribution metrics
    all_anime_attrs: List[np.ndarray] = []  # Store anime attrs per scene
    all_sel_local_indices: List[List[int]] = []  # Selected indices per scene
    all_gidx: List[List[int]] = []  # Global indices per scene

    resize_tuple = (args.resize_w, args.resize_h)

    for sid, sc in enumerate(scenes):
        # sc is a Scene object (start_frame, end_frame in original index space)
        s = int(sc.start_frame)
        e = int(sc.end_frame)
        if e < s:
            continue

        frames, gidx = grab_frames(video_path, s, e, args.sample_stride, resize_tuple)
        if not frames:
            continue

        feats = encode(frames)  # (T, D)
        
        # Compute and concatenate extra features
        extra_feats = []
        if args.use_anime_attrs:
            try:
                anime_attrs = compute_anime_attrs(frames, device=dev)  # (T, K)
                extra_feats.append(anime_attrs)
            except Exception as e:
                print(f"  [Warning] Failed to compute anime attrs: {e}")
        
        if args.use_raft_motion and raft_model:
            try:
                motion_feats = compute_motion_features(raft_model, motion_extractor, frames, dev)
                extra_feats.append(motion_feats)
            except Exception as e:
                print(f"  [Warning] Failed to compute motion features: {e}")

        if extra_feats:
            feats = np.concatenate([feats] + extra_feats, axis=1)

        T = feats.shape[0]
        if T == 0:
            continue

        # Convert to torch, run DSN
        x = torch.from_numpy(feats).unsqueeze(0).to(dev)  # (1, T, D)
        with torch.no_grad():
            if model_type == "baseline":
                h = enc(x)                  # (1, T, H)
                probs = pol(h).squeeze(0)   # (T,)
            elif model_type in ["advanced", "multitask_v5", "multitask_v7"]:
                # Both advanced and multitask_v5/v7 share same interface
                scene_id = f"scene_{sid}"
                probs = model(x, scene_id=scene_id).squeeze(0)  # (T,)
            elif model_type == "v8":
                # V8 model outputs (probs, values, [alpha])
                out = model(x)
                if isinstance(out, tuple):
                    probs = out[0].squeeze(0)
                else:
                    probs = out.squeeze(0)
            probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        probs_np = probs.cpu().numpy().astype(np.float32)

        sel_local = select_by_budget(
            probs_np,
            T=len(frames),
            budget_ratio=args.budget_ratio,
            Bmin=args.Bmin,
            Bmax=args.Bmax,
        )
        
        # Store for distribution metrics
        if args.use_anime_attrs and 'anime_attrs' in dir():
            all_anime_attrs.append(anime_attrs)
            all_sel_local_indices.append(sel_local)
            all_gidx.append(gidx)
        
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
                    "selected": int(li in sel_local),  # 1 if selected, 0 otherwise
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

    # 6) Compute and save distribution metrics (if anime attrs were computed)
    if HAS_DISTRIBUTION_METRICS and all_anime_attrs:
        # Concatenate all anime attrs and compute combined selection indices
        combined_attrs = np.concatenate(all_anime_attrs, axis=0)
        
        # Map local selection indices to combined indices
        combined_sel_idx = []
        offset = 0
        for scene_attrs, scene_sel in zip(all_anime_attrs, all_sel_local_indices):
            for local_idx in scene_sel:
                if local_idx < len(scene_attrs):
                    combined_sel_idx.append(offset + local_idx)
            offset += len(scene_attrs)
        
        # Compute distribution metrics
        dist_metrics = compute_distribution_metrics_for_eval(combined_attrs, combined_sel_idx)
        
        # Get visualization data
        metrics_computer = DistributionAwareMetrics()
        viz_data = metrics_computer.get_selection_distribution_data(combined_attrs, combined_sel_idx)
        
        # Build distribution data
        video_id = Path(video_path).stem
        distribution_data = {
            "video_id": video_id,
            "video_path": str(video_path),
            "total_sampled_frames": len(combined_attrs),
            "num_selected": len(combined_sel_idx),
            "num_scenes": len(scenes),
            "frame_indices_selected": combined_sel_idx,
            "attrs_all": combined_attrs.tolist(),
            "metrics": dist_metrics,
        }
        
        # Save distribution data
        with open(out_dir / "distribution.json", "w", encoding="utf-8") as f:
            json.dump(distribution_data, f, indent=2)
        
        print(f"  📊 Distribution metrics saved: mean_percentile={dist_metrics.get('mean_percentile_rank', 0):.3f}, "
              f"top10_coverage={dist_metrics.get('top_10_coverage', 0):.1%}")

    print(
        f"[run_dsn_pipeline] Done for {video_path}. "
        f"Scenes={len(scene_rows)}, Keys={len(key_rows)}, All frames={len(all_prob_rows)} -> {out_dir}"
    )


if __name__ == "__main__":
    main()

"""
python -m eval.run_dsn_pipeline \
  --video data/samples/Sakuga/14652.mp4 \
  --out_dir outputs/test_eval_track_a \
  --checkpoint runs/dsn_track_a_features/dsn_checkpoint_ep2.pt \
  --device cuda \
  --embedder clip_vitb32 \
  --backend transnetv2 \
  --model_dir src/models/TransNetV2 \
  --prob_threshold 0.5 \
  --scene_device cuda \
  --min_scene_len 48
"""