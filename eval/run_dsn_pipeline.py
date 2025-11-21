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
    --checkpoint /home/serverai/ltdoanh/LayoutGeneration/runs/dsn_advanced_v1_no_motion/dsn_checkpoint_ep15.pt \
    --device cuda \
    --feat_dim 512 \
    --enc_hidden 256 \
    --lstm_hidden 128 \
    --budget_ratio 0.06 --Bmin 3 --Bmax 15 \
    --sample_stride 5 \
    --resize_w 320 --resize_h 180 \
    --backend transnetv2 --threshold 27 \
    --embedder clip_vitb32
"""

from __future__ import annotations
import os
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from src.scene_detection import create_detector, available_detectors, Scene
from src.models.dsn import EncoderFC, DSNPolicy
from src.models.dsn_advanced import DSNAdvanced, DSNConfig


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


def detect_scenes_generic(video_path: str, backend: str, **det_kwargs) -> List[Scene]:
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
    parser.add_argument("--budget_ratio", type=float, default=0.06)
    parser.add_argument("--Bmin", type=int, default=3)
    parser.add_argument("--Bmax", type=int, default=15)

    # Sampling / resize
    parser.add_argument("--sample_stride", type=int, default=5)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--resize_h", type=int, default=180)

    # Scene detection
    parser.add_argument(
        "--backend",
        type=str,
        default="pyscenedetect",
        choices=available_detectors(),
    )
    parser.add_argument("--threshold", type=float, default=None, help="[pyscenedetect] ContentDetector threshold.")
    parser.add_argument("--model_dir", type=str, default='./src/models/TransNetV2', help="[transnetv2] directory with weights/")
    parser.add_argument("--weights_path", type=str, default=None, help="[transnetv2] direct .pth path (override model_dir)")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="[transnetv2] boundary probability threshold.")
    parser.add_argument("--scene_device", type=str, default="cuda", help="[transnetv2] device for model ('cuda'/'cpu').")

    # Embedder
    parser.add_argument(
        "--embedder",
        type=str,
        default="clip_vitb32",
        help="Embedding backend: clip_vitb32 | resnet50 | classic",
    )

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
    scenes = detect_scenes_generic(video_path, args.backend, **det_kwargs)

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
    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        print(f"[run_dsn_pipeline] Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=dev)
        
        # Check if it's an advanced model checkpoint
        if "model_type" in ckpt and ckpt["model_type"] == "advanced":
            model_type = "advanced"
            print("[run_dsn_pipeline] Detected advanced DSN model")
            config = ckpt["config"]
            model = DSNAdvanced(config).to(dev).eval()
            model.load_state_dict(ckpt["model"])
            print(f"  Config: {config}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
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

    # 4) Per-scene inference
    scene_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    all_prob_rows: List[Dict[str, Any]] = []  # Store ALL frame probabilities for visualization

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

    print(
        f"[run_dsn_pipeline] Done for {video_path}. "
        f"Scenes={len(scene_rows)}, Keys={len(key_rows)}, All frames={len(all_prob_rows)} -> {out_dir}"
    )


if __name__ == "__main__":
    main()
