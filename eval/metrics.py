# Core evaluation metrics: RecErr, Frechet, coverage, diversity, NR-IQA proxies.

from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, csv, argparse
from src.distance_selector.registry import create_metric
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from scipy.stats import wasserstein_distance


# ---------- Representativeness ----------

def reconstruction_error(
    feats_all: np.ndarray,  # (N, D) normalized
    feats_keys: np.ndarray  # (M, D) normalized
) -> float:
    """
    Mean nearest-neighbor distance of all frames to the keyframe set.
    Using cosine distance on L2-normalized embeddings: d = 1 - cos_sim.
    """
    if feats_keys.shape[0] == 0 or feats_all.shape[0] == 0:
        return float("nan")
    # Cosine similarity matrix: (N,M) = Fa @ Fk^T
    S = feats_all @ feats_keys.T
    # Cosine distance = 1 - max cos_sim to any key
    d = 1.0 - np.max(S, axis=1)
    return float(np.mean(d).astype(np.float32))


def frechet_distance(
    feats_all: np.ndarray, feats_keys: np.ndarray, eps: float = 1e-6
) -> float:
    """
    Frechet-like distance between two Gaussian approximations of features.
    FD = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2(S1 S2)^{1/2})
    Using eigen decomposition for symmetric PSD matrices.
    """
    if feats_all.shape[0] < 2 or feats_keys.shape[0] < 2:
        return float("nan")

    mu1 = np.mean(feats_all, axis=0)
    mu2 = np.mean(feats_keys, axis=0)
    S1 = np.cov(feats_all, rowvar=False) + np.eye(feats_all.shape[1]) * eps
    S2 = np.cov(feats_keys, rowvar=False) + np.eye(feats_keys.shape[1]) * eps

    diff = mu1 - mu2
    diff2 = float(diff.dot(diff))

    # Compute (S1 S2)^{1/2} via symmetric sqrt
    cov_prod = S1.dot(S2)
    # symmetrize
    cov_prod = (cov_prod + cov_prod.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    eigvals[eigvals < 0] = 0.0
    sqrt_cov_prod = eigvecs.dot(np.diag(np.sqrt(eigvals))).dot(eigvecs.T)

    trace = np.trace(S1 + S2 - 2.0 * sqrt_cov_prod)
    return float(diff2 + trace)


# ---------- Coverage ----------

def scene_coverage(
    scenes: List[Tuple[int, int]],
    key_indices: List[int],
) -> float:
    """Fraction of scenes that contain at least one keyframe."""
    if not scenes:
        return float("nan")
    covered = 0
    keys = np.array(sorted(key_indices), dtype=np.int64)
    for (s, e) in scenes:
        # binary search:
        left = np.searchsorted(keys, s, side="left")
        ok = left < len(keys) and keys[left] <= e
        covered += int(ok)
    return covered / len(scenes)


def temporal_coverage(
    feats_all: np.ndarray,
    feats_keys: np.ndarray,
    tau: float = 0.3,
) -> float:
    """
    Fraction of frames whose nearest-key cosine distance <= tau.
    tau ~ [0.2,0.4] depending on backbone.
    """
    if feats_keys.shape[0] == 0 or feats_all.shape[0] == 0:
        return float("nan")
    S = feats_all @ feats_keys.T
    d = 1.0 - np.max(S, axis=1)
    return float(np.mean(d <= tau))


# ---------- Diversity ----------

def redundancy_cosine(
    feats_keys: np.ndarray
) -> Dict[str, float]:
    """
    Redundancy among keyframes: mean pairwise cosine similarity (lower is better),
    plus min pairwise cosine distance.
    """
    M = feats_keys.shape[0]
    if M < 2:
        return {"redundancy_mean_cos": float("nan"), "min_pair_dist": float("nan")}
    # Cosine sim between keys: (M,M)
    S = feats_keys @ feats_keys.T
    iu = np.triu_indices(M, k=1)
    sims = S[iu]
    mean_cos = float(np.mean(sims))
    # Convert to distance:
    dists = 1.0 - sims
    min_dist = float(np.min(dists))
    return {"redundancy_mean_cos": mean_cos, "min_pair_dist": min_dist}


# ---------- Quality (NR-IQA proxies) ----------
"""
✅ 4. Quality (NR-IQA proxies – chất lượng hình ảnh)
Implemented:

sharpness_laplacian() → đo độ sắc nét bằng phương sai Laplacian.

exposure_score() → đo độ phơi sáng, phạt nếu lệch khỏi mức sáng trung bình.

noise_proxy() → đo năng lượng tần số cao (proxy cho nhiễu).

technical_quality_scores() → tổng hợp 3 chỉ số trên bằng median.
"""
def sharpness_laplacian(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def exposure_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mu = float(np.mean(gray))
    # Penalize distance from mid-gray (128)
    return float(255.0 - abs(mu - 128.0) * 2.0)

def noise_proxy(img_bgr: np.ndarray) -> float:
    # High-frequency energy proxy
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    high = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(np.abs(high)))

def technical_quality_scores(
    key_images: List[np.ndarray]
) -> Dict[str, float]:
    """Aggregate simple proxies as a fallback NR-IQA."""
    if not key_images:
        return {"sharpness_med": float("nan"), "exposure_med": float("nan"), "noise_med": float("nan")}
    sharps = [sharpness_laplacian(im) for im in key_images]
    expos  = [exposure_score(im) for im in key_images]
    noise  = [noise_proxy(im) for im in key_images]
    return {
        "sharpness_med": float(np.median(sharps)),
        "exposure_med":  float(np.median(expos)),
        "noise_med":     float(np.median(noise)),
    }

# ---------- Multi-scale SWD (color) ----------

def _pyr(img: np.ndarray, num_scales: int) -> List[np.ndarray]:
    out = [img]
    for _ in range(1, num_scales):
        img = cv2.pyrDown(img) if cv2 is not None else img[::2, ::2]
        out.append(img)
    return out

def _dirs(k: int, seed: int=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(k,3)); v /= (np.linalg.norm(v,axis=1,keepdims=True)+1e-12)
    return v

def _collect(frames: List[np.ndarray], scale: int) -> np.ndarray:
    rng = np.random.default_rng(1234+scale)
    mx = 50_000; buf = []
    for img in frames:
        x = img.astype(np.float32)
        if x.max()>1.0: x/=255.0
        if x.ndim==2: x=np.stack([x,x,x],axis=-1)
        xs = _pyr(x, scale+1)[scale]
        H,W,C = xs.shape; N = H*W
        if N>mx:
            idx=rng.choice(N,size=mx,replace=False)
            buf.append(xs.reshape(-1,C)[idx])
        else:
            buf.append(xs.reshape(-1,C))
    return np.concatenate(buf,axis=0).astype(np.float32) if buf else np.zeros((0,3),np.float32)

def ms_swd_color(frames_all: List[np.ndarray], frames_keys: List[np.ndarray], num_scales=3, num_dirs=16, seed=42) -> float:
    if len(frames_all)<1 or len(frames_keys)<1: return float("nan")
    dirs = _dirs(num_dirs, seed); vals=[]
    for s in range(num_scales):
        A = _collect(frames_all, s); B = _collect(frames_keys, s)
        if A.shape[0]==0 or B.shape[0]==0: continue
        K = min(A.shape[0], B.shape[0], 200_000)
        rng = np.random.default_rng(9876+s)
        A = A[rng.choice(A.shape[0], size=K, replace=False)]
        B = B[rng.choice(B.shape[0], size=K, replace=False)]
        for v in dirs:
            vals.append(wasserstein_distance(A@v, B@v))
    return float(np.mean(vals).astype(np.float32)) if vals else float("nan")




def load_keyframes_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def sample_video_frames(video_path: str, frame_ids: List[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    out = []
    for i in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frm = cap.read()
        if not ok: continue
        out.append(frm)
    cap.release()
    return out

def read_all_frames_sparse(video_path: str, stride: int = 5, resize: tuple[int,int] = (320,180)):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for i in range(0, n, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frm = cap.read()
        if not ok: break
        if resize[0]>0 and resize[1]>0:
            frm = cv2.resize(frm, resize, interpolation=cv2.INTER_AREA)
        frames.append(frm)
    cap.release()
    return frames

def lpips_gap(video_path: str, key_frames: List[int], device="cuda", net="alex") -> float:
    if not key_frames:
        return float("nan")
    metric = create_metric("lpips", net=net, device=device)
    # Grab a light sampling of all frames and the selected keyframes
    all_frames = read_all_frames_sparse(video_path, stride=5)
    if not all_frames:
        return float("nan")
    keys = sample_video_frames(video_path, key_frames)
    if not keys:
        return float("nan")
    # Preprocess to tensors once
    Ts_all = [metric.preprocess_bgr(f) for f in all_frames]
    Ts_keys = [metric.preprocess_bgr(f) for f in keys]
    vals = []
    import torch
    with torch.no_grad():
        for Ta in Ts_all:
            m = +1e9
            for Tk in Ts_keys:
                d = metric.pair_distance(Ta, Tk)
                if d < m: m = d
            vals.append(m)
    return float(np.mean(vals)) if vals else float("nan")

def lpips_diversity(video_path: str, key_frames: List[int], device="cuda", net="alex") -> float:
    if len(key_frames) < 2:
        return 0.0
    metric = create_metric("lpips", net=net, device=device)
    imgs = sample_video_frames(video_path, key_frames)
    if len(imgs) < 2:
        return 0.0
    Ts = [metric.preprocess_bgr(f) for f in imgs]
    vals = []
    import torch
    with torch.no_grad():
        for i in range(len(Ts)):
            for j in range(i+1, len(Ts)):
                vals.append(metric.pair_distance(Ts[i], Ts[j]))
    return float(np.mean(vals)) if vals else 0.0

def ms_swd_color_gap(video_path: str, key_frames: List[int]) -> float:
    all_frames = read_all_frames_sparse(video_path, stride=5)
    keys = sample_video_frames(video_path, key_frames)
    if not all_frames or not keys:
        return float("nan")
    return ms_swd_color(all_frames, keys, num_scales=3, num_dirs=16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--keyframes_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--lpips_device", type=str, default="cuda")
    ap.add_argument("--lpips_net", type=str, default="alex")
    args = ap.parse_args()

    rows = load_keyframes_csv(args.keyframes_csv)
    key_ids = sorted({int(r["frame_global"]) for r in rows})

    metrics = {}
    try:
        metrics["LPIPS_PerceptualGap"] = lpips_gap(args.video, key_ids, device=args.lpips_device, net=args.lpips_net)
    except Exception:
        metrics["LPIPS_PerceptualGap"] = float("nan")
    try:
        metrics["LPIPS_DiversitySel"] = lpips_diversity(args.video, key_ids, device=args.lpips_device, net=args.lpips_net)
    except Exception:
        metrics["LPIPS_DiversitySel"] = float("nan")
    try:
        metrics["MS_SWD_Color"] = ms_swd_color_gap(args.video, key_ids)
    except Exception:
        metrics["MS_SWD_Color"] = float("nan")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[extra_metrics] Saved -> {args.out_json}")

if __name__ == "__main__":
    main()
