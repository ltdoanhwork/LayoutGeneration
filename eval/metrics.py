# Core evaluation metrics: RecErr, Frechet, coverage, diversity, NR-IQA proxies.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2


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
