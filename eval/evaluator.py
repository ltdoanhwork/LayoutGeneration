# Orchestrates the evaluation over one or multiple keyframe sets.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import csv
import os
import numpy as np
import cv2

from .feature_extractor import FeatureExtractor, FrameSample
from .metrics import (
    reconstruction_error, frechet_distance, scene_coverage, temporal_coverage,
    redundancy_cosine, technical_quality_scores
)
from .baselines import baseline_uniform, baseline_middle_of_scene, baseline_motion_peaks


@dataclass
class EvalConfig:
    backbone: str = "resnet50"
    device: Optional[str] = None
    input_size: Tuple[int, int] = (224, 224)
    sample_stride: int = 10
    max_frames_eval: int = 200   # cap number of frames to embed for "all frames" set
    tau_temporal: float = 0.3    # threshold for temporal coverage


def load_scenes_json(path: str) -> List[Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect a list of dict with start_frame/end_frame
    scenes = [(int(d["start_frame"]), int(d["end_frame"])) for d in data]
    # Sort by start_frame just in case
    scenes.sort(key=lambda x: x[0])
    return scenes


def load_keyframes_csv(path: str) -> List[int]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "frame_idx" not in reader.fieldnames:
            raise ValueError("keyframes.csv must contain a 'frame_idx' column.")
        for row in reader:
            out.append(int(row["frame_idx"]))
    return sorted(out)


def embed_video_frames(
    video_path: str,
    extractor: FeatureExtractor,
    stride: int,
    max_frames: int,
) -> Tuple[np.ndarray, List[int]]:
    """Return (features_all, indices_all)."""
    # Sample by stride and cap
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    idx = list(range(0, total, max(1, stride)))
    if max_frames > 0 and len(idx) > max_frames:
        sel = np.linspace(0, len(idx)-1, max_frames, dtype=int)
        idx = [idx[i] for i in sel]

    # Load frames
    samples = [FrameSample(i) for i in idx]
    frames = extractor.load_frames(video_path, samples)
    Fe = extractor.embed_images(frames, batch_size=32)
    return Fe, idx


def eval_one_set(
    video_path: str,
    scenes: List[Tuple[int, int]],
    key_indices: List[int],
    cfg: EvalConfig,
    precomputed_all_feats: Optional[np.ndarray] = None,
    precomputed_all_idx: Optional[List[int]] = None,
) -> Dict[str, float]:
    # Feature extractor
    extractor = FeatureExtractor(backbone=cfg.backbone, device=cfg.device, input_size=cfg.input_size)

    # Embed "all frames" (sampled) if not precomputed
    if precomputed_all_feats is None or precomputed_all_idx is None:
        feats_all, idx_all = embed_video_frames(
            video_path, extractor, stride=cfg.sample_stride, max_frames=cfg.max_frames_eval
        )
    else:
        feats_all, idx_all = precomputed_all_feats, precomputed_all_idx
    feats_all = feats_all[~np.isnan(feats_all).any(axis=1)]
    


    # Intersect provided key_indices with idx_all space for fair comparison in embedding space
    keys_in_all = sorted(set(key_indices).intersection(set(idx_all)))
    # If none intersect, map each key to the nearest index in idx_all
    if len(keys_in_all) == 0 and len(key_indices) > 0:
        keys_in_all = []
        for k in key_indices:
            nearest = int(np.argmin([abs(k - j) for j in idx_all]))
            keys_in_all.append(idx_all[nearest])
        keys_in_all = sorted(set(keys_in_all))

    # Extract keyframe embeddings
    key_samples = [FrameSample(i) for i in keys_in_all]
    key_frames = extractor.load_frames(video_path, key_samples)
    feats_keys = extractor.embed_images(key_frames, batch_size=32)
    feats_keys = feats_keys[~np.isnan(feats_keys).any(axis=1)]

    # Metrics
    rec = reconstruction_error(feats_all, feats_keys)
    fd  = frechet_distance(feats_all, feats_keys)
    cov = scene_coverage(scenes, key_indices)
    tcv = temporal_coverage(feats_all, feats_keys, tau=cfg.tau_temporal)
    div = redundancy_cosine(feats_keys)

    # Technical quality (simple proxies)
    tq  = technical_quality_scores(key_frames)

    result = {
        "RecErr": rec,
        "Frechet": fd,
        "SceneCoverage": cov,
        "TemporalCoverage@tau": tcv,
        "RedundancyMeanCos": div["redundancy_mean_cos"],
        "MinPairwiseDist": div["min_pair_dist"],
        "Sharpness_med": tq["sharpness_med"],
        "Exposure_med":  tq["exposure_med"],
        "Noise_med":     tq["noise_med"],
        "NumKeys": float(len(key_indices)),
        "NumAllEmbed": float(feats_all.shape[0]),
    }
    return result


def eval_with_baselines(
    video_path: str,
    scenes: List[Tuple[int, int]],
    key_indices_method: List[int],
    cfg: EvalConfig,
    total_frames: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate your method and standard baselines with shared 'all-frames' embeddings."""
    # Prepare extractor + shared embeddings to be reused
    extractor = FeatureExtractor(backbone=cfg.backbone, device=cfg.device, input_size=cfg.input_size)
    feats_all, idx_all = embed_video_frames(video_path, extractor, cfg.sample_stride, cfg.max_frames_eval)

    # Determine m
    m = len(key_indices_method)
    if total_frames is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

    # Baselines
    uni = baseline_uniform(total_frames, m)
    mid = baseline_middle_of_scene(scenes, m)
    mot = baseline_motion_peaks(video_path, m, stride=max(1, cfg.sample_stride//2))

    # Evaluate
    results: Dict[str, Dict[str, float]] = {}
    results["method"] = eval_one_set(video_path, scenes, key_indices_method, cfg, feats_all, idx_all)
    results["uniform"] = eval_one_set(video_path, scenes, uni, cfg, feats_all, idx_all)
    results["middle_of_scene"] = eval_one_set(video_path, scenes, mid, cfg, feats_all, idx_all)
    results["motion_peak"] = eval_one_set(video_path, scenes, mot, cfg, feats_all, idx_all)
    return results
