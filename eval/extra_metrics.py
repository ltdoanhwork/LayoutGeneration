#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, csv, argparse
from typing import List, Dict, Any
import numpy as np
import cv2

from src.distance_selector.registry import create_metric
from src.metrics.ms_swd import ms_swd_color

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
