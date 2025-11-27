#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, csv, argparse
from typing import List, Dict, Any
import numpy as np
import cv2

from src.distance_selector.registry import create_metric
from eval.metrics import ms_swd_color

def load_keyframes_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def sample_video_frames(video_path: str, frame_ids: List[int], resize: tuple[int,int] = (320,180)) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    out = []
    for i in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frm = cap.read()
        if not ok: continue
        if resize[0] > 0 and resize[1] > 0:
            frm = cv2.resize(frm, resize, interpolation=cv2.INTER_AREA)
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

def compute_anime_attr_stats(video_path: str, key_frames: List[int], device="cuda") -> Dict[str, float]:
    """
    Compute mean anime attribute scores for selected keyframes.
    Returns dict with keys like 'Anime_Sharpness_Mean', 'Anime_Sakuga_Mean', etc.
    """
    try:
        import clip
        import torch
        from PIL import Image
        
        # Load frames
        frames = sample_video_frames(video_path, key_frames, resize=(320, 180))
        if not frames:
            return {}
        
        # Load CLIP model
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
        attr_names = ["Sharpness", "Colorfulness", "Brightness", "Sakuga", "Cinematic", "Expression"]
        
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
        
        # Compute mean for each attribute
        all_scores = np.array(all_scores)  # (N, K)
        result = {}
        for i, name in enumerate(attr_names):
            result[f"Anime_{name}_Mean"] = float(all_scores[:, i].mean())
        
        return result
        
    except Exception as e:
        print(f"[extra_metrics] Anime attribute computation failed: {e}")
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--keyframes_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--lpips_device", type=str, default="cuda")
    ap.add_argument("--lpips_net", type=str, default="alex")
    ap.add_argument("--compute_anime_attrs", type=int, default=1, help="Compute anime attribute stats (0 or 1)")
    args = ap.parse_args()

    rows = load_keyframes_csv(args.keyframes_csv)
    key_ids = sorted({int(r["frame_global"]) for r in rows})

    metrics = {}
    
    # Create LPIPS metric once and reuse
    try:
        print(f"[extra_metrics] Computing LPIPS metrics on {args.lpips_device}...")
        metric = create_metric("lpips", net=args.lpips_net, device=args.lpips_device)
        
        # Get all frames and keyframes with consistent resize
        all_frames = read_all_frames_sparse(args.video, stride=5)
        keys = sample_video_frames(args.video, key_ids)
        
        if all_frames and keys:
            # Preprocess once
            import torch
            Ts_all = [metric.preprocess_bgr(f) for f in all_frames]
            Ts_keys = [metric.preprocess_bgr(f) for f in keys]
            
            # Compute LPIPS Gap
            vals_gap = []
            with torch.no_grad():
                for Ta in Ts_all:
                    m = +1e9
                    for Tk in Ts_keys:
                        d = metric.pair_distance(Ta, Tk)
                        if d < m: m = d
                    vals_gap.append(m)
            metrics["LPIPS_PerceptualGap"] = float(np.mean(vals_gap)) if vals_gap else float("nan")
            
            # Compute LPIPS Diversity
            if len(Ts_keys) >= 2:
                vals_div = []
                with torch.no_grad():
                    for i in range(len(Ts_keys)):
                        for j in range(i+1, len(Ts_keys)):
                            vals_div.append(metric.pair_distance(Ts_keys[i], Ts_keys[j]))
                metrics["LPIPS_DiversitySel"] = float(np.mean(vals_div)) if vals_div else 0.0
            else:
                metrics["LPIPS_DiversitySel"] = 0.0
        else:
            metrics["LPIPS_PerceptualGap"] = float("nan")
            metrics["LPIPS_DiversitySel"] = float("nan")
            
    except Exception as e:
        print(f"[extra_metrics] LPIPS computation failed: {e}")
        metrics["LPIPS_PerceptualGap"] = float("nan")
        metrics["LPIPS_DiversitySel"] = float("nan")
    
    # Compute MS-SWD
    try:
        print(f"[extra_metrics] Computing MS-SWD Color...")
        metrics["MS_SWD_Color"] = ms_swd_color_gap(args.video, key_ids)
    except Exception as e:
        print(f"[extra_metrics] MS-SWD computation failed: {e}")
        metrics["MS_SWD_Color"] = float("nan")
    
    # Compute Anime Attributes
    if args.compute_anime_attrs:
        try:
            print(f"[extra_metrics] Computing Anime-CLIP-IQA attributes...")
            anime_stats = compute_anime_attr_stats(args.video, key_ids, device=args.lpips_device)
            metrics.update(anime_stats)
        except Exception as e:
            print(f"[extra_metrics] Anime attribute computation failed: {e}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[extra_metrics] Saved -> {args.out_json}")

if __name__ == "__main__":
    main()
