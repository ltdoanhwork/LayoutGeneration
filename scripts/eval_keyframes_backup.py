#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# All comments are in English.

from __future__ import annotations
import os
import json
import argparse
import csv

import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.evaluator import (
    EvalConfig, load_scenes_json, load_keyframes_csv,
    eval_one_set, eval_with_baselines
)


def save_dict_as_csv(d: dict, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in d.items():
            w.writerow([k, v])


def main():
    ap = argparse.ArgumentParser(description="Evaluate keyframes without GT using multi-metric protocol.")
    ap.add_argument("--video", required=True, type=str)
    ap.add_argument("--scenes_json", required=True, type=str, help="scenes.json from pipeline")
    ap.add_argument("--keyframes_csv", required=True, type=str, help="keyframes.csv (must include frame_idx column)")
    ap.add_argument("--out_dir", required=True, type=str)

    # Feature backbone / sampling
    ap.add_argument("--backbone", type=str, default="resnet50", help="resnet50 | vit_b_16 (torchvision)")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    ap.add_argument("--input_w", type=int, default=224)
    ap.add_argument("--input_h", type=int, default=224)
    ap.add_argument("--sample_stride", type=int, default=10)
    ap.add_argument("--max_frames_eval", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.3, help="Threshold for TemporalCoverage@tau")

    # Run baselines toggle
    ap.add_argument("--with_baselines", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scenes = load_scenes_json(args.scenes_json)
    keys  = load_keyframes_csv(args.keyframes_csv)

    cfg = EvalConfig(
        backbone=args.backbone,
        device=args.device,
        input_size=(args.input_w, args.input_h),
        sample_stride=args.sample_stride,
        max_frames_eval=args.max_frames_eval,
        tau_temporal=args.tau,
    )

    if args.with_baselines:
        res = eval_with_baselines(args.video, scenes, keys, cfg)
        # Save JSON + per-method CSV
        out_json = os.path.join(args.out_dir, "eval_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        for name, d in res.items():
            save_dict_as_csv(d, os.path.join(args.out_dir, f"eval_{name}.csv"))
        print(f"[DONE] Saved results with baselines to: {args.out_dir}")
    else:
        res = eval_one_set(args.video, scenes, keys, cfg)
        out_json = os.path.join(args.out_dir, "eval_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        save_dict_as_csv(res, os.path.join(args.out_dir, "eval_results.csv"))
        print(f"[DONE] Saved results to: {args.out_dir}")


if __name__ == "__main__":
    main()

"""
# Evaluate your keyframes only
python eval_keyframes.py \
  --video samples/Sakuga/10736.mp4 \
  --scenes_json outputs/run_psd_lpips/scenes.json \
  --keyframes_csv outputs/run_psd_lpips/keyframes.csv \
  --out_dir outputs/eval_psd_lpips \
  --backbone resnet50 \
  --sample_stride 10 --max_frames_eval 200 --tau 0.3

# Evaluate + compare with baselines (uniform / middle-of-scene / motion-peak)
python eval_keyframes.py \
  --video samples/Sakuga/10736.mp4 \
  --scenes_json outputs/run_psd_lpips/scenes.json \
  --keyframes_csv outputs/run_psd_lpips/keyframes.csv \
  --out_dir outputs/eval_psd_lpips \
  --with_baselines



"""