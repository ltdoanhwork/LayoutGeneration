#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for keyframe pipeline + evaluation + visualization.
- Backends: dists / lpips
- Save everything under outputs/
    outputs/pipeline/<backend>/run_tv2_<backend>_<video>/
    outputs/eval/<backend>/eval_<video>/
"""

import os
import glob
import subprocess
from pathlib import Path
import argparse
from typing import List, Tuple
from glob import glob as gglob


# ---------------------------- I/O utils ----------------------------

def find_mp4_videos(data_folder: str, pattern: str | None = None) -> List[str]:
    """Find all MP4 files (case-insensitive) in data_folder (recursive)."""
    if pattern:
        cand = glob.glob(os.path.join(data_folder, pattern), recursive=True)
        return sorted({p for p in cand if p.lower().endswith(".mp4")})

    patterns = ['*.mp4', '*.MP4', '*.Mp4', '*.mP4']
    files = set()
    for pat in patterns:
        files.update(glob.glob(os.path.join(data_folder, pat)))
        files.update(glob.glob(os.path.join(data_folder, '**', pat), recursive=True))
    return sorted(files)


def _run(cmd: List[str], env=None) -> Tuple[bool, str, str]:
    try:
        p = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        return True, p.stdout, p.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout or "", e.stderr or f"Return code: {e.returncode}"


# ---------------------------- Path helpers ----------------------------

def pipeline_base_dir(output_base: str, backend: str) -> str:
    """
    Base dir passed to pipeline.py (pipeline.py sẽ tự nối _<video> phía sau).
    Example: outputs/pipeline/lpips/run_tv2_lpips
    """
    d = os.path.join(output_base, "pipeline", backend, f"run_tv2_{backend}")
    os.makedirs(d, exist_ok=True)
    return d

def eval_dir_for(video_path: str, output_base: str, backend: str) -> str:
    """outputs/eval/<backend>/eval_<video>/"""
    name = Path(video_path).stem
    d = os.path.join(output_base, "eval", backend, f"eval_{name}")
    os.makedirs(d, exist_ok=True)
    return d

def resolve_pipeline_dir(video_name: str, backend: str, output_base: str) -> str:
    """
    Tìm thư mục artifacts có scenes.json & keyframes.csv.
    Hỗ trợ các kiểu:
      - .../run_tv2_<backend>_<video>
      - .../run_tv2_<backend>_<video>_<video> (case cũ bị nhân đôi)
      - .../run_tv2_<backend>/<video> (một số phiên bản khác)
    """
    base = os.path.join(output_base, "pipeline", backend)
    candidates = [
        os.path.join(base, f"run_tv2_{backend}_{video_name}"),
        os.path.join(base, f"run_tv2_{backend}_{video_name}_{video_name}"),
        os.path.join(base, f"run_tv2_{backend}", video_name),
        os.path.join(base, video_name),
    ]
    # thêm glob fallback
    candidates += sorted(gglob(os.path.join(base, f"*{video_name}*")))

    checked = []
    for c in candidates:
        scenes = os.path.join(c, "scenes.json")
        keys   = os.path.join(c, "keyframes.csv")
        checked.append(c)
        if os.path.isfile(scenes) and os.path.isfile(keys):
            return c

    existing = sorted(gglob(os.path.join(base, "run_tv2_*")))
    raise FileNotFoundError(
        f"[resolve_pipeline_dir] Not found artifacts for '{video_name}' (backend={backend}).\n"
        f"Tried:\n  - " + "\n  - ".join(checked) + "\n"
        f"Existing under '{base}':\n  - " + ("\n  - ".join(existing) if existing else "(none)")
    )


# ---------------------------- Pipeline step ----------------------------

def run_pipeline_for_video(
    video_path: str,
    out_base: str,
    backend: str,
    model_dir: str,
    prob_threshold: float,
    distance_backend: str,
    dists_as_distance: bool,
    sample_stride: int,
    max_frames_per_scene: int,
    keyframes_per_scene: int,
    nms_radius: int,
    resize_w: int,
    resize_h: int,
) -> Tuple[bool, str]:
    """
    Run pipeline.py cho 1 video.
    Trả về (success, pipeline_dir_đã_resolve_chứa artifacts).
    """
    # IMPORTANT: đưa base (không kèm tên video) cho pipeline.py
    out_dir_base = pipeline_base_dir(out_base, distance_backend)
    video_name = Path(video_path).stem

    cmd = [
        "python", "pipeline.py",
        "--video", video_path,
        "--backend", backend,
        "--model_dir", model_dir,
        "--prob_threshold", str(prob_threshold),
        "--distance_backend", distance_backend,
        "--dists_as_distance", "1" if dists_as_distance else "0",
        "--sample_stride", str(sample_stride),
        "--max_frames_per_scene", str(max_frames_per_scene),
        "--keyframes_per_scene", str(keyframes_per_scene),
        "--nms_radius", str(nms_radius),
        "--resize_w", str(resize_w),
        "--resize_h", str(resize_h),
        "--out_dir", out_dir_base,
    ]

    print(f"\n{'='*70}\nProcessing: {video_path}\nPipeline base: {out_dir_base}\n{'='*70}")
    ok, out, err = _run(cmd)

    # Resolve thư mục thực tế sau khi pipeline chạy
    try:
        resolved_dir = resolve_pipeline_dir(video_name, distance_backend, out_base)
    except FileNotFoundError as e:
        resolved_dir = os.path.join(out_dir_base, f"_{video_name}")  # fallback hiển thị
        print(str(e))

    if ok:
        print(f"✅ PIPELINE OK: {video_name}")
        print(f"Artifacts: {resolved_dir}")
        if out.strip():
            print(out)
    else:
        print(f"❌ PIPELINE FAIL: {video_name}")
        if err.strip():
            print(err)

    return ok, resolved_dir


# ---------------------------- Eval + Viz step ----------------------------

def run_eval_and_visualize_for_video(
    video_path: str,
    pipeline_dir: str | None,
    output_base: str,
    distance_backend: str,
    eval_script: str,
    viz_module: str,
    eval_backbone: str,
    eval_sample_stride: int,
    eval_max_frames: int,
    eval_tau: float,
):
    """Run eval và visualize; tự resolve pipeline_dir nếu cần."""
    video_name = Path(video_path).stem

    if pipeline_dir is None or \
       not (os.path.isfile(os.path.join(pipeline_dir, "scenes.json")) and
            os.path.isfile(os.path.join(pipeline_dir, "keyframes.csv"))):
        pipeline_dir = resolve_pipeline_dir(video_name, distance_backend, output_base)

    scenes_json = os.path.join(pipeline_dir, "scenes.json")
    keyframes_csv = os.path.join(pipeline_dir, "keyframes.csv")

    out_dir = eval_dir_for(video_path, output_base, distance_backend)

    # Eval
    eval_cmd = [
        "python", eval_script,
        "--video", video_path,
        "--scenes_json", scenes_json,
        "--keyframes_csv", keyframes_csv,
        "--out_dir", out_dir,
        "--backbone", eval_backbone,
        "--sample_stride", str(eval_sample_stride),
        "--max_frames_eval", str(eval_max_frames),
        "--tau", str(eval_tau),
    ]
    print(f"\n=== Evaluating keyframes for {video_name} ({distance_backend}) ===")
    ok, out, err = _run(eval_cmd)
    if ok:
        print(f"✅ Eval done for {video_name}")
        if out.strip():
            print(out)
    else:
        print(f"❌ Eval failed for {video_name}")
        if err.strip():
            print(err)

    # Visualize
    viz_cmd = [
        "python", "-m", viz_module,
        "--video", video_path,
        "--scenes_json", scenes_json,
        "--keyframes_csv", keyframes_csv,
        "--out_dir", out_dir,
    ]
    print(f"\n=== Visualizing keyframes for {video_name} ({distance_backend}) ===")
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    ok, out, err = _run(viz_cmd, env=env)
    if ok:
        print(f"✅ Visualization done for {video_name}")
        if out.strip():
            print(out)
    else:
        print(f"❌ Visualization failed for {video_name}")
        if err.strip():
            print(err)


# ---------------------------- CLI ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch process videos -> pipeline -> eval -> visualize (all under outputs/)."
    )

    # Data & toggles
    p.add_argument("--data_folder", default="samples", help="Folder containing input videos")
    p.add_argument("--videos_glob", default=None,
                   help="Optional glob (relative to data_folder) to filter videos, e.g. 'classA/**/*.mp4'")
    p.add_argument("--run_pipeline", action="store_true", help="Run pipeline step")
    p.add_argument("--run_evalviz", action="store_true", help="Run eval + visualize step")

    # Roots
    p.add_argument("--output_base", default="outputs", help="Root folder to save ALL outputs")
    p.add_argument("--pipeline_out_dir", default=None, 
                   help="Custom pipeline output directory (overrides --output_base structure)")
    p.add_argument("--eval_out_dir", default=None,
                   help="Custom eval output directory (overrides --output_base structure)")

    # Pipeline options
    p.add_argument("--backend", default="transnetv2", help="Scene detector backend")
    p.add_argument("--model_dir", default="src/models/TransNetV2", help="Path to TransNetV2 model")
    p.add_argument("--prob_threshold", type=float, default=0.5)
    p.add_argument("--distance_backend", choices=["dists", "lpips"], default="dists",
                   help="Distance function for keyframe selection")
    p.add_argument("--dists_as_distance", type=int, choices=[0, 1], default=1,
                   help="Only meaningful for 'dists' backend; leave 1 by default")
    p.add_argument("--sample_stride", type=int, default=12)
    p.add_argument("--max_frames_per_scene", type=int, default=40)
    p.add_argument("--keyframes_per_scene", type=int, default=2)
    p.add_argument("--nms_radius", type=int, default=4)
    p.add_argument("--resize_w", type=int, default=320)
    p.add_argument("--resize_h", type=int, default=320)

    # Eval + Viz options
    p.add_argument("--eval_script", default="eval_keyframes.py")
    p.add_argument("--viz_module", default="eval.visualize.viz_medoids",
                   help="Python module path for visualization (used with -m)")
    p.add_argument("--eval_backbone", default="resnet50")
    p.add_argument("--eval_sample_stride", type=int, default=10)
    p.add_argument("--eval_max_frames", type=int, default=200)
    p.add_argument("--eval_tau", type=float, default=0.3)

    # Parallelization
    p.add_argument("--num_workers", type=int, default=1,
                   help="Number of parallel workers (default=1, sequential)")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve output directories
    if args.pipeline_out_dir:
        output_base_for_pipeline = args.pipeline_out_dir
    else:
        output_base_for_pipeline = args.output_base

    if args.eval_out_dir:
        output_base_for_eval = args.eval_out_dir
    else:
        output_base_for_eval = args.output_base

    videos = find_mp4_videos(args.data_folder, args.videos_glob)
    if not videos:
        print(f"❌ No MP4 videos found in '{args.data_folder}'")
        return

    print(f"📹 Found {len(videos)} video(s):")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v}")

    # Ensure roots
    os.makedirs(os.path.join(output_base_for_pipeline, "pipeline", args.distance_backend), exist_ok=True)
    os.makedirs(os.path.join(output_base_for_eval, "eval", args.distance_backend), exist_ok=True)

    print(f"\n⚙️  Configuration:")
    print(f"  Data folder: {args.data_folder}")
    print(f"  Distance backend: {args.distance_backend}")
    print(f"  Pipeline output: {os.path.join(output_base_for_pipeline, 'pipeline', args.distance_backend)}")
    print(f"  Eval output: {os.path.join(output_base_for_eval, 'eval', args.distance_backend)}")
    print(f"  Workers: {args.num_workers}\n")

    success_count = 0
    failed = []

    for idx, video_path in enumerate(videos, 1):
        print(f"\n--- [{idx}/{len(videos)}] {Path(video_path).name} ---")
        resolved_dir = None

        # PIPELINE
        if args.run_pipeline:
            ok, resolved_dir = run_pipeline_for_video(
                video_path=video_path,
                out_base=output_base_for_pipeline,
                backend=args.backend,
                model_dir=args.model_dir,
                prob_threshold=args.prob_threshold,
                distance_backend=args.distance_backend,
                dists_as_distance=bool(args.dists_as_distance),
                sample_stride=args.sample_stride,
                max_frames_per_scene=args.max_frames_per_scene,
                keyframes_per_scene=args.keyframes_per_scene,
                nms_radius=args.nms_radius,
                resize_w=args.resize_w,
                resize_h=args.resize_h,
            )
            if not ok:
                failed.append(video_path)

        # EVAL + VIZ
        if args.run_evalviz:
            run_eval_and_visualize_for_video(
                video_path=video_path,
                pipeline_dir=resolved_dir,   # có thể None => hàm sẽ tự resolve
                output_base=output_base_for_eval,
                distance_backend=args.distance_backend,
                eval_script=args.eval_script,
                viz_module=args.viz_module,
                eval_backbone=args.eval_backbone,
                eval_sample_stride=args.eval_sample_stride,
                eval_max_frames=args.eval_max_frames,
                eval_tau=args.eval_tau,
            )

        success_count += 1

    print("\n" + "=" * 70)
    print("🎯 BATCH COMPLETE")
    print("=" * 70)
    print(f"✅ Processed: {success_count}/{len(videos)}")
    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for v in failed:
            print(f"  - {v}")
    print(f"\nPipeline output    → {os.path.join(output_base_for_pipeline, 'pipeline', args.distance_backend)}")
    print(f"Eval output        → {os.path.join(output_base_for_eval, 'eval', args.distance_backend)}")


if __name__ == "__main__":
    main()

"""
python batch_processing.py \
  --run_pipeline \
  --run_evalviz \
  --data_folder samples \
  --output_base outputs \
  --eval_out_base outputs_eval \
  --num_workers 4


  python batch_processing.py \
  --run_pipeline \
  --run_evalviz \
  --data_folder samples \
  --distance_backend lpips \
  --num_workers 4
"""