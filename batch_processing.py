#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for keyframe pipeline + evaluation + visualization.
- Backends: dists / lpips
- Save everything under outputs/
    outputs/pipeline/<backend>/run_tv2_<backend>_<video>/
    outputs/eval/<backend>/eval_<video>/

Supports:
- Video input (MP4 files)
- Image folder input (folder containing images)
- Image list input (text file with image paths)
"""

import os
import glob
import subprocess
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
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


def find_images_in_folder(folder: str, extensions: List[str] = None) -> List[str]:
    """Find all image files in a folder."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif']
    
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder, f'*{ext}')))
        images.extend(glob.glob(os.path.join(folder, f'*{ext.upper()}')))
    return sorted(images)


def load_image_list_from_file(list_file: str) -> List[str]:
    """Load image paths from a text file (one path per line)."""
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"Image list file not found: {list_file}")
    
    images = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                if os.path.isfile(line):
                    images.append(line)
                else:
                    print(f"  [WARN] Image not found, skipping: {line}")
    return images


def find_image_folders(data_folder: str) -> List[str]:
    """Find all subfolders containing images."""
    folders = []
    for root, dirs, files in os.walk(data_folder):
        images = find_images_in_folder(root)
        if images:
            folders.append(root)
    return sorted(folders)


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

    # Pre-check: file exists and is readable
    if not os.path.isfile(video_path):
        print(f"\n❌ SKIP: Video file not found: {video_path}")
        return False, ""

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
    resolved_dir = ""
    if ok:
        try:
            resolved_dir = resolve_pipeline_dir(video_name, distance_backend, out_base)
        except FileNotFoundError as e:
            resolved_dir = os.path.join(out_dir_base, f"_{video_name}")  # fallback hiển thị
            ok = False  # Mark as failed if artifacts not found
            print(str(e))
    
    if ok and resolved_dir:
        print(f"✅ PIPELINE OK: {video_name}")
        print(f"Artifacts: {resolved_dir}")
        if out.strip():
            print(out)
    else:
        print(f"❌ PIPELINE FAIL: {video_name}")
        if err.strip():
            print(f"   Error: {err[:200]}...")  # Truncate long errors
        resolved_dir = ""

    return ok, resolved_dir


# ---------------------------- Image List Pipeline ----------------------------

def run_dsn_pipeline_for_images(
    image_list: List[str],
    out_dir: str,
    checkpoint: str,
    device: str = "cuda",
    embedder: str = "clip_vitb32",
    budget_ratio: float = 0.06,
    Bmin: int = 3,
    Bmax: int = 8,
    input_shape_layout: str = None,
    scaling_factor: int = 1,
    use_prob_priority: bool = True,
    resize_w: int = 320,
    resize_h: int = 180,
    name: str = "image_batch",
) -> Tuple[bool, str]:
    """
    Run DSN pipeline for a list of images (instead of video).
    Uses layout_decomposer_dsn_pipeline.py with --image_list option.
    
    Args:
        image_list: List of image file paths
        out_dir: Output directory
        checkpoint: Path to DSN checkpoint
        device: cuda or cpu
        embedder: Embedding model (clip_vitb32, resnet50, etc.)
        budget_ratio: Budget ratio for keyframe selection
        Bmin, Bmax: Min/max keyframes per scene
        input_shape_layout: Path to shape layout image for Colla
        scaling_factor: Scaling factor for collage
        use_prob_priority: Whether to use prob priority for collage
        resize_w, resize_h: Resize dimensions for processing
        name: Name for output folder
    
    Returns:
        (success, output_dir)
    """
    import tempfile
    import shutil
    from datetime import datetime
    
    if not image_list:
        print("❌ No images provided")
        return False, ""
    
    # Create temp file with image list
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_full = os.path.join(out_dir, f"{name}_{timestamp}")
    os.makedirs(out_dir_full, exist_ok=True)
    
    # Write image list to file
    image_list_file = os.path.join(out_dir_full, "image_list.txt")
    with open(image_list_file, 'w') as f:
        for img_path in image_list:
            f.write(f"{img_path}\n")
    
    print(f"\n{'='*70}")
    print(f"Processing {len(image_list)} images")
    print(f"Output: {out_dir_full}")
    print(f"{'='*70}")
    
    # Build command
    cmd = [
        "python", "layout_decomposer_dsn_pipeline.py",
        "--image_list", image_list_file,
        "--out_dir", out_dir_full,
        "--checkpoint", checkpoint,
        "--device", device,
        "--embedder", embedder,
        "--budget_ratio", str(budget_ratio),
        "--Bmin", str(Bmin),
        "--Bmax", str(Bmax),
        "--resize_w", str(resize_w),
        "--resize_h", str(resize_h),
    ]
    
    if input_shape_layout:
        cmd.extend(["--input_shape_layout", input_shape_layout])
    
    cmd.extend(["--scaling_factor", str(scaling_factor)])
    
    if use_prob_priority:
        cmd.append("--use_prob_priority")
    else:
        cmd.append("--no_prob_priority")
    
    ok, out, err = _run(cmd)
    
    if ok:
        print(f"✅ IMAGE PIPELINE OK: {name}")
        print(f"Output: {out_dir_full}")
        if out.strip():
            # Print last 20 lines
            lines = out.strip().split('\n')
            if len(lines) > 20:
                print("... (truncated)")
            print('\n'.join(lines[-20:]))
    else:
        print(f"❌ IMAGE PIPELINE FAIL: {name}")
        if err.strip():
            print(f"   Error: {err[:500]}...")
    
    return ok, out_dir_full


def run_pipeline_for_image_folder(
    image_folder: str,
    out_base: str,
    checkpoint: str,
    device: str = "cuda",
    embedder: str = "clip_vitb32",
    budget_ratio: float = 0.06,
    Bmin: int = 3,
    Bmax: int = 8,
    input_shape_layout: str = None,
    scaling_factor: int = 1,
    use_prob_priority: bool = True,
    resize_w: int = 320,
    resize_h: int = 180,
) -> Tuple[bool, str]:
    """
    Run DSN pipeline for all images in a folder.
    """
    folder_name = Path(image_folder).name
    images = find_images_in_folder(image_folder)
    
    if not images:
        print(f"❌ No images found in {image_folder}")
        return False, ""
    
    print(f"📁 Found {len(images)} images in {image_folder}")
    
    return run_dsn_pipeline_for_images(
        image_list=images,
        out_dir=out_base,
        checkpoint=checkpoint,
        device=device,
        embedder=embedder,
        budget_ratio=budget_ratio,
        Bmin=Bmin,
        Bmax=Bmax,
        input_shape_layout=input_shape_layout,
        scaling_factor=scaling_factor,
        use_prob_priority=use_prob_priority,
        resize_w=resize_w,
        resize_h=resize_h,
        name=folder_name,
    )


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

    # Check if pipeline_dir is valid
    if pipeline_dir is None or pipeline_dir == "" or \
       not (os.path.isfile(os.path.join(pipeline_dir, "scenes.json")) and
            os.path.isfile(os.path.join(pipeline_dir, "keyframes.csv"))):
        try:
            pipeline_dir = resolve_pipeline_dir(video_name, distance_backend, output_base)
        except FileNotFoundError as e:
            print(f"\n❌ SKIP EVAL: Pipeline artifacts not found for {video_name}")
            print(f"   {str(e)[:100]}...")
            return  # Skip eval for this video

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
        description="Batch process videos/images -> pipeline -> eval -> visualize (all under outputs/)."
    )

    # =========================
    # Input mode selection
    # =========================
    p.add_argument("--mode", choices=["video", "images", "image_folder", "image_list"], 
                   default="video",
                   help="Input mode: 'video' (default), 'images' (folder of images), "
                        "'image_folder' (process each subfolder), 'image_list' (text file with paths)")
    
    # Data & toggles
    p.add_argument("--data_folder", default="samples", 
                   help="Folder containing input videos or images")
    p.add_argument("--image_list_file", default=None,
                   help="Path to text file containing image paths (one per line)")
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

    # =========================
    # Video Pipeline options (original)
    # =========================
    p.add_argument("--backend", default="transnetv2", help="Scene detector backend")
    p.add_argument("--model_dir", default="src/models/TransNetV2", help="Path to TransNetV2 model")
    p.add_argument("--prob_threshold", type=float, default=0.5)
    p.add_argument("--distance_backend", choices=["dists", "lpips"], default="dists",
                   help="Distance function for keyframe selection")
    p.add_argument("--dists_as_distance", type=int, choices=[0, 1], default=1,
                   help="Only meaningful for 'dists' backend; leave 1 by default")
    p.add_argument("--sample_stride", type=int, default=5)
    p.add_argument("--max_frames_per_scene", type=int, default=40)
    p.add_argument("--keyframes_per_scene", type=int, default=2)
    p.add_argument("--nms_radius", type=int, default=4)
    p.add_argument("--resize_w", type=int, default=320)
    p.add_argument("--resize_h", type=int, default=320)

    # =========================
    # Image/DSN Pipeline options (new)
    # =========================
    p.add_argument("--checkpoint", default="runs/dsn_advanced_v1/dsn_checkpoint_ep8.pt",
                   help="Path to DSN model checkpoint")
    p.add_argument("--device", default="cuda", help="Device for inference (cuda/cpu)")
    p.add_argument("--embedder", default="clip_vitb32", 
                   choices=["clip_vitb32", "resnet50", "classic"],
                   help="Embedding model for image features")
    p.add_argument("--budget_ratio", type=float, default=0.06,
                   help="Budget ratio for keyframe selection")
    p.add_argument("--Bmin", type=int, default=3, help="Minimum keyframes per scene")
    p.add_argument("--Bmax", type=int, default=8, help="Maximum keyframes per scene")
    p.add_argument("--input_shape_layout", default=None,
                   help="Path to shape layout image for Colla collage")
    p.add_argument("--scaling_factor", type=int, default=1,
                   help="Scaling factor for collage output")
    p.add_argument("--use_prob_priority", action="store_true", default=True,
                   help="Use DSN probability for priority placement in collage")
    p.add_argument("--no_prob_priority", action="store_false", dest="use_prob_priority",
                   help="Disable prob-based priority")

    # Eval + Viz options
    p.add_argument("--eval_script", default="scripts/eval_keyframes.py")
    p.add_argument("--viz_module", default="utils.visualize.viz_medoids",
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

    # =========================
    # IMAGE MODE PROCESSING
    # =========================
    if args.mode in ["images", "image_folder", "image_list"]:
        print(f"\n{'='*70}")
        print(f"🖼️  IMAGE MODE: {args.mode}")
        print(f"{'='*70}")
        
        os.makedirs(output_base_for_pipeline, exist_ok=True)
        
        success_count = 0
        failed = []
        
        if args.mode == "image_list":
            # Process images from a text file
            if not args.image_list_file:
                print("❌ --image_list_file required for mode 'image_list'")
                return
            
            try:
                images = load_image_list_from_file(args.image_list_file)
                print(f"📄 Loaded {len(images)} images from {args.image_list_file}")
                
                if args.run_pipeline:
                    ok, out_dir = run_dsn_pipeline_for_images(
                        image_list=images,
                        out_dir=output_base_for_pipeline,
                        checkpoint=args.checkpoint,
                        device=args.device,
                        embedder=args.embedder,
                        budget_ratio=args.budget_ratio,
                        Bmin=args.Bmin,
                        Bmax=args.Bmax,
                        input_shape_layout=args.input_shape_layout,
                        scaling_factor=args.scaling_factor,
                        use_prob_priority=args.use_prob_priority,
                        resize_w=args.resize_w,
                        resize_h=args.resize_h,
                        name=Path(args.image_list_file).stem,
                    )
                    if ok:
                        success_count += 1
                    else:
                        failed.append((args.image_list_file, "Pipeline failed"))
            except Exception as e:
                print(f"❌ Error: {e}")
                failed.append((args.image_list_file, str(e)[:50]))
        
        elif args.mode == "images":
            # Process all images in data_folder as one batch
            images = find_images_in_folder(args.data_folder)
            if not images:
                print(f"❌ No images found in '{args.data_folder}'")
                return
            
            print(f"🖼️  Found {len(images)} images in {args.data_folder}")
            for i, img in enumerate(images[:10], 1):
                print(f"  {i}. {Path(img).name}")
            if len(images) > 10:
                print(f"  ... and {len(images) - 10} more")
            
            if args.run_pipeline:
                ok, out_dir = run_dsn_pipeline_for_images(
                    image_list=images,
                    out_dir=output_base_for_pipeline,
                    checkpoint=args.checkpoint,
                    device=args.device,
                    embedder=args.embedder,
                    budget_ratio=args.budget_ratio,
                    Bmin=args.Bmin,
                    Bmax=args.Bmax,
                    input_shape_layout=args.input_shape_layout,
                    scaling_factor=args.scaling_factor,
                    use_prob_priority=args.use_prob_priority,
                    resize_w=args.resize_w,
                    resize_h=args.resize_h,
                    name=Path(args.data_folder).name,
                )
                if ok:
                    success_count += 1
                else:
                    failed.append((args.data_folder, "Pipeline failed"))
        
        elif args.mode == "image_folder":
            # Process each subfolder as a separate batch
            folders = find_image_folders(args.data_folder)
            if not folders:
                print(f"❌ No image folders found in '{args.data_folder}'")
                return
            
            print(f"📁 Found {len(folders)} folder(s) with images:")
            for i, f in enumerate(folders, 1):
                n_imgs = len(find_images_in_folder(f))
                print(f"  {i}. {f} ({n_imgs} images)")
            
            if args.run_pipeline:
                for idx, folder in enumerate(folders, 1):
                    print(f"\n--- [{idx}/{len(folders)}] {folder} ---")
                    try:
                        ok, out_dir = run_pipeline_for_image_folder(
                            image_folder=folder,
                            out_base=output_base_for_pipeline,
                            checkpoint=args.checkpoint,
                            device=args.device,
                            embedder=args.embedder,
                            budget_ratio=args.budget_ratio,
                            Bmin=args.Bmin,
                            Bmax=args.Bmax,
                            input_shape_layout=args.input_shape_layout,
                            scaling_factor=args.scaling_factor,
                            use_prob_priority=args.use_prob_priority,
                            resize_w=args.resize_w,
                            resize_h=args.resize_h,
                        )
                        if ok:
                            success_count += 1
                        else:
                            failed.append((folder, "Pipeline failed"))
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        failed.append((folder, str(e)[:50]))
        
        # Summary
        print("\n" + "=" * 70)
        print("🎯 IMAGE BATCH COMPLETE")
        print("=" * 70)
        total = 1 if args.mode in ["images", "image_list"] else len(folders)
        print(f"✅ Processed: {success_count}/{total}")
        if failed:
            print(f"❌ Failed ({len(failed)}):")
            for item, reason in failed:
                print(f"  - {item}: {reason}")
        print(f"\nOutput → {output_base_for_pipeline}")
        return

    # =========================
    # VIDEO MODE PROCESSING (original)
    # =========================
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
        pipeline_ok = False

        try:
            # PIPELINE
            if args.run_pipeline:
                pipeline_ok, resolved_dir = run_pipeline_for_video(
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
                if not pipeline_ok:
                    failed.append((video_path, "Pipeline processing failed"))
                else:
                    success_count += 1

            # EVAL + VIZ (only if pipeline succeeded or not running pipeline)
            if args.run_evalviz and (pipeline_ok or not args.run_pipeline):
                try:
                    run_eval_and_visualize_for_video(
                        video_path=video_path,
                        pipeline_dir=resolved_dir if pipeline_ok else None,
                        output_base=output_base_for_eval,
                        distance_backend=args.distance_backend,
                        eval_script=args.eval_script,
                        viz_module=args.viz_module,
                        eval_backbone=args.eval_backbone,
                        eval_sample_stride=args.eval_sample_stride,
                        eval_max_frames=args.eval_max_frames,
                        eval_tau=args.eval_tau,
                    )
                except Exception as e:
                    print(f"\n❌ EVAL/VIZ ERROR for {Path(video_path).name}: {str(e)[:100]}")
                    if pipeline_ok:
                        failed.append((video_path, f"Eval/Viz failed: {str(e)[:50]}"))
            elif args.run_evalviz:
                failed.append((video_path, "Skipped eval (pipeline failed)"))

        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR for {Path(video_path).name}: {str(e)[:100]}")
            failed.append((video_path, f"Unexpected error: {str(e)[:50]}"))

    print("\n" + "=" * 70)
    print("🎯 BATCH COMPLETE")
    print("=" * 70)
    print(f"✅ Processed: {success_count}/{len(videos)}")
    if failed:
        print(f"❌ Failed/Skipped ({len(failed)}):")
        for v, reason in failed:
            print(f"  - {Path(v).name}: {reason}")
    print(f"\nPipeline output    → {os.path.join(output_base_for_pipeline, 'pipeline', args.distance_backend)}")
    print(f"Eval output        → {os.path.join(output_base_for_eval, 'eval', args.distance_backend)}")


if __name__ == "__main__":
    main()

"""
=============================================================================
USAGE EXAMPLES
=============================================================================

# 1. VIDEO MODE (original) - Process videos in a folder
python batch_processing.py \
  --mode video \
  --run_pipeline \
  --data_folder /path/to/videos \
  --output_base outputs \
  --distance_backend dists

# 2. IMAGE MODE - Process all images in a folder as one batch
python batch_processing.py \
  --mode images \
  --run_pipeline \
  --data_folder /path/to/images \
  --output_base outputs \
  --checkpoint runs/dsn_advanced_v1/dsn_checkpoint_ep8.pt \
  --input_shape_layout repos/Colla/input_data/layout/baby.png

# 3. IMAGE FOLDER MODE - Process each subfolder as a separate batch
python batch_processing.py \
  --mode image_folder \
  --run_pipeline \
  --data_folder /path/to/parent_folder \
  --output_base outputs \
  --checkpoint runs/dsn_advanced_v1/dsn_checkpoint_ep8.pt

# 4. IMAGE LIST MODE - Process images listed in a text file
python batch_processing.py \
  --mode image_list \
  --run_pipeline \
  --image_list_file /path/to/image_list.txt \
  --output_base outputs \
  --checkpoint runs/dsn_advanced_v1/dsn_checkpoint_ep8.pt \
  --use_prob_priority

# Example image_list.txt format:
# /path/to/image1.jpg
# /path/to/image2.png
# # This is a comment (lines starting with # are ignored)
# /path/to/image3.jpg

=============================================================================
VIDEO PIPELINE EXAMPLES (with evaluation)
=============================================================================

# LPIPS backend:
python batch_processing.py \
  --mode video \
  --run_pipeline \
  --run_evalviz \
  --data_folder /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test \
  --output_base outputs/batch_eval_lpips \
  --distance_backend lpips \
  --backend transnetv2 \
  --prob_threshold 0.5 \
  --sample_stride 5

# DISTS backend:
python batch_processing.py \
  --mode video \
  --run_pipeline \
  --run_evalviz \
  --data_folder /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test \
  --output_base outputs/batch_eval_dists \
  --distance_backend dists \
  --backend transnetv2 \
  --prob_threshold 0.5

=============================================================================
IMAGE DSN PIPELINE EXAMPLES (with Colla collage)
=============================================================================

# Process keyframes folder with DSN scoring + Colla collage:
python batch_processing.py \
  --mode images \
  --run_pipeline \
  --data_folder outputs/test_prob_priority_70025_20251203_064529/keyframes \
  --output_base outputs/image_batch_test \
  --checkpoint runs/dsn_advanced_v1/dsn_checkpoint_ep8.pt \
  --embedder clip_vitb32 \
  --budget_ratio 0.06 \
  --Bmin 3 --Bmax 8 \
  --input_shape_layout repos/Colla/input_data/layout/baby.png \
  --use_prob_priority
"""