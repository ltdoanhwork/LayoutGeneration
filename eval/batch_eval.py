#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Evaluation Pipeline (DSN version)
- Use eval/run_dsn_pipeline.py to produce scenes.json + keyframes.csv via DSN checkpoint
- Then run scripts/eval_keyframes.py for base metrics
- Then add extra metrics (LPIPS gap/diversity, MS-SWD Color)
"""
import os, json, argparse, subprocess, glob
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count

class BatchEvaluationPipeline:
    def __init__(self, 
                 videos_dir: str,
                 output_base_dir: str,
                 checkpoint: str,
                 feat_dim: int = 512,
                 enc_hidden: int = 256,
                 lstm_hidden: int = 128,
                 budget_ratio: float = 0.06,
                 Bmin: int = 3,
                 Bmax: int = 15,
                 sample_stride: int = 5,
                 resize_w: int = 320,
                 resize_h: int = 180,
                 embedder: str = "resnet50",
                 scene_threshold: int = 27,
                 device: str = "cuda",
                 backend: str = "pyscenedetect",
                 threshold: float | None = None,
                 model_dir: str | None = None,
                 weights_path: str | None = None,
                 prob_threshold: float | None = None,
                 scene_device: str | None = None,
                 debug: bool = False):
        self.videos_dir = videos_dir
        self.output_base_dir = output_base_dir
        self.checkpoint = checkpoint
        self.feat_dim = feat_dim
        self.enc_hidden = enc_hidden
        self.lstm_hidden = lstm_hidden
        self.budget_ratio = budget_ratio
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.sample_stride = sample_stride
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.embedder = embedder
        self.scene_threshold = scene_threshold
        self.device = device
        self.debug = debug
        self.backend = backend
        self.threshold = threshold
        self.model_dir = model_dir
        self.weights_path = weights_path
        self.prob_threshold = prob_threshold
        self.scene_device = scene_device
        self.pipeline_out_dir = os.path.join(output_base_dir, "pipeline_results")
        self.eval_out_dir = os.path.join(output_base_dir, "eval_results")
        os.makedirs(self.pipeline_out_dir, exist_ok=True)
        os.makedirs(self.eval_out_dir, exist_ok=True)
        self.results = {}
        self.errors = {}

    def find_videos(self) -> List[str]:
        exts = ["*.mp4","*.mkv","*.avi","*.mov"]
        vids = []
        for e in exts: vids.extend(glob.glob(os.path.join(self.videos_dir, e)))
        return sorted(vids)

    def _run_keyframe_extraction(self, video_path: str, output_dir: str) -> Optional[Dict[str,str]]:
        try:
            cmd = [
            "python", "-m", "eval.run_dsn_pipeline",
            "--video", video_path,
            "--out_dir", output_dir,
            "--device", self.device,
            "--feat_dim", str(self.feat_dim),
            "--enc_hidden", str(self.enc_hidden),
            "--lstm_hidden", str(self.lstm_hidden),
            "--budget_ratio", str(self.budget_ratio),
            "--Bmin", str(self.Bmin),
            "--Bmax", str(self.Bmax),
            "--sample_stride", str(self.sample_stride),
            "--resize_w", str(self.resize_w),
            "--resize_h", str(self.resize_h),
            "--embedder", self.embedder,
            "--backend", self.backend,
            ]
            if self.checkpoint:
                cmd += ["--checkpoint", self.checkpoint]
            # Detector-specific args (chỉ add nếu có)
            if self.threshold is not None:      cmd += ["--threshold", str(self.threshold)]
            if self.model_dir:                  cmd += ["--model_dir", self.model_dir]
            if self.weights_path:               cmd += ["--weights_path", self.weights_path]
            if self.prob_threshold is not None: cmd += ["--prob_threshold", str(self.prob_threshold)]
            if self.scene_device:               cmd += ["--scene_device", self.scene_device]

            if self.debug: print("  [Debug] Run:", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("  [Error] DSN pipeline failed:", result.stderr)
                return None

            scenes_json = os.path.join(output_dir, "scenes.json")
            keyframes_csv = os.path.join(output_dir, "keyframes.csv")
            if not (os.path.exists(scenes_json) and os.path.exists(keyframes_csv)):
                print("  [Error] Missing output files in", output_dir)
                return None
            return {
                "scenes_json": scenes_json,
                "keyframes_csv": keyframes_csv,
                "output_dir": output_dir,
            }
        except Exception as e:
            print(f"  [Error] Keyframe extraction failed: {e}")
            return None

    def _run_evaluation(self, video_path: str, scenes_json: str, keyframes_csv: str, eval_output_dir: str,
                        eval_backbone: str = "resnet50", eval_device: str = "cuda",
                        eval_sample_stride: int = 1, eval_max_frames: int = 200, eval_tau: float = 0.5,
                        with_baselines: bool = True) -> Optional[Dict]:
        try:
            os.makedirs(eval_output_dir, exist_ok=True)
            cmd = [
                "python", "scripts/eval_keyframes.py",
                "--video", video_path,
                "--scenes_json", scenes_json,
                "--keyframes_csv", keyframes_csv,
                "--out_dir", eval_output_dir,
                "--backbone", eval_backbone,
                "--device", eval_device,
                "--sample_stride", str(eval_sample_stride),
                "--max_frames_eval", str(eval_max_frames),
                "--tau", str(eval_tau),
            ]
            if with_baselines: cmd.append("--with_baselines")
            if self.debug: print("  [Debug] Eval:", " ".join(cmd))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print("  [Error] Evaluation failed:", r.stderr)
                return None

            # Merge extra metrics
            extra_json = os.path.join(eval_output_dir, "extra_metrics.json")
            cmd2 = [
                "python", "-m", "eval.extra_metrics",
                "--video", video_path,
                "--keyframes_csv", keyframes_csv,
                "--out_json", extra_json,
                "--lpips_device", self.device,
                "--lpips_net", "alex",
            ]
            if self.debug: print("  [Debug] Extra:", " ".join(cmd2))
            r2 = subprocess.run(cmd2, capture_output=True, text=True)
            # Load base
            res_json = os.path.join(eval_output_dir, "eval_results.json")
            if not os.path.exists(res_json):
                print("  [Error] eval_results.json not found in", eval_output_dir)
                return None
            with open(res_json, "r", encoding="utf-8") as f:
                base = json.load(f)
            # Merge extras if exists
            if os.path.exists(extra_json):
                with open(extra_json, "r", encoding="utf-8") as f:
                    extra = json.load(f)
                base.update(extra)
                with open(res_json, "w", encoding="utf-8") as f:
                    json.dump(base, f, indent=2, ensure_ascii=False)
            return base
        except Exception as e:
            print(f"  [Error] Evaluation failed: {e}")
            return None

    def process_video(self, video_path: str, video_id: str,
                      eval_backbone="resnet50", eval_device="cuda",
                      eval_sample_stride=1, eval_max_frames=200, eval_tau=0.5,
                      with_baselines=True) -> bool:
        tqdm.write(f"\n=== Processing: {video_id} ===")
        try:
            # 1) Extract (DSN)
            pipeline_out = os.path.join(self.pipeline_out_dir, video_id)
            os.makedirs(pipeline_out, exist_ok=True)
            extraction = self._run_keyframe_extraction(video_path, pipeline_out)
            if extraction is None:
                self.errors[video_id] = "Extraction failed"
                return False
            tqdm.write(f"  ✅ keyframes: {extraction['keyframes_csv']}")

            # 2) Evaluate
            eval_out = os.path.join(self.eval_out_dir, video_id)
            metrics = self._run_evaluation(
                video_path, extraction["scenes_json"], extraction["keyframes_csv"],
                eval_out, eval_backbone, eval_device, eval_sample_stride, eval_max_frames, eval_tau, with_baselines
            )
            if metrics is None:
                self.errors[video_id] = "Evaluation failed"
                return False
            tqdm.write(f"  ✅ evaluation done")

            # 3) Store
            self.results[video_id] = {
                "video_path": video_path,
                "pipeline_output": extraction["output_dir"],
                "eval_output": eval_out,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            tqdm.write(f"  📊 RecErr: {metrics.get('RecErr', 'N/A')} "
                       f"Frechet: {metrics.get('Frechet', 'N/A')} "
                       f"LPIPS_Gap: {metrics.get('LPIPS_PerceptualGap', 'N/A')}")
            return True
        except Exception as e:
            self.errors[video_id] = str(e)
            return False

    def _save_summary(self):
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "videos_dir": self.videos_dir,
                "output_base_dir": self.output_base_dir,
                "checkpoint": self.checkpoint,
                "feat_dim": self.feat_dim,
                "budget_ratio": self.budget_ratio,
            },
            "statistics": {"total_processed": len(self.results), "total_failed": len(self.errors)},
            "results": self.results, "errors": self.errors
        }
        # aggregates
        def safe_mean(xs): 
            vals = [x for x in xs if x is not None and not (isinstance(x,float) and np.isnan(x))]
            return float(np.mean(vals)) if vals else None
        if self.results:
            rec = []; fre = []; scov=[]; tcov=[]; lp_gap=[]; lp_div=[]; ms=[]
            for vid, r in self.results.items():
                # Metrics are nested: r["metrics"]["method"] contains DSN results
                m = r["metrics"].get("method", r["metrics"])  # fallback to r["metrics"] if no "method" key
                rec.append(m.get("RecErr")); fre.append(m.get("Frechet"))
                scov.append(m.get("SceneCoverage")); tcov.append(m.get("TemporalCoverage@tau"))
                lp_gap.append(m.get("LPIPS_PerceptualGap")); lp_div.append(m.get("LPIPS_DiversitySel"))
                ms.append(m.get("MS_SWD_Color"))
            summary["aggregate_metrics"] = {
                "RecErr_mean": safe_mean(rec),
                "Frechet_mean": safe_mean(fre),
                "SceneCoverage_mean": safe_mean(scov),
                "TemporalCoverage@tau_mean": safe_mean(tcov),
                "LPIPS_PerceptualGap_mean": safe_mean(lp_gap),
                "LPIPS_DiversitySel_mean": safe_mean(lp_div),
                "MS_SWD_Color_mean": safe_mean(ms),
            }
        outp = os.path.join(self.output_base_dir, "summary_results.json")
        with open(outp, "w", encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False)
        tqdm.write(f"\n📊 Summary saved to: {outp}")

def main():
    ps = argparse.ArgumentParser("Batch eval with DSN")
    ps.add_argument("--videos_dir", required=True)
    ps.add_argument("--output_dir", required=True)
    ps.add_argument("--checkpoint", type=str, default=None, help="DSN checkpoint path. If None, use untrained DSN.")
    ps.add_argument("--device", default="cuda")
    ps.add_argument("--feat_dim", type=int, default=512)
    ps.add_argument("--enc_hidden", type=int, default=256)
    ps.add_argument("--lstm_hidden", type=int, default=128)
    ps.add_argument("--budget_ratio", type=float, default=0.06)
    ps.add_argument("--Bmin", type=int, default=3)
    ps.add_argument("--Bmax", type=int, default=15)
    ps.add_argument("--sample_stride", type=int, default=5)
    ps.add_argument("--resize_w", type=int, default=320)
    ps.add_argument("--resize_h", type=int, default=180)
    ps.add_argument("--embedder", type=str, default="resnet50")
    ps.add_argument("--scene_threshold", type=int, default=27)
    ps.add_argument("--eval_backbone", type=str, default="resnet50")
    ps.add_argument("--eval_device", type=str, default="cuda")
    ps.add_argument("--eval_sample_stride", type=int, default=1)
    ps.add_argument("--eval_max_frames", type=int, default=200)
    ps.add_argument("--eval_tau", type=float, default=0.5)
    ps.add_argument("--with_baselines", action="store_true")
    ps.add_argument("--max_videos", "--limit", type=int, default=None, 
                    help="Limit the number of videos to process")
    ps.add_argument("--num_workers", type=int, default=1)
    ps.add_argument("--debug", action="store_true")
    ps.add_argument("--backend", type=str, default="pyscenedetect",
                help="Scene detector backend: pyscenedetect | transnetv2")
    ps.add_argument("--threshold", type=float, default=27.0)
    ps.add_argument("--model_dir", type=str, default=None)
    ps.add_argument("--weights_path", type=str, default=None)
    ps.add_argument("--prob_threshold", type=float, default=None)
    ps.add_argument("--scene_device", type=str, default=None)

    args = ps.parse_args()

    pipe = BatchEvaluationPipeline(
    videos_dir=args.videos_dir, output_base_dir=args.output_dir,
    checkpoint=args.checkpoint, feat_dim=args.feat_dim,
    enc_hidden=args.enc_hidden, lstm_hidden=args.lstm_hidden,
    budget_ratio=args.budget_ratio, Bmin=args.Bmin, Bmax=args.Bmax,
    sample_stride=args.sample_stride, resize_w=args.resize_w, resize_h=args.resize_h,
    embedder=args.embedder, device=args.eval_device, debug=args.debug,
    backend=args.backend, threshold=args.threshold, model_dir=args.model_dir,
    weights_path=args.weights_path, prob_threshold=args.prob_threshold, scene_device=args.scene_device
    )


    videos = pipe.find_videos()
    if args.max_videos: videos = videos[:args.max_videos]
    ok = 0; fail = 0
    for vp in tqdm(videos, desc="Batch Eval"):
        vid = os.path.splitext(os.path.basename(vp))[0]
        if pipe.process_video(vp, vid, args.eval_backbone, args.eval_device,
                              args.eval_sample_stride, args.eval_max_frames,
                              args.eval_tau, args.with_baselines):
            ok += 1
        else:
            fail += 1
    print(f"\nDone. Ok={ok} Fail={fail}")
    pipe._save_summary()



if __name__ == "__main__":
    main()

"""
python -m eval.batch_eval \
    --videos_dir "data/samples/Sakuga" \
    --output_dir "outputs/val_runs/test_no_ckpt" \
    --resize_w 320 \
    --resize_h 180 \
    --sample_stride 5 \
    --embedder clip_vitb32 \
    --backend pyscenedetect \
    --threshold 27 \
    --eval_backbone resnet50 \
    --eval_device cuda \
    --eval_sample_stride 1 \
    --eval_max_frames 200 \
    --eval_tau 0.5 \
    --with_baselines \
    --debug \
    --limit 1
"""