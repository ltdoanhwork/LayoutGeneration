#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# All comments are in English.

from __future__ import annotations
import os
import json
import argparse
import glob
import subprocess
import shutil
from itertools import product
from typing import Dict, Any, List, Tuple

import numpy as np
import sys
# Ensure project root is on sys.path so imports like `eval` and `src` work
# when this script is executed from a subdirectory or via a relative path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import logic cốt lõi từ evaluator, giống như eval_keyframes.py
from eval.evaluator import (
    EvalConfig, load_scenes_json, load_keyframes_csv,
    eval_one_set
)

# --- HELPER FUNCTIONS ---

def save_best_params(params: dict, path: str):
    """Saves the best hyperparameter set to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    print(f"[Optimizer] Saved best params to: {path}")

def load_param_grid(json_path: str) -> dict:
    """Loads the hyperparameter search space from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        grid = json.load(f)
    print(f"[Optimizer] Loaded parameter grid from: {json_path}")
    return grid

# --- CORE CLASSES ---

class HyperparameterOptimizer:
    def __init__(self, samples_dir: str, pipeline_script: str,
                 base_output_dir: str, eval_config: EvalConfig,
                 pipeline_base_args: List[str],
                 target_metric: str = "Combined",
                 max_videos: int = 2,
                 model_dir: str = None,
                 early_stopping_patience: int = 5):
        self.samples_dir = samples_dir
        self.pipeline_script = pipeline_script
        self.base_output_dir = base_output_dir
        self.eval_config = eval_config
        self.model_dir = model_dir
        # Use the pipeline_base_args provided by OptimizerApp (do not force lpips)
        self.pipeline_base_args = list(pipeline_base_args) if pipeline_base_args else []
        # Ensure model_dir is present in base args when provided
        if model_dir and "--model_dir" not in self.pipeline_base_args:
            self.pipeline_base_args.extend(["--model_dir", model_dir])
        self.target_metric = target_metric
        self.video_paths = self._find_video_files()
        self.max_videos = max_videos  # Limit number of videos to test
        self.early_stopping_patience = early_stopping_patience  # Early stopping patience
        self.best_score_history = []  # Track best scores over iterations


    def _find_video_files(self) -> List[str]:
        """Finds all video files (mp4, mkv, avi) in the samples directory."""
        extensions = ["*.mp4", "*.mkv", "*.avi", "*.mov"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.samples_dir, ext)))
        return files

    def _run_selection_pipeline(self, video_path: str, hyperparams: Dict[str, Any], run_output_dir: str) -> Tuple[str, str]:
        """
        Runs the external keyframe selection script (e.g., pipeline.py)
        using the provided hyperparameters.
        """
        os.makedirs(run_output_dir, exist_ok=True)
        
        cmd = [
            "python", self.pipeline_script,
            "--video", video_path,
            "--out_dir", run_output_dir
        ]
        
        # 1. Add static base arguments (e.g., --backend, --sample_stride 3)
        cmd.extend(self.pipeline_base_args)
        
        # 2. Add dynamic hyperparameters from grid search (e.g. --threshold 27)
        for key, value in hyperparams.items():
            cmd.extend([f"--{key}", str(value)])
            
        
        full_command_str = ' '.join(cmd)
        
        try:
            # 🔧 FIX: Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 🔧 FIX: Set PYTHONPATH to include project root
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
            
            # Chạy lệnh từ project root để fix import errors
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                cwd=project_root,  # Run từ project root
                env=env              # Set PYTHONPATH
            )

            # IN OUTPUT NGAY CẢ KHI THÀNH CÔNG (để gỡ lỗi)
            if result.stdout:
                print(f"  [Pipeline STDOUT] {result.stdout.strip()}")
            if result.stderr:
                print(f"  [Pipeline STDERR] {result.stderr.strip()}")

        except subprocess.CalledProcessError as e:
            # Lỗi này xảy ra nếu pipeline.py thoát với code khác 0
            print(f"  [Error] Pipeline script FAILED (non-zero exit) for {video_path}.")
            print(f"  [Error] Full command: {full_command_str}")
            print(f"  [Error] Working directory: {project_root}")
            print(f"  [Error] PYTHONPATH: {env.get('PYTHONPATH')}")
            print(f"  [Error] STDOUT: {e.stdout.strip()}")
            print(f"  [Error] STDERR: {e.stderr.strip()}")
            raise  # Re-raise exception để dừng


        # pipeline.py appends video name to out_dir, so we need to find the actual output dir
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        actual_output_dir = f"{run_output_dir}_{video_name}"
        
        # Kiểm tra file (như cũ)
        scenes_json_path = os.path.join(actual_output_dir, "scenes.json")
        keyframes_csv_path = os.path.join(actual_output_dir, "keyframes.csv")

        if not os.path.exists(scenes_json_path) or not os.path.exists(keyframes_csv_path):
            print(f"  [Debug] Command executed successfully but files are missing.")
            print(f"  [Debug] Checked for: {scenes_json_path}")
            print(f"  [Debug] Full command was: {full_command_str}")
            print(f"  [Debug] Actual output dir: {actual_output_dir}")
            raise FileNotFoundError(
                f"Pipeline script did not produce expected 'scenes.json' or 'keyframes.csv' in {actual_output_dir}"
            )
            
        return scenes_json_path, keyframes_csv_path

    def _run_evaluation(self, video_path: str, scenes_json_path: str, keyframes_csv_path: str, eval_output_dir: str) -> Dict[str, float]:
        """
        Runs the evaluation script (eval_keyframes.py) with the provided arguments.
        """
        try:
            os.makedirs(eval_output_dir, exist_ok=True)
            
            cmd = [
                "python", "scripts/eval_keyframes.py",
                "--video", video_path,
                "--scenes_json", scenes_json_path,
                "--keyframes_csv", keyframes_csv_path,
                "--out_dir", eval_output_dir,
                "--backbone", self.eval_config.backbone,
                "--sample_stride", str(self.eval_config.sample_stride),
                "--max_frames_eval", str(self.eval_config.max_frames_eval),
                "--tau", str(self.eval_config.tau_temporal),
            ]
            if self.eval_config.device:
                cmd.extend(["--device", self.eval_config.device])

            # 🔧 FIX: Set PYTHONPATH and working directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"

            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                cwd=project_root,  # Run từ project root
                env=env              # Set PYTHONPATH
            )
            print(f"  [Evaluation STDOUT] {result.stdout.strip()}")
            if result.stderr:
                print(f"  [Evaluation STDERR] {result.stderr.strip()}")

            # Load evaluation results from the eval output directory
            eval_results_path = os.path.join(eval_output_dir, "eval_results.json")
            if not os.path.exists(eval_results_path):
                print(f"  [Error] eval_results.json not found at {eval_results_path}")
                return {}
                
            with open(eval_results_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            print(f"  [Evaluation Metrics] {metrics}")
            return metrics

        except Exception as e:
            print(f"  [Error] Evaluation failed for {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _get_target_metric(self, metrics: Dict[str, float]) -> float:
        """
        Calculates the final score to be optimized from the metrics dict.
        """
        if not metrics:
            print(f"  [Warning] Metrics is empty, returning 0.0")
            return 0.0 # Penalize failures

        if self.target_metric == "Combined":
            rep = 1 - metrics.get("RecErr", 0.0)

            
            # Build the tau key
            tau_key = f"TemporalCoverage@{self.eval_config.tau_temporal}"
            cov = metrics.get(tau_key, 0.0)
            if cov == 0.0:
                # Try alternative names
                cov = metrics.get("TemporalCoverage@tau", 0.0)
                cov = metrics.get("SceneCoverage", 0.0) if cov == 0.0 else cov
            
            combined_score = (rep * 0.7) + (cov * 0.3)
            print(f"    [Score Details] Rep={rep:.4f}, Cov={cov:.4f}, Combined={combined_score:.4f}")
            return combined_score
        elif self.target_metric in metrics:
            score = metrics.get(self.target_metric, 0.0)
            print(f"    [Score Details] {self.target_metric}={score:.4f}")
            return score
        else:
            print(f"  [Warning] Target metric '{self.target_metric}' not found in {list(metrics.keys())}")
            print(f"  [Warning] Available metrics: {list(metrics.keys())}")
            # Return first numeric value as fallback
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    print(f"  [Warning] Using fallback metric {key}={val:.4f}")
                    return val
            return 0.0

    def _objective_function(self, hyperparams: Dict[str, Any]) -> float:
        """
        The main fitness function.
        """
        print(f"\n[Testing Params] {hyperparams}")
        total_score = 0.0
        processed_videos = 0
        
        param_hash = hash(frozenset(hyperparams.items()))
        param_run_dir = os.path.join(self.base_output_dir, f"param_run_{param_hash}")

        for video_path in self.video_paths[:self.max_videos]:  # Limit to max_videos
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            run_output_dir = os.path.join(param_run_dir, video_name, "pipeline")
            eval_output_dir = os.path.join(param_run_dir, video_name, "eval")
            
            try:
                # 1. Run selection pipeline (pipeline.py)
                scenes_json, keyframes_csv = self._run_selection_pipeline(
                    video_path, hyperparams, run_output_dir
                )
                
                # 2. Run evaluation (using eval_keyframes.py)
                metrics = self._run_evaluation(video_path, scenes_json, keyframes_csv, eval_output_dir)
                
                # 3. Get score
                score = self._get_target_metric(metrics)
                total_score += score
                processed_videos += 1
                print(f"  [Video] {video_name}: Score = {score:.4f}")

            except Exception as e:
                print(f"  [Video Failed] {video_name}. Reason: {e}")
                import traceback
                traceback.print_exc()
        
        if processed_videos == 0:
            print(f"[Param Result] No videos processed successfully")
            return 0.0
            
        avg_score = total_score / processed_videos
        print(f"[Param Avg Score] {avg_score:.4f}")
        
        return avg_score

    def _should_stop_early(self, current_best_score: float) -> bool:
        """
        Check if we should stop early based on patience.
        Returns True if no improvement for 'patience' iterations.
        """
        self.best_score_history.append(current_best_score)
        
        if len(self.best_score_history) < self.early_stopping_patience + 1:
            return False
        
        # Check if best score improved in the last 'patience' iterations
        recent_scores = self.best_score_history[-self.early_stopping_patience:]
        if len(set(recent_scores)) == 1:  # All scores are the same (no improvement)
            print(f"\n⚠️  [Early Stopping] No improvement for {self.early_stopping_patience} iterations!")
            print(f"   Best score: {current_best_score:.4f}")
            return True
        
        return False

    def run_grid_search(self, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Performs a full grid search over the hyperparameter space with early stopping.
        """
        keys = param_grid.keys()
        values = param_grid.values()
        
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        print(f"[Optimizer] Starting Grid Search with {len(param_combinations)} combinations.")
        print(f"[Optimizer] Early stopping patience: {self.early_stopping_patience} iterations\n")
        
        best_score = -float('inf')
        best_params = None
        all_results = []
        iteration = 0

        for hyperparams in param_combinations:
            iteration += 1
            avg_score = self._objective_function(hyperparams)
            all_results.append({"params": hyperparams, "score": avg_score})
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = hyperparams
                print(f"🎯 [New Best Score] {best_score:.4f} at iteration {iteration}")
            
            # Check early stopping
            if self._should_stop_early(best_score):
                print(f"\n⏹️  Stopping early at iteration {iteration}/{len(param_combinations)}")
                break
        
        results_path = os.path.join(self.base_output_dir, "optimizer_all_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "="*70)
        print("--- [Optimization Complete] ---")
        print("="*70)
        print(f"✅ Total iterations: {iteration}/{len(param_combinations)}")
        print(f"🏆 Best Score: {best_score:.4f}")
        print(f"📋 Best Hyperparameters: {best_params}")
        
        best_params_path = os.path.join(self.base_output_dir, "optimizer_best_params.json")
        save_best_params(best_params, best_params_path)
        
        return best_params


class OptimizerApp:
    """
    Handles command-line argument parsing and initialization
    of the optimization process.
    """
    
    def __init__(self):
        # Parse known args and collect unknown args as base args to forward to the pipeline
        known_args, unknown_args = self._parse_args()
        self.args = known_args

        # Start pipeline_base from any extra CLI args the user provided
        pipeline_base: List[str] = list(unknown_args)

        # Ensure backend and distance backend flags are present (use CLI defaults otherwise)
        if "--backend" not in pipeline_base:
            pipeline_base.extend(["--backend", self.args.backend])
        if "--distance_backend" not in pipeline_base:
            pipeline_base.extend(["--distance_backend", self.args.distance_backend])

        # Append lpips/dists specific flags only when relevant and not already provided
        if self.args.distance_backend == "lpips" and "--lpips_net" not in pipeline_base:
            pipeline_base.extend(["--lpips_net", self.args.lpips_net])
        if self.args.distance_backend == "dists" and "--dists_as_distance" not in pipeline_base:
            pipeline_base.extend(["--dists_as_distance", str(self.args.dists_as_distance)])

        # Ensure model_dir is forwarded to the pipeline if provided
        if self.args.model_dir and "--model_dir" not in pipeline_base:
            pipeline_base.extend(["--model_dir", self.args.model_dir])

        self.pipeline_base_args = pipeline_base

        self.eval_config = self._build_eval_config()
        self.param_grid = load_param_grid(self.args.param_config_json)

        os.makedirs(self.args.out_dir, exist_ok=True)

    def _parse_args(self):
        ap = argparse.ArgumentParser(description="Optimize keyframe selection pipeline hyperparameters.")
        
        # --- Core Optimizer Args ---
        ap.add_argument("--samples_dir", required=True, type=str, help="Directory containing all sample videos.")
        ap.add_argument("--pipeline_script", required=True, type=str, help="Path to the 'pipeline.py' script to be optimized.")
        ap.add_argument("--param_config_json", required=True, type=str, help="Path to JSON file defining the parameter grid for grid search.")
        ap.add_argument("--out_dir", required=True, type=str, help="Base directory to store all optimization runs and results.")
        ap.add_argument("--target_metric", type=str, default="Combined", help="Metric from eval results to optimize (e.g., 'Representativeness_mean' or 'Combined').")
        ap.add_argument("--max_videos", type=int, default=2, help="Maximum number of videos to evaluate per parameter set (default: 2).")
        ap.add_argument("--early_stopping_patience", type=int, default=10, help="Number of iterations without improvement before stopping (default: 5).")
        
        # --- Backend and Distance Metric ---
        ap.add_argument("--backend", type=str, default="transnetv2", help="Scene detection backend (default: transnetv2).")
        ap.add_argument("--model_dir", type=str, required=True, help="Path to model directory (e.g., src/models/TransNetV2).")
        ap.add_argument("--distance_backend", type=str, default="lpips", help="Distance metric backend (e.g., lpips or dists).")
        ap.add_argument("--lpips_net", type=str, default="alex", help="[lpips] Backbone: 'alex'|'vgg'|'squeeze'.")
        ap.add_argument("--dists_as_distance", type=int, default=1, help="[dists] Use raw DISTS as distance (1) or negate as similarity (0).")

        # --- EvalConfig Args ---
        ap.add_argument("--eval_backbone", type=str, required=True, help="Backbone model for evaluation (e.g., 'resnet50').")
        ap.add_argument("--eval_device", type=str, default="cuda", help="Device for evaluation (e.g., 'cuda' or 'cpu').")
        ap.add_argument("--eval_input_w", type=int, default=224, help="Input width for evaluation model.")
        ap.add_argument("--eval_input_h", type=int, default=224, help="Input height for evaluation model.")
        ap.add_argument("--eval_sample_stride", type=int, default=1, help="Stride for sampling frames during evaluation.")
        ap.add_argument("--eval_max_frames", type=int, default=100, help="Maximum number of frames to evaluate.")
        ap.add_argument("--eval_tau", type=float, default=0.3, help="Temporal consistency threshold for evaluation.")

        # --- Các tham số khác giữ nguyên ---
        
        known_args, unknown_args = ap.parse_known_args()
        return known_args, unknown_args

    def _build_eval_config(self) -> EvalConfig:
        """Tạo EvalConfig từ các tham số --eval_..."""
        return EvalConfig(
            backbone=self.args.eval_backbone,
            device=self.args.eval_device,
            input_size=(self.args.eval_input_w, self.args.eval_input_h),
            sample_stride=self.args.eval_sample_stride,
            max_frames_eval=self.args.eval_max_frames,
            tau_temporal=self.args.eval_tau,
        )

    def run(self):
        """Initializes and runs the optimizer."""
        optimizer = HyperparameterOptimizer(
            samples_dir=self.args.samples_dir,
            pipeline_script=self.args.pipeline_script,
            base_output_dir=self.args.out_dir,
            eval_config=self.eval_config,
            pipeline_base_args=self.pipeline_base_args,
            target_metric=self.args.target_metric,
            max_videos=self.args.max_videos,
            model_dir=self.args.model_dir,
            early_stopping_patience=self.args.early_stopping_patience
        )
        
        optimizer.run_grid_search(self.param_grid)


def main():
    app = OptimizerApp()
    app.run()


if __name__ == "__main__":
    main()

#example usage:
""" python outputs/outputs_eval/Evaluation_All.py \
  --eval_dir outputs_eval \
  --output_report quality_report.txt \
  --output_json detailed_results.json \
  --output_csv summary_results.csv """