#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Evaluation Pipeline
Processes all videos in a folder:
1. Extract keyframes using pipeline.py (LPIPS or DISTS)
2. Evaluate using eval_keyframes.py
3. Aggregate results into summary JSON
"""

import os
import json
import argparse
import subprocess
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
import traceback
import sys
import traceback

class BatchEvaluationPipeline:
    """
    Orchestrates batch video processing: keyframe extraction + evaluation.
    """
    
    def __init__(self, 
                 videos_dir: str,
                 output_base_dir: str,
                 distance_backend: str = "lpips",
                 debug: bool = False):
        """
        Args:
            videos_dir: Directory containing input videos
            output_base_dir: Base output directory for all results
            distance_backend: "lpips" or "dists"
            debug: Print debug info
        """
        self.videos_dir = videos_dir
        self.output_base_dir = output_base_dir
        self.distance_backend = distance_backend
        self.debug = debug
        
        # Create output directories
        self.pipeline_out_dir = os.path.join(output_base_dir, "pipeline_results")
        self.eval_out_dir = os.path.join(output_base_dir, "eval_results")
        os.makedirs(self.pipeline_out_dir, exist_ok=True)
        os.makedirs(self.eval_out_dir, exist_ok=True)
        
        # Track results
        self.results = {}
        self.errors = {}
    
    def find_videos(self) -> List[str]:
        """Find all video files in videos_dir."""
        extensions = ["*.mp4", "*.mkv", "*.avi", "*.mov"]
        videos = []
        for ext in extensions:
            videos.extend(glob.glob(os.path.join(self.videos_dir, ext)))
        return sorted(videos)
    
    def _run_keyframe_extraction(self, 
                                  video_path: str,
                                  output_dir: str,
                                  prob_threshold: float = 0.5,
                                  sample_stride: int = 5,
                                  max_frames_per_scene: int = 40,
                                  keyframes_per_scene: int = 1,
                                  nms_radius: int = 2,
                                  resize_w: int = 320,
                                  resize_h: int = 180) -> Optional[Dict[str, str]]:
        """
        Run pipeline.py to extract keyframes.
        Returns dict with paths to scenes.json and keyframes.csv, or None on error.
        """
        try:
            # Build command
            cmd = [
                "python", "pipeline.py",
                "--video", video_path,
                "--out_dir", output_dir,
                # "--backend", "transnetv2",
                # "--model_dir", "src/models/TransNetV2",
                # "--prob_threshold", str(prob_threshold),
                "--backend", "pyscenedetect",
                "--threshold", str(27),
                "--distance_backend", self.distance_backend,
                "--sample_stride", str(sample_stride),
                "--max_frames_per_scene", str(max_frames_per_scene),
                "--keyframes_per_scene", str(keyframes_per_scene),
                "--nms_radius", str(nms_radius),
                "--resize_w", str(resize_w),
                "--resize_h", str(resize_h),
            ]
            
            # Add distance-specific flags
            if self.distance_backend == "lpips":
                cmd.extend(["--lpips_net", "vgg"])
            elif self.distance_backend == "dists":
                cmd.extend(["--dists_as_distance", "1"])
            
            if self.debug:
                print(f"  [Debug] Running: {' '.join(cmd)}")
            
            # Run pipeline
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/home/serverai/ltdoanh/LayoutGeneration"
            )
            
            if result.returncode != 0:
                print(f"  [Error] Pipeline failed with code {result.returncode}")
                print(f"  [STDERR] {result.stderr}")
                return None
            
            if self.debug:
                print(f"  [Pipeline Output] {result.stdout}")
            
            # Find output directory - pipeline creates nested structure:
            # pipeline_results/VIDEO_ID/pipeline_VIDEOID_TIMESTAMP/
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Search recursively for keyframes.csv
            pipeline_base = os.path.dirname(output_dir)  # pipeline_results folder
            
            matching_dirs = []
            if os.path.exists(pipeline_base):
                for root, dirs, files in os.walk(pipeline_base):
                    if "keyframes.csv" in files and "scenes.json" in files:
                        # Check if this directory belongs to our video
                        if video_name in root or video_name in os.listdir(root)[0] if os.listdir(root) else False:
                            matching_dirs.append(root)
            
            if not matching_dirs:
                print(f"  [Error] Could not find pipeline output directory for video: {video_name}")
                print(f"  [Debug] Searched in: {pipeline_base}")
                if os.path.exists(pipeline_base):
                    print(f"  [Debug] Contents: {os.listdir(pipeline_base)[:5]}")
                return None
            
            # Use the most recently created directory (latest timestamp)
            actual_output_dir = max(matching_dirs, key=os.path.getctime)
            
            scenes_json = os.path.join(actual_output_dir, "scenes.json")
            keyframes_csv = os.path.join(actual_output_dir, "keyframes.csv")
            
            if not os.path.exists(scenes_json) or not os.path.exists(keyframes_csv):
                print(f"  [Error] Output files not found in {actual_output_dir}")
                return None
            
            return {
                "scenes_json": scenes_json,
                "keyframes_csv": keyframes_csv,
                "output_dir": actual_output_dir
            }
            
        except Exception as e:
            print(f"  [Error] Keyframe extraction failed: {e}")
            return None
    
    def _run_evaluation(self,
                       video_path: str,
                       scenes_json: str,
                       keyframes_csv: str,
                       eval_output_dir: str,
                       backbone: str = "resnet50",
                       device: str = "cuda",
                       sample_stride: int = 1,
                       max_frames_eval: int = 100,
                       tau: float = 0.5,
                       with_baselines: bool = True) -> Optional[Dict]:
        """
        Run eval_keyframes.py to evaluate keyframes.
        Returns metrics dict or None on error.
        """
        try:
            os.makedirs(eval_output_dir, exist_ok=True)
            
            cmd = [
                "python", "scripts/eval_keyframes.py",
                "--video", video_path,
                "--scenes_json", scenes_json,
                "--keyframes_csv", keyframes_csv,
                "--out_dir", eval_output_dir,
                "--backbone", backbone,
                "--device", device,
                "--sample_stride", str(sample_stride),
                "--max_frames_eval", str(max_frames_eval),
                "--tau", str(tau),
            ]
            
            if with_baselines:
                cmd.append("--with_baselines")
            
            if self.debug:
                print(f"  [Debug] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/home/serverai/ltdoanh/LayoutGeneration"
            )
            
            if result.returncode != 0:
                print(f"  [Error] Evaluation failed with code {result.returncode}")
                print(f"  [STDERR] {result.stderr}")
                return None
            
            if self.debug:
                print(f"  [Eval Output] {result.stdout}")
            
            # Load results
            results_json = os.path.join(eval_output_dir, "eval_results.json")
            if not os.path.exists(results_json):
                print(f"  [Error] eval_results.json not found in {eval_output_dir}")
                return None
            
            with open(results_json, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            return metrics
            
        except Exception as e:
            print(f"  [Error] Evaluation failed: {e}")
            return None
    
    def process_video(self,
                     video_path: str,
                     video_id: str,
                     prob_threshold: float = 0.5,
                     sample_stride: int = 5,
                     max_frames_per_scene: int = 40,
                     keyframes_per_scene: int = 1,
                     nms_radius: int = 2,
                     resize_w: int = 320,
                     resize_h: int = 180,
                     eval_backbone: str = "resnet50",
                     eval_device: str = "cuda",
                     eval_sample_stride: int = 1,
                     eval_max_frames: int = 200,
                     eval_tau: float = 0.5,
                     with_baselines: bool = True) -> bool:
        """
        Process single video: extract keyframes + evaluate.
        Returns True on success, False on failure.
        """
        print(f"\n{'='*80}")
        print(f"Processing: {video_id}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Keyframe extraction
            print(f"[1/3] Extracting keyframes...")
            pipeline_out = os.path.join(self.pipeline_out_dir, video_id, "pipeline")
            extraction_result = self._run_keyframe_extraction(
                video_path=video_path,
                output_dir=pipeline_out,
                prob_threshold=prob_threshold,
                sample_stride=sample_stride,
                max_frames_per_scene=max_frames_per_scene,
                keyframes_per_scene=keyframes_per_scene,
                nms_radius=nms_radius,
                resize_w=resize_w,
                resize_h=resize_h
            )
            
            if extraction_result is None:
                self.errors[video_id] = "Keyframe extraction failed"
                return False
            
            print(f"  ✅ Keyframes extracted: {extraction_result['keyframes_csv']}")
            
            # Step 2: Evaluation
            print(f"[2/3] Evaluating keyframes...")
            eval_out = os.path.join(self.eval_out_dir, video_id)
            metrics = self._run_evaluation(
                video_path=video_path,
                scenes_json=extraction_result['scenes_json'],
                keyframes_csv=extraction_result['keyframes_csv'],
                eval_output_dir=eval_out,
                backbone=eval_backbone,
                device=eval_device,
                sample_stride=eval_sample_stride,
                max_frames_eval=eval_max_frames,
                tau=eval_tau,
                with_baselines=with_baselines
            )
            
            if metrics is None:
                self.errors[video_id] = "Evaluation failed"
                return False
            
            print(f"  ✅ Evaluation complete")
            
            # Step 3: Store results
            print(f"[3/3] Storing results...")
            self.results[video_id] = {
                "video_path": video_path,
                "pipeline_output": extraction_result['output_dir'],
                "eval_output": eval_out,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"  ✅ Results stored")
            print(f"  📊 Key metrics:")
            
            # Safely format metrics
            rec_err = metrics.get('RecErr')
            rec_err_str = f"{rec_err:.4f}" if isinstance(rec_err, (int, float)) and not np.isnan(rec_err) else "N/A"
            
            scene_cov = metrics.get('SceneCoverage')
            scene_cov_str = f"{scene_cov:.4f}" if isinstance(scene_cov, (int, float)) and not np.isnan(scene_cov) else "N/A"
            
            temp_cov = metrics.get('TemporalCoverage@tau')
            temp_cov_str = f"{temp_cov:.4f}" if isinstance(temp_cov, (int, float)) and not np.isnan(temp_cov) else "N/A"
            
            print(f"     RecErr: {rec_err_str}")
            print(f"     SceneCoverage: {scene_cov_str}")
            print(f"     TemporalCoverage@tau: {temp_cov_str}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.errors[video_id] = str(e)
            return False
    
    def run_batch(self,
                 prob_threshold: float = 0.5,
                 sample_stride: int = 5,
                 max_frames_per_scene: int = 40,
                 keyframes_per_scene: int = 1,
                 nms_radius: int = 2,
                 resize_w: int = 320,
                 resize_h: int = 180,
                 eval_backbone: str = "resnet50",
                 eval_device: str = "cuda",
                 eval_sample_stride: int = 1,
                 eval_max_frames: int = 200,
                 eval_tau: float = 0.5,
                 with_baselines: bool = True,
                 max_videos: Optional[int] = None):
        """
        Process all videos in batch.
        """
        videos = self.find_videos()
        
        if not videos:
            print(f"❌ No videos found in {self.videos_dir}")
            return
        
        if max_videos:
            videos = videos[:max_videos]
        
        print(f"\n{'='*80}")
        print(f"BATCH EVALUATION PIPELINE")
        print(f"{'='*80}")
        print(f"Videos directory: {self.videos_dir}")
        print(f"Output directory: {self.output_base_dir}")
        print(f"Distance backend: {self.distance_backend}")
        print(f"Videos to process: {len(videos)}")
        print(f"{'='*80}\n")
        
        successful = 0
        failed = 0
        
        for i, video_path in enumerate(videos, 1):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            print(f"[{i}/{len(videos)}] ", end="")
            
            success = self.process_video(
                video_path=video_path,
                video_id=video_id,
                prob_threshold=prob_threshold,
                sample_stride=sample_stride,
                max_frames_per_scene=max_frames_per_scene,
                keyframes_per_scene=keyframes_per_scene,
                nms_radius=nms_radius,
                resize_w=resize_w,
                resize_h=resize_h,
                eval_backbone=eval_backbone,
                eval_device=eval_device,
                eval_sample_stride=eval_sample_stride,
                eval_max_frames=eval_max_frames,
                eval_tau=eval_tau,
                with_baselines=with_baselines
            )
            
            if success:
                successful += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successful: {successful}/{len(videos)}")
        print(f"❌ Failed: {failed}/{len(videos)}")
        
        if self.errors:
            print(f"\nErrors:")
            for video_id, error in self.errors.items():
                print(f"  - {video_id}: {error}")
        
        # Save summary
        self._save_summary()
    
    def run_batch_parallel(self,
                          prob_threshold: float = 0.5,
                          sample_stride: int = 5,
                          max_frames_per_scene: int = 40,
                          keyframes_per_scene: int = 1,
                          nms_radius: int = 2,
                          resize_w: int = 320,
                          resize_h: int = 180,
                          eval_backbone: str = "resnet50",
                          eval_device: str = "cuda",
                          eval_sample_stride: int = 1,
                          eval_max_frames: int = 200,
                          eval_tau: float = 0.5,
                          with_baselines: bool = True,
                          max_videos: Optional[int] = None,
                          num_workers: Optional[int] = None):
        """
        Process all videos in batch using parallel workers.
        """
        videos = self.find_videos()
        
        if not videos:
            print(f"❌ No videos found in {self.videos_dir}")
            return
        
        if max_videos:
            videos = videos[:max_videos]
        
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        print(f"\n{'='*80}")
        print(f"BATCH EVALUATION PIPELINE (PARALLEL)")
        print(f"{'='*80}")
        print(f"Videos directory: {self.videos_dir}")
        print(f"Output directory: {self.output_base_dir}")
        print(f"Distance backend: {self.distance_backend}")
        print(f"Videos to process: {len(videos)}")
        print(f"Workers: {num_workers}")
        print(f"{'='*80}\n")
        
        # Prepare task arguments
        task_args = [
            (
                video_path,
                os.path.splitext(os.path.basename(video_path))[0],
                prob_threshold,
                sample_stride,
                max_frames_per_scene,
                keyframes_per_scene,
                nms_radius,
                resize_w,
                resize_h,
                eval_backbone,
                eval_device,
                eval_sample_stride,
                eval_max_frames,
                eval_tau,
                with_baselines
            )
            for video_path in videos
        ]
        
        # Process videos in parallel
        successful = 0
        failed = 0
        
        with Pool(num_workers) as pool:
            results = pool.starmap(self._process_video_worker, task_args)
        
        # Aggregate results
        for i, (video_id, success, result_data, error) in enumerate(results, 1):
            print(f"[{i}/{len(videos)}] {video_id}: ", end="")
            
            if success:
                self.results[video_id] = result_data
                successful += 1
                print("✅")
            else:
                self.errors[video_id] = error
                failed += 1
                print(f"❌ {error}")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successful: {successful}/{len(videos)}")
        print(f"❌ Failed: {failed}/{len(videos)}")
        
        if self.errors:
            print(f"\nErrors:")
            for video_id, error in self.errors.items():
                print(f"  - {video_id}: {error}")
        
        # Save summary
        self._save_summary()
    
    def _process_video_worker(self,
                             video_path: str,
                             video_id: str,
                             prob_threshold: float,
                             sample_stride: int,
                             max_frames_per_scene: int,
                             keyframes_per_scene: int,
                             nms_radius: int,
                             resize_w: int,
                             resize_h: int,
                             eval_backbone: str,
                             eval_device: str,
                             eval_sample_stride: int,
                             eval_max_frames: int,
                             eval_tau: float,
                             with_baselines: bool) -> Tuple[str, bool, Dict, str]:
        """
        Worker function for parallel processing.
        Returns (video_id, success, result_data, error_message)
        """
        try:
            # Step 1: Keyframe extraction
            pipeline_out = os.path.join(self.pipeline_out_dir, video_id, "pipeline")
            extraction_result = self._run_keyframe_extraction(
                video_path=video_path,
                output_dir=pipeline_out,
                prob_threshold=prob_threshold,
                sample_stride=sample_stride,
                max_frames_per_scene=max_frames_per_scene,
                keyframes_per_scene=keyframes_per_scene,
                nms_radius=nms_radius,
                resize_w=resize_w,
                resize_h=resize_h
            )
            
            if extraction_result is None:
                return (video_id, False, {}, "Keyframe extraction failed")
            
            # Step 2: Evaluation
            eval_out = os.path.join(self.eval_out_dir, video_id)
            metrics = self._run_evaluation(
                video_path=video_path,
                scenes_json=extraction_result['scenes_json'],
                keyframes_csv=extraction_result['keyframes_csv'],
                eval_output_dir=eval_out,
                backbone=eval_backbone,
                device=eval_device,
                sample_stride=eval_sample_stride,
                max_frames_eval=eval_max_frames,
                tau=eval_tau,
                with_baselines=with_baselines
            )
            
            if metrics is None:
                return (video_id, False, {}, "Evaluation failed")
            
            # Step 3: Store results
            result_data = {
                "video_path": video_path,
                "pipeline_output": extraction_result['output_dir'],
                "eval_output": eval_out,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            return (video_id, True, result_data, "")
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return (video_id, False, {}, error_msg)
    
    def _save_summary(self):
        """Save summary JSON with all results."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "videos_dir": self.videos_dir,
                "output_base_dir": self.output_base_dir,
                "distance_backend": self.distance_backend,
            },
            "statistics": {
                "total_processed": len(self.results),
                "total_failed": len(self.errors),
            },
            "results": self.results,
            "errors": self.errors
        }
        
        # Calculate aggregate metrics
        if self.results:
            metrics_list = []
            for video_id, result in self.results.items():
                metrics = result['metrics']
                metrics_list.append({
                    'video_id': video_id,
                    'RecErr': metrics.get('RecErr'),
                    'Frechet': metrics.get('Frechet'),
                    'SceneCoverage': metrics.get('SceneCoverage'),
                    'TemporalCoverage@tau': metrics.get('TemporalCoverage@tau'),
                    'RedundancyMeanCos': metrics.get('RedundancyMeanCos'),
                    'MinPairwiseDist': metrics.get('MinPairwiseDist'),
                    'Sharpness_med': metrics.get('Sharpness_med'),
                    'Exposure_med': metrics.get('Exposure_med'),
                    'Noise_med': metrics.get('Noise_med'),
                    'NumKeys': metrics.get('NumKeys'),
                    'NumAllEmbed': metrics.get('NumAllEmbed'),
                })
            
            # Compute means (filtering NaN)
            def safe_mean(values):
                valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
                return float(np.mean(valid)) if valid else None
            
            summary['aggregate_metrics'] = {
                'RecErr_mean': safe_mean([m['RecErr'] for m in metrics_list]),
                'Frechet_mean': safe_mean([m['Frechet'] for m in metrics_list]),
                'SceneCoverage_mean': safe_mean([m['SceneCoverage'] for m in metrics_list]),
                'TemporalCoverage@tau_mean': safe_mean([m['TemporalCoverage@tau'] for m in metrics_list]),
                'RedundancyMeanCos_mean': safe_mean([m['RedundancyMeanCos'] for m in metrics_list]),
                'MinPairwiseDist_mean': safe_mean([m['MinPairwiseDist'] for m in metrics_list]),
                'Sharpness_med_mean': safe_mean([m['Sharpness_med'] for m in metrics_list]),
                'Exposure_med_mean': safe_mean([m['Exposure_med'] for m in metrics_list]),
                'Noise_med_mean': safe_mean([m['Noise_med'] for m in metrics_list]),
            }
            
            summary['per_video_metrics'] = metrics_list
        
        output_path = os.path.join(self.output_base_dir, "summary_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation pipeline for keyframe extraction.")
    
    # Input/Output
    parser.add_argument("--videos_dir", required=True, type=str, 
                       help="Directory containing input videos")
    parser.add_argument("--output_dir", required=True, type=str,
                       help="Base output directory for results")
    
    # Pipeline parameters
    parser.add_argument("--distance_backend", type=str, default="lpips", 
                       choices=["lpips", "dists"],
                       help="Distance metric: lpips or dists")
    parser.add_argument("--prob_threshold", type=float, default=0.5,
                       help="[TransNetV2] Probability threshold")
    parser.add_argument("--sample_stride", type=int, default=5,
                       help="Sample stride for keyframe selection")
    parser.add_argument("--max_frames_per_scene", type=int, default=40,
                       help="Maximum frames per scene")
    parser.add_argument("--keyframes_per_scene", type=int, default=1,
                       help="Number of keyframes per scene")
    parser.add_argument("--nms_radius", type=int, default=2,
                       help="NMS radius")
    parser.add_argument("--resize_w", type=int, default=320,
                       help="Resize width")
    parser.add_argument("--resize_h", type=int, default=180,
                       help="Resize height")
    
    # Evaluation parameters
    parser.add_argument("--eval_backbone", type=str, default="resnet50",
                       help="Backbone for feature extraction")
    parser.add_argument("--eval_device", type=str, default="cuda",
                       help="Device for evaluation (cuda/cpu)")
    parser.add_argument("--eval_sample_stride", type=int, default=1,
                       help="Sample stride for evaluation")
    parser.add_argument("--eval_max_frames", type=int, default=200,
                       help="Maximum frames to evaluate")
    parser.add_argument("--eval_tau", type=float, default=0.5,
                       help="Temporal coverage threshold")
    parser.add_argument("--with_baselines", action="store_true",
                       help="Evaluate baselines (uniform, mid, motion)")
    
    # Batch parameters
    parser.add_argument("--max_videos", type=int, default=None,
                       help="Maximum number of videos to process")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count - 1). Set to 1 for sequential mode")
    parser.add_argument("--debug", action="store_true",
                       help="Print debug information")
    
    args = parser.parse_args()
    
    # Create pipeline and run
    pipeline = BatchEvaluationPipeline(
        videos_dir=args.videos_dir,
        output_base_dir=args.output_dir,
        distance_backend=args.distance_backend,
        debug=args.debug
    )
    
    # Choose between sequential and parallel based on num_workers
    if args.num_workers is not None and args.num_workers != 1:
        # Run parallel
        pipeline.run_batch_parallel(
            prob_threshold=args.prob_threshold,
            sample_stride=args.sample_stride,
            max_frames_per_scene=args.max_frames_per_scene,
            keyframes_per_scene=args.keyframes_per_scene,
            nms_radius=args.nms_radius,
            resize_w=args.resize_w,
            resize_h=args.resize_h,
            eval_backbone=args.eval_backbone,
            eval_device=args.eval_device,
            eval_sample_stride=args.eval_sample_stride,
            eval_max_frames=args.eval_max_frames,
            eval_tau=args.eval_tau,
            with_baselines=args.with_baselines,
            max_videos=args.max_videos,
            num_workers=args.num_workers
        )
    else:
        # Run sequential
        pipeline.run_batch(
            prob_threshold=args.prob_threshold,
            sample_stride=args.sample_stride,
            max_frames_per_scene=args.max_frames_per_scene,
            keyframes_per_scene=args.keyframes_per_scene,
            nms_radius=args.nms_radius,
            resize_w=args.resize_w,
            resize_h=args.resize_h,
            eval_backbone=args.eval_backbone,
            eval_device=args.eval_device,
            eval_sample_stride=args.eval_sample_stride,
            eval_max_frames=args.eval_max_frames,
            eval_tau=args.eval_tau,
            with_baselines=args.with_baselines,
            max_videos=args.max_videos
        )


if __name__ == "__main__":
    main()
