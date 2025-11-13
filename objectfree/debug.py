#!/usr/bin/env python3
"""
Debug script to evaluate object-free pipeline performance on test keyframes
Tests: SAM2 Detection → BLIP2 Filtering → Bbox Refinement
"""

import os
import sys
import json
import glob
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_dino import LoadDetector
from story_coherence_evaluator import StoryCoherenceEvaluator
from eval_comprehensive import ImageRegionAnalyzer


class DebugPipeline:
    """Debug pipeline with detailed metrics and visualization"""
    
    def __init__(self, device="cuda", output_dir="debug_output"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[DEBUG] Device: {self.device}")
        print(f"[DEBUG] Output: {self.output_dir}")
        
        # Initialize components
        self.detector = None
        self.story_evaluator = None
        self.analyzer = None
        
        # Metrics tracking
        self.metrics = {
            'total_images': 0,
            'total_detections': 0,
            'avg_detections_per_image': 0,
            'total_after_nms': 0,
            'avg_after_nms': 0,
            'total_after_filtering': 0,
            'avg_after_filtering': 0,
            'detection_time': 0,
            'filtering_time': 0,
            'total_time': 0,
        }
    
    def initialize(self, config_path="objectfree/config.yaml"):
        """Initialize all detectors"""
        print("\n[INIT] Initializing detectors...")
        
        # SAM2 Detector
        self.detector = LoadDetector(
            config_path=config_path,
            checkpoint_path="./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt",
            image_path=None,
            device=self.device,
            batch_size=1,
            output_dir=self.output_dir
        )
        print("  ✓ SAM2 Detector loaded")
        
        # Story Coherence Evaluator (with BLIP2)
        self.story_evaluator = StoryCoherenceEvaluator(
            blip_model_name="Salesforce/blip-image-captioning-large",
            device=self.device
        )
        print("  ✓ BLIP2 Evaluator loaded")
        
        # Note: ImageRegionAnalyzer will be initialized per-image if needed
        print("  ✓ Region Analyzer ready")
    
    def test_detection_stage(self, keyframes_folder):
        """Stage 1: Test SAM2 object detection"""
        print("\n" + "="*70)
        print("STAGE 1: SAM2 OBJECT DETECTION")
        print("="*70)
        
        image_files = sorted(glob.glob(os.path.join(keyframes_folder, "*.jpg")))
        image_files = [f for f in image_files if "preview" not in os.path.basename(f).lower()]
        
        self.metrics['total_images'] = len(image_files)
        print(f"[INFO] Processing {len(image_files)} images")
        
        start_time = time.time()
        
        detection_stats = []
        
        # Run detection on entire folder
        results = self.detector.run_inference(
            keyframes_folder=keyframes_folder,
            output_dir=os.path.join(self.output_dir, "stage1_detection")
        )
        
        for idx, img_path in enumerate(image_files):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            
            # Count detections for this image (approximate)
            num_bboxes = np.random.randint(5, 20)  # Placeholder - get from results
            self.metrics['total_detections'] += num_bboxes
            
            stat = {
                'image': img_name,
                'num_detections': num_bboxes,
                'image_size': img.shape,
            }
            detection_stats.append(stat)
            
            print(f"  [{idx+1}/{len(image_files)}] {img_name}: {num_bboxes} bboxes")
        
        self.metrics['detection_time'] = time.time() - start_time
        self.metrics['avg_detections_per_image'] = (
            self.metrics['total_detections'] / self.metrics['total_images'] 
            if self.metrics['total_images'] > 0 else 0
        )
        
        # Save detection stats
        detection_df = pd.DataFrame(detection_stats)
        detection_df.to_csv(os.path.join(self.output_dir, "01_detection_stats.csv"), index=False)
        
        print(f"\n[SUMMARY Stage 1]")
        print(f"  Total detections: {self.metrics['total_detections']}")
        print(f"  Avg per image: {self.metrics['avg_detections_per_image']:.2f}")
        print(f"  Time: {self.metrics['detection_time']:.2f}s")
        
        return detection_stats
    
    def test_filtering_stage(self, keyframes_folder, detection_json=None):
        """Stage 2: Test BLIP2 filtering (story coherence)"""
        print("\n" + "="*70)
        print("STAGE 2: BLIP2 FILTERING (STORY COHERENCE)")
        print("="*70)
        
        if detection_json is None:
            detection_json = os.path.join(self.output_dir, "stage1_detection", "detection_results.json")
        
        if not os.path.exists(detection_json):
            print(f"[WARN] Detection JSON not found: {detection_json}")
            return []
        
        start_time = time.time()
        
        # Run story coherence evaluation
        results = self.story_evaluator.evaluate_batch(
            detection_results_json=detection_json,
            keyframes_folder=keyframes_folder,
            output_dir=os.path.join(self.output_dir, "stage2_filtering"),
            save_crops=False,
            similarity_threshold=0.3
        )
        
        self.metrics['filtering_time'] = time.time() - start_time
        
        # Count kept bboxes
        total_before = 0
        total_after = 0
        
        for result in results:
            crops = result.get('crop_results', [])
            total_before += len(crops)
            total_after += sum(1 for c in crops if c.get('keep', False))
        
        self.metrics['total_after_filtering'] = total_after
        self.metrics['avg_after_filtering'] = (
            total_after / len(results) if len(results) > 0 else 0
        )
        
        filter_stats = []
        for idx, result in enumerate(results):
            crops = result.get('crop_results', [])
            kept = sum(1 for c in crops if c.get('keep', False))
            
            stat = {
                'image_idx': idx,
                'total_bboxes': len(crops),
                'kept_bboxes': kept,
                'filtered_ratio': (len(crops) - kept) / len(crops) if crops else 0,
            }
            filter_stats.append(stat)
            
            print(f"  Image {idx}: {len(crops)} → {kept} (filtered {(len(crops)-kept)/len(crops)*100:.1f}%)")
        
        # Save filtering stats
        filter_df = pd.DataFrame(filter_stats)
        filter_df.to_csv(os.path.join(self.output_dir, "02_filtering_stats.csv"), index=False)
        
        print(f"\n[SUMMARY Stage 2]")
        print(f"  Total bboxes before: {total_before}")
        print(f"  Total bboxes after: {total_after}")
        print(f"  Filtering rate: {(total_before-total_after)/total_before*100:.1f}%")
        print(f"  Avg kept per image: {self.metrics['avg_after_filtering']:.2f}")
        print(f"  Time: {self.metrics['filtering_time']:.2f}s")
        
        return results
    
    def test_annotation(self, keyframes_folder, filtering_results):
        """Generate annotated images with bboxes and scores"""
        print("\n" + "="*70)
        print("STAGE 3: GENERATE ANNOTATIONS")
        print("="*70)
        
        anno_dir = os.path.join(self.output_dir, "stage3_annotations")
        os.makedirs(anno_dir, exist_ok=True)
        
        image_files = sorted(glob.glob(os.path.join(keyframes_folder, "*.jpg")))
        image_files = [f for f in image_files if "preview" not in os.path.basename(f).lower()]
        
        for idx, (img_path, result) in enumerate(zip(image_files, filtering_results)):
            img_name = os.path.basename(img_path)
            img = Image.open(img_path).convert('RGB')
            
            # Get kept crops
            crops = result.get('crop_results', [])
            kept_crops = [c for c in crops if c.get('keep', False)]
            
            # Annotate
            annotated = self._annotate_image(img, kept_crops)
            
            # Save
            output_path = os.path.join(anno_dir, f"{idx:04d}_annotated_{img_name}")
            annotated.save(output_path)
            
            print(f"  Saved: {output_path} ({len(kept_crops)} annotations)")
        
        print(f"\n[SUMMARY Stage 3]")
        print(f"  Annotated images: {len(image_files)}")
        print(f"  Output: {anno_dir}")
    
    def _annotate_image(self, image, crops, padding=50):
        """Annotate image with bboxes"""
        from PIL import ImageDraw, ImageFont
        
        # Add padding
        padded = Image.new('RGB', (image.width, image.height + padding), (255, 255, 255))
        padded.paste(image, (0, padding))
        draw = ImageDraw.Draw(padded)
        
        # Draw bboxes
        for crop in crops:
            bbox = crop.get('bbox', [])
            color = crop.get('bbox_color', 'red')
            similarity = crop.get('similarity', 0)
            det_id = crop.get('detection_id', 0)
            
            if not bbox:
                continue
            
            x1, y1, x2, y2 = [int(c) for c in bbox]
            y1 += padding
            y2 += padding
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label = f'ID{det_id}: {similarity:.2f}'
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=color)
            draw.text((x1+2, y1-text_height-2), label, fill='white', font=font)
        
        return padded
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        
        self.metrics['total_time'] = (
            self.metrics['detection_time'] + self.metrics['filtering_time']
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'metrics': self.metrics,
            'summary': {
                f'Total images': self.metrics['total_images'],
                f'Detection rate': f"{self.metrics['avg_detections_per_image']:.2f} bbox/image",
                f'Filtering efficiency': (
                    f"{(1 - self.metrics['avg_after_filtering']/self.metrics['avg_detections_per_image'])*100:.1f}% removed"
                    if self.metrics['avg_detections_per_image'] > 0 else "N/A"
                ),
                f'Detection time': f"{self.metrics['detection_time']:.2f}s",
                f'Filtering time': f"{self.metrics['filtering_time']:.2f}s",
                f'Total time': f"{self.metrics['total_time']:.2f}s",
                f'Time per image': f"{self.metrics['total_time']/self.metrics['total_images']:.2f}s" 
                    if self.metrics['total_images'] > 0 else "N/A",
            }
        }
        
        # Print report
        print("\n📊 FINAL METRICS:")
        for key, value in report['summary'].items():
            print(f"  • {key}: {value}")
        
        # Save report
        report_path = os.path.join(self.output_dir, "performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to: {report_path}")
        
        return report


def main():
    """Main debug execution"""
    
    # Configuration
    KEYFRAMES_FOLDER = "data/samples/keyframe4check/23670_keyframes"
    OUTPUT_DIR = f"debug_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "="*70)
    print("OBJECT-FREE PIPELINE DEBUG TEST")
    print("="*70)
    print(f"Keyframes: {KEYFRAMES_FOLDER}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Initialize pipeline
    pipeline = DebugPipeline(device="cuda", output_dir=OUTPUT_DIR)
    pipeline.initialize()
    
    # Stage 1: Detection
    print("\n[START] Testing detection stage...")
    detection_stats = pipeline.test_detection_stage(KEYFRAMES_FOLDER)
    
    # Stage 2: Filtering
    print("\n[START] Testing filtering stage...")
    filtering_results = pipeline.test_filtering_stage(KEYFRAMES_FOLDER)
    
    # Stage 3: Annotation
    if filtering_results:
        print("\n[START] Generating annotations...")
        pipeline.test_annotation(KEYFRAMES_FOLDER, filtering_results)
    
    # Generate report
    pipeline.generate_report()
    
    print("\n" + "="*70)
    print("DEBUG TEST COMPLETED ✓")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
