#!/usr/bin/env python3
"""
Complete 3-Stage Advanced BBox Processing Pipeline
1. SAM2 Detection → 2. BLIP2 Caption Filtering → 3. Advanced BBox Refinement
   - KDE-based thresholding
   - Distance weighting (gaussian decay)
   - HSV-normalized color variance
   - Selective percentile (not taking all details)
"""

import cv2
import numpy as np
import torch
import glob
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Import custom modules
from inference_dino import LoadDetector
from story_coherence_evaluator import StoryCoherenceEvaluator
from bbox_refinement import BBoxRefinement


class AdvancedBBoxPipeline:
    """
    Complete pipeline: Detection → Caption Filtering → Advanced Refinement
    """
    
    def __init__(self, device="cuda", output_dir="pipeline_output"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"[INIT] Device: {self.device}")
        print(f"[INIT] Output: {self.output_dir}")
        
        # Stage 1: SAM2 Detector
        self.detector = None
        
        # Stage 2: BLIP2 Filter
        self.story_evaluator = None
        
        # Stage 3: Advanced BBox Refinement
        self.refiner = BBoxRefinement(device=self.device)
    
    def initialize(self, config_path="objectfree/config.yaml"):
        """Initialize all components"""
        print("\n" + "="*70)
        print("INITIALIZING ADVANCED PIPELINE COMPONENTS")
        print("="*70)
        
        # Initialize SAM2 Detector
        print("\n[STAGE 1] Initializing SAM2 Detector...")
        self.detector = LoadDetector(
            config_path=config_path,
            checkpoint_path="./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt",
            image_path=None,
            device=self.device,
            batch_size=1,
            output_dir=self.output_dir
        )
        print("  ✓ SAM2 Detector ready")
        
        # Initialize BLIP2 Filter
        print("\n[STAGE 2] Initializing BLIP2 Caption Filter...")
        self.story_evaluator = StoryCoherenceEvaluator(
            blip_model_name="Salesforce/blip-image-captioning-large",
            device=self.device
        )
        print("  ✓ BLIP2 Filter ready")
        
        # Stage 3 is already initialized
        print("\n[STAGE 3] Advanced BBox Refinement ready")
        
        print("\n✅ All components initialized!")
    
    def process_image(self, image_path, config=None):
        """Process single image through all 3 stages"""
        
        if config is None:
            config = {
                'caption_similarity_threshold': 0.3,
                'keep_top_k_bboxes': 3,
                'refinement_use_kde': True,
                'refinement_distance_weight': True,
                'refinement_percentile': 75,
                'search_margin': 0.25,
                'expansion_margin': 0.15,
            }
        
        image_name = Path(image_path).stem
        print(f"\n{'='*70}")
        print(f"PROCESSING: {image_name}")
        print(f"{'='*70}")
        
        image = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        result = {
            'image_name': image_name,
            'image_path': image_path,
            'image_size': [w, h],
            'timestamp': datetime.now().isoformat(),
        }
        
        # ===== STAGE 1: SAM2 DETECTION =====
        print(f"\n[STAGE 1/3] SAM2 Detection...")
        stage1_dir = Path(self.output_dir) / "stage1_detection" / image_name
        stage1_dir.mkdir(parents=True, exist_ok=True)
        
        detections = self._run_sam2_detection(image_path, stage1_dir)
        
        result['stage1'] = {
            'total_detections': len(detections),
            'bboxes': detections,
            'detection_file': str(stage1_dir / "detections.json"),
        }
        
        print(f"  ✓ Detected {len(detections)} bboxes")
        
        # ===== STAGE 2: BLIP2 CAPTION FILTERING =====
        print(f"\n[STAGE 2/3] BLIP2 Caption Filtering...")
        stage2_dir = Path(self.output_dir) / "stage2_filtering" / image_name
        stage2_dir.mkdir(parents=True, exist_ok=True)
        
        filtered_bboxes = self._filter_by_caption(
            image,
            detections,
            stage2_dir,
            similarity_threshold=config['caption_similarity_threshold'],
            keep_top_k=config['keep_top_k_bboxes']
        )
        
        filtering_ratio = (len(detections) - len(filtered_bboxes)) / len(detections) * 100 if detections else 0
        
        result['stage2'] = {
            'original_count': len(detections),
            'filtered_count': len(filtered_bboxes),
            'filtering_ratio': filtering_ratio,
            'bboxes': filtered_bboxes,
        }
        
        print(f"  ✓ Filtered to {len(filtered_bboxes)} bboxes ({filtering_ratio:.1f}% removed)")
        
        # ===== STAGE 3: ADVANCED BBOX REFINEMENT =====
        print(f"\n[STAGE 3/3] Advanced BBox Refinement...")
        stage3_dir = Path(self.output_dir) / "stage3_refinement" / image_name
        stage3_dir.mkdir(parents=True, exist_ok=True)
        
        refined_bboxes = self._refine_bboxes(
            image,
            filtered_bboxes,
            stage3_dir,
            use_kde=config['refinement_use_kde'],
            distance_weight=config['refinement_distance_weight'],
            percentile_threshold=config['refinement_percentile'],
            search_margin=config['search_margin'],
            expansion_margin=config['expansion_margin'],
            use_mean_shift=config['refinement_use_mean_shift'],
        )
        
        result['stage3'] = {
            'refined_bboxes': refined_bboxes,
            'average_expansion': np.mean([r['expansion_ratio'] for r in refined_bboxes]) if refined_bboxes else 1.0,
            'refinement_dir': str(stage3_dir),
        }
        
        avg_expansion = result['stage3']['average_expansion']
        print(f"  ✓ Refined {len(refined_bboxes)} bboxes (avg {avg_expansion:.2f}x expansion)")
        
        # Save results
        self._save_results(result, image, img_rgb)
        
        return result
    
    def _run_sam2_detection(self, image_path, output_dir):
        """Stage 1: Run SAM2 detection - use real LoadDetector"""
        # Use real LoadDetector instead of fake test data
        detector_output_dir = output_dir / "detector_output"
        detector_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create detector with image path as list
        detector = LoadDetector(
            config_path="config.yaml",
            checkpoint_path="./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt",
            image_path=[image_path],  # Must be iterable
            device=self.device,
            batch_size=1,
            output_dir=str(detector_output_dir)
        )
        
        # Run detection
        results_list = detector()  # Returns list of results per image
        results = results_list[0] if results_list else {"detections": []}
        
        # Extract detections from LoadDetector format
        detections = []
        if "detections" in results:
            for idx, det in enumerate(results["detections"]):
                detections.append({
                    'bbox_id': idx,
                    'bbox': [float(x) for x in det['bbox']],  # Convert to list
                    'confidence': float(det.get('confidence', 0.5)),
                    'score': float(det.get('score', 0.5)),
                })
        
        detection_file = output_dir / "detections.json"
        detection_file.parent.mkdir(parents=True, exist_ok=True)
        with open(detection_file, 'w') as f:
            json.dump(detections, f, indent=2)
        
        return detections
    
    def _filter_by_caption(self, image, detections, output_dir, 
                          similarity_threshold=0.3, keep_top_k=None):
        """Stage 2: Filter bboxes using BLIP2 captions"""
        filtered = []
        filter_results = []
        
        # Get full image caption
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        full_caption = self.story_evaluator.generate_caption(img_pil)
        
        for det in tqdm(detections, desc="Filtering by caption"):
            bbox = det['bbox']
            bbox_id = det['bbox_id']
            
            x1, y1, x2, y2 = [int(c) for c in bbox]
            crop = image[y1:y2, x1:x2]
            
            # Get BLIP2 caption for crop
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_caption = self.story_evaluator.generate_caption(crop_pil)
            
            # Calculate similarity
            similarity = self.story_evaluator.compute_similarity(
                full_caption, crop_caption
            )
            
            filter_result = {
                'bbox_id': bbox_id,
                'bbox': bbox,
                'crop_caption': crop_caption,
                'full_caption': full_caption,
                'similarity': float(similarity),
                'keep': similarity > similarity_threshold,
            }
            filter_results.append(filter_result)
            
            if filter_result['keep']:
                filtered.append({
                    **det,
                    'crop_caption': crop_caption,
                    'similarity': similarity,
                })
        
        # Save filtering results
        filter_file = output_dir / "filter_results.json"
        with open(filter_file, 'w') as f:
            json.dump(filter_results, f, indent=2)
        
        # Keep top-k if specified
        if keep_top_k is not None and len(filtered) > keep_top_k:
            filtered = sorted(filtered, key=lambda x: x['similarity'], reverse=True)[:keep_top_k]
        
        return filtered
    
    def _refine_bboxes(self, image, bboxes, output_dir, 
                      use_kde=True, distance_weight=True, percentile_threshold=75,
                      search_margin=0.2, expansion_margin=0.1, use_mean_shift=True):
        """
        Stage 3: Advanced BBox refinement with Mean Shift
        - KDE/Percentile thresholding
        - Distance weighting
        - Mean shift clustering (direction-aware expansion)
        """
        refined_bboxes = []
        
        for bbox_data in tqdm(bboxes, desc="Refining bboxes"):
            bbox = bbox_data['bbox']
            bbox_id = bbox_data['bbox_id']
            
            # Run advanced refinement with mean shift
            refined_bbox, heatmap, info = self.refiner.refine_bbox(
                image, bbox,
                weights=(0.4, 0.4, 0.2),
                use_kde=use_kde,
                distance_weight=distance_weight,
                percentile_threshold=percentile_threshold,
                search_margin=search_margin,
                expansion_margin=expansion_margin,
                use_mean_shift=use_mean_shift,  # ← NEW
            )
            
            refined_data = {
                'bbox_id': int(bbox_id),
                'original_bbox': [int(c) for c in bbox],
                'refined_bbox': [int(c) for c in refined_bbox],
                'expansion_ratio': float(info['expansion_ratio']),
                'crop_caption': bbox_data.get('crop_caption', ''),
                'similarity': float(bbox_data.get('similarity', 0)),
                'kde_threshold': float(info['kde_threshold']),
                'important_pixels': int(info['important_pixel_count']),
                'mean_shift_used': info.get('mean_shift_used', False),  # ← NEW
                'direction_adjustments': info.get('direction_adjustments', {}),  # ← NEW
            }
            refined_bboxes.append(refined_data)
            
            # Save heatmap
            heatmap_file = output_dir / f"bbox{bbox_id}_heatmap.png"
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            cv2.imwrite(str(heatmap_file), heatmap_colored)
        
        # Save refinement results
        refine_file = output_dir / "refinement_results.json"
        with open(refine_file, 'w') as f:
            json.dump(refined_bboxes, f, indent=2)
        
        return refined_bboxes
    
    def _save_results(self, result, image, img_rgb):
        """Save comprehensive results"""
        
        # Save result JSON
        result_file = Path(self.output_dir) / f"result_{result['image_name']}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Create comparison visualization
        self._create_comparison_viz(result, img_rgb)
    
    def _create_comparison_viz(self, result, img_rgb):
        """Create before/after visualization"""
        
        image_name = result['image_name']
        
        stage1_bboxes = [d['bbox'] for d in result['stage1']['bboxes']]
        stage2_bboxes = [d['bbox'] for d in result['stage2']['bboxes']]
        stage3_bboxes = [d['refined_bbox'] for d in result['stage3']['refined_bboxes']]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Stage 1
        img1 = self._draw_bboxes(img_rgb, stage1_bboxes)
        axes[0].imshow(img1)
        axes[0].set_title(f"Stage 1: SAM2 Detection\n{len(stage1_bboxes)} bboxes", 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Stage 2
        img2 = self._draw_bboxes(img_rgb, stage2_bboxes, color=(0, 255, 0))
        axes[1].imshow(img2)
        axes[1].set_title(f"Stage 2: After BLIP2 Filter\n{len(stage2_bboxes)} bboxes ({result['stage2']['filtering_ratio']:.0f}% removed)", 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Stage 3
        img3 = self._draw_bboxes(img_rgb, stage3_bboxes, color=(0, 0, 255))
        axes[2].imshow(img3)
        axes[2].set_title(f"Stage 3: After Advanced Refinement\n{len(stage3_bboxes)} bboxes ({result['stage3']['average_expansion']:.2f}x)", 
                         fontsize=14, fontweight='bold', color='red')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        viz_file = Path(self.output_dir) / f"pipeline_comparison_{image_name}.png"
        plt.savefig(viz_file, dpi=100, bbox_inches='tight')
        print(f"\n  💾 Saved comparison: {viz_file}")
        plt.close()
    
    def _draw_bboxes(self, image, bboxes, color=(255, 0, 0)):
        """Draw bboxes on image"""
        img_copy = image.copy()
        
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        return img_copy
    
    def generate_report(self):
        """Generate final report"""
        
        print("\n" + "="*70)
        print("FINAL REPORT - ADVANCED PIPELINE")
        print("="*70)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'output_directory': self.output_dir,
            'pipeline_type': 'Advanced (KDE + Distance Weighting + HSV)',
        }
        
        report_file = Path(self.output_dir) / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Report saved: {report_file}")
        
        return report


def main():
    """Main execution"""
    
    # Configuration
    TEST_FOLDER = "/home/serverai/ltdoanh/LayoutGeneration/data/samples/keyframe4check/42853_keyframes"
    OUTPUT_DIR = f"pipeline_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Advanced pipeline config
    PIPELINE_CONFIG = {
        'caption_similarity_threshold': 0.4,
        'keep_top_k_bboxes': 3,
        'refinement_use_kde': True,  # Use KDE for threshold
        'refinement_distance_weight': True,  # Apply gaussian decay from center
        'refinement_percentile': 75,  # Fallback percentile
        'search_margin': 0.25,  # Search 25% around bbox
        'expansion_margin': 0.15,  # Add 15% context
        'refinement_use_mean_shift': True,  # Mean shift direction-aware expansion ✨
    }
    
    print("\n" + "="*70)
    print("ADVANCED 3-STAGE BBOX PROCESSING PIPELINE")
    print("="*70)
    print(f"Stage 1: SAM2 Detection")
    print(f"Stage 2: BLIP2 Caption Filtering")
    print(f"Stage 3: Advanced Refinement")
    print(f"  • KDE-based thresholding")
    print(f"  • Distance weighting (gaussian decay)")
    print(f"  • HSV-normalized color variance")
    print(f"  • Mean Shift clustering (direction-aware expansion) ✨")
    print("="*70)
    
    # Initialize pipeline
    pipeline = AdvancedBBoxPipeline(device="cuda", output_dir=OUTPUT_DIR)
    pipeline.initialize()
    
    # Get test images
    images = sorted(glob.glob(f"{TEST_FOLDER}/*.jpg"))
    images = [f for f in images if "preview" not in f.lower()]
    
    print(f"\nProcessing {len(images[:3])} images...")
    
    # Process images
    all_results = []
    for img_path in images:
        result = pipeline.process_image(img_path, config=PIPELINE_CONFIG)
        all_results.append(result)
    
    # Generate report
    pipeline.generate_report()
    
    print("\n" + "="*70)
    print("✅ ADVANCED PIPELINE COMPLETE")
    print(f"📁 Results: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
