#!/usr/bin/env python3
"""
Complete Pipeline: Object Detection → Story Coherence → Comprehensive Evaluation
Input: keyframes folder → Output: all evaluation results
"""
import os
import sys
import torch
import json
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import yaml

# Import các modules
from inference_dino import LoadDetector
from story_coherence_evaluator import StoryCoherenceEvaluator
from eval_comprehensive import ImageRegionAnalyzer


class CompletePipeline:
    """Complete pipeline combining all evaluation methods"""
    
    def __init__(self, device="cuda", output_dir=None, config_path="objectfree/config.yaml"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.output_dir = output_dir
        self.config_path = config_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['config']
        
        self.similarity_threshold = self.config.get('SIMILARITY_THRESHOLD', 0.3)
        
        print(f"Using device: {self.device}")
        print(f"Similarity threshold: {self.similarity_threshold}")
        
        # Initialize evaluators
        self.object_detector = None
        self.story_evaluator = None
        self.comprehensive_analyzer = None
    
    def initialize_detectors(self):
        """Initialize all detectors"""
        self.object_detector = LoadDetector(
            config_path="objectfree/config.yaml",
            checkpoint_path="/home/serverai/ltdoanh/LayoutGeneration/objectfree/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt",
            image_path=None,
            device=self.device,
            batch_size=1,
            output_dir=self.output_dir or "outputs"
        )
        
        self.story_evaluator = StoryCoherenceEvaluator(
            blip_model_name="Salesforce/blip-image-captioning-large",
            device=self.device
        )
        

    def run_object_detection(self, keyframes_folder, output_dir):
        """Step 1: Run object detection"""
        print("STEP 1: OBJECT DETECTION")
        print(f"{'='*70}")
        

        results = self.object_detector.run_inference(
            keyframes_folder=keyframes_folder,
            output_dir=output_dir
        )
        print(f"Object detection completed: {len(results)} images processed")
        return True
 

    def run_story_coherence(self, detection_json, keyframes_folder, output_dir):
        """Step 2: Run story coherence evaluation"""
        print("STEP 2: STORY COHERENCE EVALUATION")
        print(f"{'='*70}")
    
        results = self.story_evaluator.evaluate_batch(
            detection_results_json=detection_json,
            keyframes_folder=keyframes_folder,
            output_dir=output_dir,
            save_crops=False,  # Không lưu crop images nữa
            similarity_threshold=self.similarity_threshold
        )
        print(f"Story coherence completed: {len(results)} images evaluated")
        return results
    
    def run_comprehensive_eval(self, keyframes_folder, output_dir):
        """Step 3: Run comprehensive evaluation (entropy, edge, color)"""
        print("STEP 3: COMPREHENSIVE EVALUATION")
        print(f"{'='*70}")
        
        # Get all images
        image_paths = sorted(glob.glob(os.path.join(keyframes_folder, "*.jpg")))
        image_paths = [p for p in image_paths if "preview" not in os.path.basename(p).lower()]
        
        if len(image_paths) == 0:
            print(f"✗ No images found in {keyframes_folder}")
            return []
        
        print(f"Processing {len(image_paths)} images...")
        
        all_stats = []
        for image_path in tqdm(image_paths, desc="  Comprehensive evaluation"):
            # Create analyzer for each image
            analyzer = ImageRegionAnalyzer(image_path)
            
            # Compute combined score
            combined_score, individual_maps, key_regions = analyzer.visualize_and_save(output_dir)
    
        # Save results
        summary_path = os.path.join(output_dir, 'comprehensive_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        df = pd.DataFrame(all_stats)
        csv_path = os.path.join(output_dir, 'comprehensive_summary.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Comprehensive evaluation completed: {len(all_stats)} images processed")
        return all_stats
    
    def annotate_image_with_bbox(self, image_path, crop_results, output_path=None):
        """
        Annotate bbox với màu sắc trực tiếp lên ảnh bằng PIL

        Args:
            image_path: Đường dẫn ảnh gốc
            crop_results: List các crop result từ story coherence
            output_path: Đường dẫn lưu ảnh output (optional)

        Returns:
            annotated_image: PIL Image đã được annotate
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Add padding to top of image for text labels
        padding_top = 50  # Padding for text labels
        padded_image = Image.new('RGB', (image.width, image.height + padding_top), (255, 255, 255))
        padded_image.paste(image, (0, padding_top))
        draw = ImageDraw.Draw(padded_image)

        # Filter chỉ lấy bbox được keep
        kept_results = [r for r in crop_results if r['keep']]

        # Draw bboxes (adjust y coordinates for padding)
        for result in kept_results:
            bbox = result['bbox']
            color = result['bbox_color']
            similarity = result['similarity']
            det_id = result['detection_id']

            # Convert bbox coordinates (x1, y1, x2, y2) and adjust for padding
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            y1 += padding_top  # Adjust y coordinates for padding
            y2 += padding_top

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            # Draw label background
            label = f'ID{det_id}: {similarity:.3f}'
            # Try to use default font, fallback if not available
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

            # Get text size
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            # Position text above bbox, but ensure it's within image bounds
            text_y = max(text_height + 4, y1 - text_height - 4)
            
            # Draw text background
            draw.rectangle([x1, text_y - text_height - 2, x1 + text_width + 8, text_y], fill=color)

            # Draw text
            draw.text((x1 + 4, text_y - text_height), label, fill='white', font=font)

        # Save if output_path provided
        if output_path:
            padded_image.save(output_path)
            print(f"Saved annotated image to: {output_path}")

        return padded_image
    
    def create_final_report(self, folder_name, detection_results, story_results, comprehensive_results, output_dir):
        """Create final combined report"""
        print("CREATING FINAL REPORT")
        print(f"{'='*70}")
        
        # Combine all results
        final_report = {
            'folder_name': folder_name,
            'timestamp': str(pd.Timestamp.now()),
            'summary': {
                'total_images': len(detection_results) if detection_results else 0,
                'story_eval_images': len(story_results),
                'comprehensive_eval_images': len(comprehensive_results),
                'successful_comprehensive': sum(1 for r in comprehensive_results if r.get('status') == 'success')
            },
            'detection_results': detection_results,
            'story_results': story_results,
            'comprehensive_results': comprehensive_results
        }
        
        # Calculate overall metrics
        if comprehensive_results:
            df_comp = pd.DataFrame([r for r in comprehensive_results if r.get('status') == 'success'])
            if not df_comp.empty:
                final_report['overall_metrics'] = {
                    'avg_entropy': df_comp['entropy_mean'].mean(),
                    'avg_edge_density': df_comp['edge_mean'].mean(),
                    'avg_color_variance': df_comp['color_mean'].mean(),
                    'avg_combined_score': df_comp['combined_mean'].mean()
                }
        
        # Save final report
        report_path = os.path.join(output_dir, 'final_report.json')
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f" Final report saved: {report_path}")
        return final_report
    
    def process_single_folder(self, keyframes_folder, output_base="/home/serverai/ltdoanh/LayoutGeneration/outputs_complete"):
        """Process a single keyframes folder through all steps"""
        folder_name = os.path.basename(keyframes_folder)
        
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLDER: {folder_name}")
        print(f"{'='*80}")
        
        # Create output directory
        output_dir = os.path.join(output_base, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Object Detection
        detection_output = os.path.join(output_dir, 'object_detection')
        detection_success = self.run_object_detection(keyframes_folder, detection_output)
        
        detection_results = None
        story_results = []
        comprehensive_results = []
        
        if detection_success:
            detection_json = os.path.join(detection_output, 'detection_results.json')
            
            # Step 2: Story Coherence
            story_output = os.path.join(output_dir, 'story_coherence')
            story_results = self.run_story_coherence(detection_json, keyframes_folder, story_output)
            
            # Step 2.5: Create annotated images with bbox colors
            annotated_output = os.path.join(output_dir, 'annotated_images')
            os.makedirs(annotated_output, exist_ok=True)
            
            # Get all keyframe images
            keyframe_images = sorted(glob.glob(os.path.join(keyframes_folder, "*.jpg")))
            keyframe_images = [p for p in keyframe_images if "preview" not in os.path.basename(p).lower()]
            
            print("STEP 2.5: CREATING ANNOTATED IMAGES")
            print(f"{'='*70}")
            
            for idx, img_path in enumerate(keyframe_images):
                if idx < len(story_results):
                    crop_results = story_results[idx]['crop_results']
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    output_path = os.path.join(annotated_output, f"{base_name}_annotated.png")
                    self.annotate_image_with_bbox(img_path, crop_results, output_path)
            
            print(f"Annotated images created: {len(keyframe_images)} images processed")
            
            # Load detection results for final report
            try:
                with open(detection_json, 'r') as f:
                    detection_data = json.load(f)
                    detection_results = detection_data.get('results', [])
            except:
                detection_results = None

            # Step 2.6: Save cropped object images
            cropped_output = os.path.join(output_dir, 'cropped_objects')
            os.makedirs(cropped_output, exist_ok=True)

            print("STEP 2.6: SAVING CROPPED OBJECT IMAGES")
            print(f"{'='*70}")

            for idx, img_path in enumerate(keyframe_images):
                if idx < len(story_results):
                    crop_results = story_results[idx]['crop_results']
                    image = Image.open(img_path).convert("RGB")
                    base_name = os.path.splitext(os.path.basename(img_path))[0]

                    for obj_idx, result in enumerate(crop_results):
                        if result.get('keep', True):  # chỉ lưu những bbox được giữ lại
                            x1, y1, x2, y2 = [int(c) for c in result['bbox']]
                            cropped = image.crop((x1, y1, x2, y2))

                            # tạo tên file rõ ràng
                            label = f"id{result['detection_id']}"
                            crop_filename = f"{base_name}_{label}.png"
                            crop_path = os.path.join(cropped_output, crop_filename)

                            cropped.save(crop_path)
            print(f"Cropped objects saved in: {cropped_output}")

        # Step 3: Comprehensive Evaluation (always run, independent of detection)
        comprehensive_output = os.path.join(output_dir, 'comprehensive_eval')
        comprehensive_results = self.run_comprehensive_eval(keyframes_folder, comprehensive_output)
        
        # Create final report
        final_report = self.create_final_report(
            folder_name, detection_results, story_results, comprehensive_results, output_dir
        )
        
        return {
            'folder_name': folder_name,
            'detection_success': detection_success,
            'story_results_count': len(story_results),
            'comprehensive_results_count': len(comprehensive_results),
            'output_dir': output_dir
        }


def main():
    input_arg = sys.argv[1]
    
    # Initialize pipeline
    pipeline = CompletePipeline(device="cuda")
    pipeline.initialize_detectors()
    
    if input_arg == "all":
        # Process all folders
        base_folder = "/home/serverai/ltdoanh/LayoutGeneration/samples/keyframe4check"
        keyframe_folders = []
        
        for item in sorted(os.listdir(base_folder)):
            item_path = os.path.join(base_folder, item)
            if os.path.isdir(item_path) and item.endswith("_keyframes"):
                keyframe_folders.append(item_path)
        
        print(f"\nFound {len(keyframe_folders)} folders to process:")
        for i, folder in enumerate(keyframe_folders):
            print(f"  {i+1}. {os.path.basename(folder)}")
        
        # Process all folders
        all_results = []
        for folder_path in keyframe_folders:
            result = pipeline.process_single_folder(folder_path)
            all_results.append(result)

       
        total_story = sum(r['story_results_count'] for r in all_results)
        total_comprehensive = sum(r['comprehensive_results_count'] for r in all_results)
    else:
        # Process single folder
        keyframes_folder = input_arg
        if not os.path.exists(keyframes_folder):
            print(f"[ERROR] Folder not found: {keyframes_folder}")
            return
        
        result = pipeline.process_single_folder(keyframes_folder)

if __name__ == "__main__":
    main()