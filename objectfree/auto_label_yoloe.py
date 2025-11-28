"""
Auto-labeling script using YOLOE pretrained model
Creates initial labels for training dataset
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import cv2
from ultralytics import YOLOE
import torch

class AutoLabeler:
    def __init__(self, 
                 keyframes_dir: str,
                 output_json: str = "auto_detections.json",
                 model_path: str = "yoloe-v8l-seg.pt",
                 prompt: str = "character in cartoon",
                 confidence_threshold: float = 0.25):
        """
        Initialize auto-labeler
        
        Args:
            keyframes_dir: Directory containing keyframe folders
            output_json: Output JSON file for detections
            model_path: Path to YOLOE pretrained model
            prompt: Text prompt for detection
            confidence_threshold: Minimum confidence for detections
        """
        self.keyframes_dir = Path(keyframes_dir)
        self.output_json = output_json
        self.model_path = model_path
        self.prompt = prompt
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Initializing Auto-Labeler")
        print(f"  Device: {self.device}")
        print(f"  Model: {model_path}")
        print(f"  Prompt: '{prompt}'")
        print(f"  Confidence threshold: {confidence_threshold}")
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load YOLOE model and set prompt"""
        print(f"\n📦 Loading YOLOE model...")
        
        # Check if model exists, download if not
        if not os.path.exists(self.model_path):
            print(f"Model not found. Downloading {self.model_path}...")
            # YOLOE will auto-download from HuggingFace
        
        self.model = YOLOE(self.model_path)
        
        # Set text prompt
        print(f"🔤 Setting text prompt: '{self.prompt}'")
        tpe = self.model.get_text_pe([self.prompt])
        self.model.set_classes([self.prompt], tpe)
        
        print(f"✓ Model loaded and ready")
        
    def get_all_keyframe_folders(self):
        """Get all keyframe folders"""
        folders = sorted([f for f in os.listdir(self.keyframes_dir) 
                         if os.path.isdir(self.keyframes_dir / f) and f.endswith('_keyframes')])
        print(f"\n✓ Found {len(folders)} keyframe folders")
        return folders
    
    def get_images_from_folder(self, folder_path: Path):
        """Get all image files from a folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in os.listdir(folder_path) 
                 if Path(f).suffix.lower() in image_extensions]
        return sorted(images)
    
    def detect_images(self, image_paths: list):
        """
        Run detection on list of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of detection results
        """
        results = self.model.predict(
            source=image_paths,
            conf=self.confidence_threshold,
            save=False,
            stream=True,
            device=self.device,
            verbose=False
        )
        return list(results)
    
    def process_all_folders(self, max_folders: int = None):
        """
        Process all keyframe folders and generate detections
        
        Args:
            max_folders: Maximum number of folders to process (None = all)
        """
        folders = self.get_all_keyframe_folders()
        
        if max_folders:
            folders = folders[:max_folders]
            print(f"⚠️  Processing only first {max_folders} folders")
        
        all_detections = {}
        total_images = 0
        total_detections = 0
        
        print(f"\n🔄 Processing {len(folders)} folders...")
        
        for folder_name in tqdm(folders, desc="Processing folders"):
            folder_path = self.keyframes_dir / folder_name
            
            # Get all images in this folder
            images = self.get_images_from_folder(folder_path)
            
            if not images:
                all_detections[folder_name] = {}
                continue
            
            # Prepare image paths
            image_paths = [str(folder_path / img) for img in images]
            
            # Run detection
            results = self.detect_images(image_paths)
            
            # Store detections
            folder_detections = {}
            for img_name, result in zip(images, results):
                detections = []
                
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else 0
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': self.prompt
                        }
                        detections.append(detection)
                        total_detections += 1
                
                folder_detections[img_name] = detections
                total_images += 1
            
            all_detections[folder_name] = folder_detections
        
        # Save detections to JSON
        print(f"\n💾 Saving detections to: {self.output_json}")
        with open(self.output_json, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        print(f"\n✅ Auto-labeling complete!")
        print(f"  Total folders: {len(folders)}")
        print(f"  Total images: {total_images}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per image: {total_detections/total_images:.2f}")
        
        return all_detections


def main():
    """Main function for auto-labeling"""
    print("="*80)
    print("YOLOE Auto-Labeling for Character Detection")
    print("="*80)
    
    # Configuration
    keyframes_dir = "/home/serverai/ltdoanh/LayoutGeneration/data/output_top_keyframe_videos"
    output_json = "/home/serverai/ltdoanh/LayoutGeneration/data/yoloe_auto_detections.json"
    model_path = "/home/serverai/ltdoanh/LayoutGeneration/objectfree/yoloe/yoloe-v8l-seg.pt"
    
    # Check if base model exists
    if not os.path.exists(model_path):
        print(f"⚠️  Base model not found at: {model_path}")
        print("Attempting to use model name for auto-download...")
        model_path = "yoloe-v8l-seg.pt"
    
    # Create auto-labeler
    labeler = AutoLabeler(
        keyframes_dir=keyframes_dir,
        output_json=output_json,
        model_path=model_path,
        prompt="character in cartoon",
        confidence_threshold=0.25
    )
    
    # Process all folders (or limit for testing)
    # For testing, you can use: max_folders=10
    detections = labeler.process_all_folders(max_folders=None)  # Process all
    
    print("\n" + "="*80)
    print("✅ Auto-labeling Complete!")
    print("="*80)
    print(f"\n📂 Detections saved to: {output_json}")
    print(f"\n💡 Next steps:")
    print(f"  1. Review the generated detections")
    print(f"  2. Run create_yoloe_dataset.py to create training dataset")
    print(f"  3. Train YOLOE model with train_cartoon.py")


if __name__ == "__main__":
    main()
