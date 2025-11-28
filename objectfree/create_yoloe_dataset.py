"""
Script to create YOLOE dataset from keyframes and detections
Converts detection data to YOLOE format for training
"""

import os
import json
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

class YOLOEDatasetCreator:
    def __init__(self, 
                 keyframes_dir: str,
                 detections_json: str = None,
                 output_dir: str = "yoloe_dataset",
                 train_ratio: float = 0.8):
        """
        Initialize YOLOE dataset creator
        
        Args:
            keyframes_dir: Directory containing keyframe folders
            detections_json: Path to JSON file with detections (optional)
            output_dir: Output directory for YOLOE dataset
            train_ratio: Ratio of train/val split (default 0.8)
        """
        self.keyframes_dir = Path(keyframes_dir)
        self.detections_json = detections_json
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        
        # Create output directory structure
        self.setup_directories()
        
    def setup_directories(self):
        """Create YOLOE dataset directory structure"""
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        print(f"✓ Created dataset directories in: {self.output_dir}")
        
    def load_detections(self):
        """Load detections from JSON file"""
        if self.detections_json and os.path.exists(self.detections_json):
            with open(self.detections_json, 'r') as f:
                detections = json.load(f)
            print(f"✓ Loaded detections from: {self.detections_json}")
            return detections
        else:
            print("⚠️  No detections JSON provided. Will create empty labels.")
            return {}
    
    def get_all_keyframe_folders(self):
        """Get all keyframe folders"""
        folders = sorted([f for f in os.listdir(self.keyframes_dir) 
                         if os.path.isdir(self.keyframes_dir / f) and f.endswith('_keyframes')])
        print(f"✓ Found {len(folders)} keyframe folders")
        return folders
    
    def get_images_from_folder(self, folder_path: Path):
        """Get all image files from a folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in os.listdir(folder_path) 
                 if Path(f).suffix.lower() in image_extensions]
        return sorted(images)
    
    def create_label_file(self, boxes_yolo):
        """
        Create YOLO format label file content
        
        Args:
            boxes_yolo: List of YOLO format boxes [x_center, y_center, width, height] (already normalized)
            
        Returns:
            String content for label file (each line: class_id x_center y_center width height)
        """
        lines = []
        for bbox in boxes_yolo:
            # boxes_yolo already in YOLO format: [x_center, y_center, width, height]
            # All values already normalized to [0, 1]
            class_id = 0  # Single class: character
            
            # Format: class_id x_center y_center width height
            line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def process_dataset(self):
        """Process all keyframes and create YOLOE dataset"""
        # Load detections
        all_detections = self.load_detections()
        
        # Get all keyframe folders
        folders = self.get_all_keyframe_folders()
        
        # Split into train/val
        split_idx = int(len(folders) * self.train_ratio)
        train_folders = folders[:split_idx]
        val_folders = folders[split_idx:]
        
        print(f"\n📊 Dataset split:")
        print(f"  Train: {len(train_folders)} folders")
        print(f"  Val: {len(val_folders)} folders")
        
        # Process train set
        print("\n🔄 Processing training set...")
        train_stats = self._process_split(train_folders, 'train', all_detections)
        
        # Process val set
        print("\n🔄 Processing validation set...")
        val_stats = self._process_split(val_folders, 'val', all_detections)
        
        # Create data.yaml
        self.create_data_yaml(train_stats, val_stats)
        
        return train_stats, val_stats
    
    def _process_split(self, folders, split_name, all_detections):
        """Process a dataset split (train or val)"""
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'images_with_labels': 0,
            'empty_labels': 0
        }
        
        images_dir = self.output_dir / split_name / 'images'
        labels_dir = self.output_dir / split_name / 'labels'
        
        for folder_name in tqdm(folders, desc=f"Processing {split_name}"):
            folder_path = self.keyframes_dir / folder_name
            
            # Get detections for this folder from JSON
            folder_detections_list = all_detections.get(folder_name, [])
            
            # Convert list to dict mapping filename -> boxes_yolo
            folder_detections_dict = {}
            for item in folder_detections_list:
                if isinstance(item, dict) and 'filename' in item and 'boxes_yolo' in item:
                    folder_detections_dict[item['filename']] = item['boxes_yolo']
            
            # Get all images in this folder
            images = self.get_images_from_folder(folder_path)
            
            for img_name in images:
                img_path = folder_path / img_name
                
                # Check if image exists
                if not img_path.exists():
                    print(f"⚠️  Image not found: {img_path}")
                    continue
                
                # Copy image to output directory
                # Use unique name: foldername_imagename
                unique_img_name = f"{folder_name}_{img_name}"
                dst_img_path = images_dir / unique_img_name
                shutil.copy2(img_path, dst_img_path)
                
                # Get YOLO boxes for this image
                boxes_yolo = folder_detections_dict.get(img_name, [])
                
                # Create label file
                label_name = Path(unique_img_name).stem + '.txt'
                label_path = labels_dir / label_name
                
                if boxes_yolo and len(boxes_yolo) > 0:
                    label_content = self.create_label_file(boxes_yolo)
                    with open(label_path, 'w') as f:
                        f.write(label_content)
                    stats['images_with_labels'] += 1
                    stats['total_labels'] += len(boxes_yolo)
                else:
                    # Create empty label file
                    label_path.touch()
                    stats['empty_labels'] += 1
                
                stats['total_images'] += 1
        
        print(f"\n✓ {split_name.upper()} set statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Images with labels: {stats['images_with_labels']}")
        print(f"  Empty labels: {stats['empty_labels']}")
        print(f"  Total bounding boxes: {stats['total_labels']}")
        
        return stats
    
    def create_data_yaml(self, train_stats, val_stats):
        """Create data.yaml file for YOLOE training"""
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,  # Number of classes (adjust as needed)
            'names': ['character']  # Class names (adjust as needed)
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"\n✓ Created data.yaml at: {yaml_path}")
        print(f"\n📋 Dataset Summary:")
        print(f"  Total train images: {train_stats['total_images']}")
        print(f"  Total val images: {val_stats['total_images']}")
        print(f"  Total images: {train_stats['total_images'] + val_stats['total_images']}")
        print(f"  Classes: {data_yaml['nc']} - {data_yaml['names']}")
        
        return yaml_path


def main():
    """Main function to create YOLOE dataset"""
    print("="*80)
    print("YOLOE Dataset Creator for Character Detection")
    print("="*80)
    
    # Configuration
    keyframes_dir = "/home/serverai/ltdoanh/LayoutGeneration/data/output_top_keyframe_videos"
    detections_json = "/home/serverai/ltdoanh/LayoutGeneration/data/sam3_detections.json"
    output_dir = "/home/serverai/ltdoanh/LayoutGeneration/yoloe_dataset_characters"
    train_ratio = 0.85  # 85% train, 15% val
    
    # Verify input files exist
    if not os.path.exists(keyframes_dir):
        print(f"❌ Error: Keyframes directory not found: {keyframes_dir}")
        return
    
    if not os.path.exists(detections_json):
        print(f"❌ Error: Detections JSON not found: {detections_json}")
        print("Please make sure sam3_detections.json exists.")
        return
    
    print(f"\n📂 Input:")
    print(f"  Keyframes: {keyframes_dir}")
    print(f"  Detections: {detections_json}")
    print(f"\n📂 Output:")
    print(f"  Dataset: {output_dir}")
    
    # Create dataset
    creator = YOLOEDatasetCreator(
        keyframes_dir=keyframes_dir,
        detections_json=detections_json,
        output_dir=output_dir,
        train_ratio=train_ratio
    )
    
    # Process dataset
    train_stats, val_stats = creator.process_dataset()
    
    print("\n" + "="*80)
    print("✅ YOLOE Dataset Creation Complete!")
    print("="*80)
    print(f"\n📂 Dataset location: {output_dir}")
    print(f"📄 Data config: {output_dir}/data.yaml")
    print(f"\n💡 Next steps:")
    print(f"  1. Review the data.yaml file")
    print(f"  2. Update train_cartoon.py:")
    print(f"     data = '{output_dir}/data.yaml'")
    print(f"  3. Start training:")
    print(f"     cd objectfree/yoloe")
    print(f"     python3 train_cartoon.py")


if __name__ == "__main__":
    main()
