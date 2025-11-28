"""
Complete pipeline to create YOLOE training dataset
Combines auto-labeling and dataset creation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from auto_label_yoloe import AutoLabeler
from create_yoloe_dataset import YOLOEDatasetCreator


def main():
    """Complete pipeline for YOLOE dataset creation"""
    
    print("="*80)
    print("YOLOE Dataset Creation Pipeline")
    print("="*80)
    
    # ============================================================================
    # STEP 1: Configuration
    # ============================================================================
    print("\n📋 Step 1: Configuration")
    print("-" * 80)
    
    BASE_DIR = Path("/home/serverai/ltdoanh/LayoutGeneration")
    
    # Input
    keyframes_dir = BASE_DIR / "data/output_top_keyframe_videos"
    
    # Intermediate
    auto_detections_json = BASE_DIR / "data/yoloe_auto_detections.json"
    
    # Output
    output_dataset_dir = BASE_DIR / "yoloe_dataset_characters"
    
    # Model
    model_path = BASE_DIR / "objectfree/yoloe/yoloe-v8l-seg.pt"
    if not model_path.exists():
        model_path = "yoloe-v8l-seg.pt"  # Will auto-download
    
    print(f"  Keyframes: {keyframes_dir}")
    print(f"  Model: {model_path}")
    print(f"  Output dataset: {output_dataset_dir}")
    
    # ============================================================================
    # STEP 2: Auto-labeling (Optional - skip if you already have labels)
    # ============================================================================
    create_labels = True  # Set to False if you already have labels
    
    if create_labels:
        print("\n" + "="*80)
        print("📍 Step 2: Auto-Labeling with YOLOE Pretrained Model")
        print("="*80)
        
        # Ask user for confirmation
        response = input("\nDo you want to auto-label images? (y/n, default=y): ").strip().lower()
        
        if response == '' or response == 'y':
            try:
                # Create auto-labeler
                labeler = AutoLabeler(
                    keyframes_dir=str(keyframes_dir),
                    output_json=str(auto_detections_json),
                    model_path=str(model_path),
                    prompt="character in cartoon",
                    confidence_threshold=0.25
                )
                
                # Ask if user wants to process all or limit
                limit_response = input("\nProcess all folders? (y/n, default=y): ").strip().lower()
                max_folders = None
                
                if limit_response == 'n':
                    try:
                        max_folders = int(input("Enter max number of folders to process: "))
                    except:
                        max_folders = 10
                        print(f"Invalid input, using default: {max_folders}")
                
                # Process
                detections = labeler.process_all_folders(max_folders=max_folders)
                
                print(f"\n✅ Auto-labeling complete!")
                print(f"   Detections saved to: {auto_detections_json}")
                
            except Exception as e:
                print(f"\n❌ Error during auto-labeling: {e}")
                print("You can:")
                print("  1. Fix the error and run again")
                print("  2. Manually create detections JSON")
                print("  3. Continue without labels (will create empty labels)")
                return
        else:
            print("\n⏭️  Skipping auto-labeling")
    
    # ============================================================================
    # STEP 3: Create YOLOE Dataset
    # ============================================================================
    print("\n" + "="*80)
    print("📍 Step 3: Creating YOLOE Training Dataset")
    print("="*80)
    
    # Check if detections JSON exists
    detections_json_path = str(auto_detections_json) if auto_detections_json.exists() else None
    
    if detections_json_path is None:
        print("\n⚠️  No detections JSON found. Will create empty labels.")
        response = input("Continue? (y/n, default=y): ").strip().lower()
        if response == 'n':
            print("Exiting...")
            return
    
    try:
        # Create dataset creator
        creator = YOLOEDatasetCreator(
            keyframes_dir=str(keyframes_dir),
            detections_json=detections_json_path,
            output_dir=str(output_dataset_dir),
            train_ratio=0.8  # 80% train, 20% val
        )
        
        # Process dataset
        train_stats, val_stats = creator.process_dataset()
        
        print("\n✅ Dataset creation complete!")
        
    except Exception as e:
        print(f"\n❌ Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # STEP 4: Summary and Next Steps
    # ============================================================================
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    
    print(f"\n📂 Dataset Information:")
    print(f"  Location: {output_dataset_dir}")
    print(f"  Data config: {output_dataset_dir}/data.yaml")
    print(f"  Train images: {train_stats['total_images']}")
    print(f"  Val images: {val_stats['total_images']}")
    print(f"  Total images: {train_stats['total_images'] + val_stats['total_images']}")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Review the dataset structure:")
    print(f"     cd {output_dataset_dir}")
    print(f"     ls -la train/images/ train/labels/")
    print(f"  ")
    print(f"  2. Update train_cartoon.py with dataset path:")
    print(f"     data = '{output_dataset_dir}/data.yaml'")
    print(f"  ")
    print(f"  3. Start training:")
    print(f"     cd objectfree/yoloe")
    print(f"     python3 train_cartoon.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
