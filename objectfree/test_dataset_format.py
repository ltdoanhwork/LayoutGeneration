"""
Test script to verify sam3_detections.json format
"""

import json
from pathlib import Path

def test_detections_format():
    """Test and display sample from sam3_detections.json"""
    
    json_path = "/home/serverai/ltdoanh/LayoutGeneration/data/sam3_detections.json"
    
    print("="*80)
    print("Testing sam3_detections.json Format")
    print("="*80)
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n✓ Loaded JSON successfully")
    print(f"  Total folders: {len(data)}")
    
    # Count statistics
    total_images = 0
    total_boxes = 0
    images_with_boxes = 0
    folders_with_data = 0
    
    for folder_name, items in data.items():
        if items:  # Has data
            folders_with_data += 1
            total_images += len(items)
            
            for item in items:
                if 'boxes_yolo' in item and len(item['boxes_yolo']) > 0:
                    images_with_boxes += 1
                    total_boxes += len(item['boxes_yolo'])
    
    print(f"\n📊 Statistics:")
    print(f"  Folders with data: {folders_with_data}/{len(data)}")
    print(f"  Total images: {total_images}")
    print(f"  Images with boxes: {images_with_boxes}")
    print(f"  Total bounding boxes: {total_boxes}")
    print(f"  Average boxes per image: {total_boxes/total_images if total_images > 0 else 0:.2f}")
    
    # Show sample
    print(f"\n📝 Sample Data:")
    print("-" * 80)
    
    # Find first folder with data
    sample_folder = None
    for folder_name, items in data.items():
        if items and len(items) > 0:
            sample_folder = folder_name
            break
    
    if sample_folder:
        print(f"Folder: {sample_folder}")
        sample_items = data[sample_folder][:3]  # First 3 items
        
        for i, item in enumerate(sample_items, 1):
            print(f"\n  Image {i}:")
            print(f"    Filename: {item['filename']}")
            print(f"    Image size: {item['image_width']}x{item['image_height']}")
            print(f"    Number of boxes: {len(item['boxes_yolo'])}")
            
            if item['boxes_yolo']:
                print(f"    First box (YOLO format):")
                box = item['boxes_yolo'][0]
                print(f"      x_center: {box[0]:.4f}")
                print(f"      y_center: {box[1]:.4f}")
                print(f"      width: {box[2]:.4f}")
                print(f"      height: {box[3]:.4f}")
    else:
        print("No folders with data found!")
    
    print("\n" + "="*80)
    print("✅ Format verification complete!")
    print("="*80)
    
    # Expected format
    print("\n📋 Expected Format per image:")
    print("""
    {
      "filename": "image_name.jpg",
      "boxes_yolo": [
        [x_center, y_center, width, height],  // normalized 0-1
        [x_center, y_center, width, height],  // can have multiple boxes
        ...
      ],
      "image_width": 852,
      "image_height": 480
    }
    """)
    
    print("\n💡 Ready to create YOLOE dataset!")
    print("   Run: python3 create_yoloe_dataset.py")


if __name__ == "__main__":
    test_detections_format()
