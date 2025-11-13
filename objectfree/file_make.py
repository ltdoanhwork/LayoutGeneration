import os
import json
import subprocess
from multiprocessing import Pool, cpu_count

# Paths
videos_dir = "/home/serverai/ltdoanh/LayoutGeneration/videos_OD"
annotations_dir = "/home/serverai/ltdoanh/LayoutGeneration/annotations/all_frame_annotations_via"
output_dir = "/home/serverai/ltdoanh/LayoutGeneration/extracted_images"
os.makedirs(output_dir, exist_ok=True)

def extract_frame(task):
    """Function to extract one frame using FFmpeg"""
    video_path, frame_id, img_path = task
    
    cmd = [
        'ffmpeg', '-i', video_path, '-vf', f"select=eq(n\\,{frame_id})", 
        '-vframes', '1', '-q:v', '2', img_path, '-y'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"Saved: {img_path}")
        else:
            print(f"Failed to extract frame {frame_id}: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"Timeout extracting frame {frame_id}")
    except Exception as e:
        print(f"Error: {e}")

# Duyệt qua JSON files
for root, dirs, files in os.walk(annotations_dir):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(root, file)
            movie_name = os.path.basename(os.path.dirname(root))  # e.g., Coraline
            video_id = file.replace('.json', '')  # e.g., 2hcVZo8jP6A
            video_file = f"{movie_name}_{video_id}.mp4"
            video_path = os.path.join(videos_dir, video_file)
            
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue
            
            # Load JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Create subfolder
            video_output_dir = os.path.join(output_dir, f"{movie_name}_{video_id}")
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Prepare tasks: list of (video_path, frame_id, img_path) for frames có bboxes
            tasks = []
            for frame_key, frame_data in data.items():
                if frame_data.get('regions', []):  # Chỉ frames có bboxes
                    frame_id = int(frame_key.split('.')[0])  # e.g., 0 từ "0.jpg..."
                    img_path = os.path.join(video_output_dir, f"frame_{frame_id:06d}.png")
                    tasks.append((video_path, frame_id, img_path))
            
            # Parallel extract
            num_processes = min(2, cpu_count())  # Use 2 processes or less
            with Pool(num_processes) as pool:
                pool.map(extract_frame, tasks)

print("✅ Done extracting images with FFmpeg (parallel)!")