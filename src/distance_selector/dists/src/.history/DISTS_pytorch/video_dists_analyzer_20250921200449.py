"""
Video DISTS Analyzer
Analyzes video frames using DISTS metric to calculate perceptual quality scores
between consecutive frames with the first frame as reference.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import argparse
from scipy.signal import find_peaks
from DISTS_pt import DISTS, prepare_image

def extract_frames_from_video(video_path, max_frames=None):
    """
    Extract frames from video file
    
    Args:
        video_path (str): Path to video file
        max_frames (int): Maximum number of frames to extract (None for all)
    
    Returns:
        list: List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps} FPS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
        
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
            
        if frame_count % 10 == 0:
            print(f"Extracted {frame_count}/{total_frames if not max_frames else max_frames} frames")
    
    cap.release()
    print(f"Extracted {len(frames)} frames total")
    return frames

def calculate_dists_scores(frames, device='cpu'):
    """
    Calculate DISTS scores between consecutive frames using first frame as reference
    
    Args:
        frames (list): List of PIL Images
        device (str): Device to run computation on ('cpu' or 'cuda')
    
    Returns:
        list: DISTS scores for each frame comparison
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for comparison")
    
    # Load DISTS model
    model = DISTS().to(device)
    model.eval()
    
    scores = []
    
    # Prepare reference frame (first frame)
    ref_frame = prepare_image(frames[0].convert("RGB"))
    ref_frame = ref_frame.to(device)
    
    print("Calculating DISTS scores...")
    
    # Calculate scores for each subsequent frame
    for i in range(1, len(frames)):
        # Prepare current frame
        current_frame = prepare_image(frames[i].convert("RGB"))
        current_frame = current_frame.to(device)
        
        # Ensure same dimensions
        if ref_frame.shape != current_frame.shape:
            print(f"Warning: Frame {i} has different dimensions. Resizing...")
            # Resize current frame to match reference
            _, _, h, w = ref_frame.shape
            current_frame = torch.nn.functional.interpolate(
                current_frame, size=(h, w), mode='bilinear', align_corners=False
            )
        
        # Calculate DISTS score
        with torch.no_grad():
            score = model(ref_frame, current_frame)
            scores.append(score.item())
        
        if (i) % 10 == 0:
            print(f"Processed {i}/{len(frames)-1} frame comparisons")
    
    return scores

def detect_peaks_and_valleys(scores, prominence=0.05, distance=5, max_keyframes=100):
    """
    Detect significant peaks and valleys in DISTS scores to identify keyframes
    
    Args:
        scores (list): DISTS scores
        prominence (float): Minimum prominence for peak detection (higher = more selective)
        distance (int): Minimum distance between peaks in frames
        max_keyframes (int): Maximum number of keyframes to return (default: 100)
    
    Returns:
        tuple: (peaks_indices, valleys_indices, keyframes_info)
    """
    scores_array = np.array(scores)
    
    # Find peaks (high points - sudden increases in score)
    peaks, peak_properties = find_peaks(scores_array, 
                                      prominence=prominence, 
                                      distance=distance)
    
    # Find valleys by finding peaks in inverted signal
    valleys, valley_properties = find_peaks(-scores_array, 
                                          prominence=prominence, 
                                          distance=distance)
    
    # Combine and sort all keyframes
    all_keyframes = []
    
    # Add peaks
    for i, peak_idx in enumerate(peaks):
        frame_number = peak_idx + 2  # +2 because we start comparison from frame 2
        all_keyframes.append({
            'frame': frame_number,
            'score': scores_array[peak_idx],
            'type': 'peak',
            'prominence': peak_properties['prominences'][i]
        })
    
    # Add valleys
    for i, valley_idx in enumerate(valleys):
        frame_number = valley_idx + 2  # +2 because we start comparison from frame 2
        all_keyframes.append({
            'frame': frame_number,
            'score': scores_array[valley_idx],
            'type': 'valley',
            'prominence': valley_properties['prominences'][i]
        })
    
    # Sort by frame number
    all_keyframes.sort(key=lambda x: x['frame'])
    
    # Limit number of keyframes if needed
    if len(all_keyframes) > max_keyframes:
        # Sort by prominence (descending) to get the most significant keyframes
        all_keyframes_by_prominence = sorted(all_keyframes, key=lambda x: x['prominence'], reverse=True)
        selected_keyframes = all_keyframes_by_prominence[:max_keyframes]
        # Sort back by frame number for output
        selected_keyframes.sort(key=lambda x: x['frame'])
        print(f"Limited keyframes from {len(all_keyframes)} to {max_keyframes} (selected most prominent)")
        all_keyframes = selected_keyframes
    
    return peaks, valleys, all_keyframes

def plot_dists_scores(scores, save_path=None, video_name="Video", keyframes=None):
    """
    Plot DISTS scores over frame sequence with keyframes highlighted
    
    Args:
        scores (list): DISTS scores
        save_path (str): Path to save plot (optional)
        video_name (str): Name of video for title
        keyframes (list): List of keyframe information from detect_peaks_and_valleys
    """
    plt.figure(figsize=(14, 8))
    
    frame_numbers = list(range(2, len(scores) + 2))  # Start from frame 2 (comparison with frame 1)
    
    plt.plot(frame_numbers, scores, 'b-', linewidth=2, marker='o', markersize=3, label='DISTS Scores')
    plt.title(f'DISTS Scores Over Time with Keyframes - {video_name}', fontsize=14)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('DISTS Score (vs Frame 1)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Mark keyframes if provided
    if keyframes:
        peaks = [kf for kf in keyframes if kf['type'] == 'peak']
        valleys = [kf for kf in keyframes if kf['type'] == 'valley']
        
        if peaks:
            peak_frames = [kf['frame'] for kf in peaks]
            peak_scores = [kf['score'] for kf in peaks]
            plt.scatter(peak_frames, peak_scores, color='red', s=100, marker='^', 
                       label=f'Peaks ({len(peaks)})', zorder=5)
        
        if valleys:
            valley_frames = [kf['frame'] for kf in valleys]
            valley_scores = [kf['score'] for kf in valleys]
            plt.scatter(valley_frames, valley_scores, color='orange', s=100, marker='v', 
                       label=f'Valleys ({len(valleys)})', zorder=5)
    
    # Add statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    plt.axhline(y=mean_score, color='gray', linestyle='--', alpha=0.7, label=f'Mean: {mean_score:.4f}')
    
    # Add text box with statistics
    stats_text = f'Statistics:\nMean: {mean_score:.4f}\nStd: {std_score:.4f}\nMin: {min_score:.4f}\nMax: {max_score:.4f}'
    if keyframes:
        stats_text += f'\nKeyframes: {len(keyframes)}'
        stats_text += f'\nPeaks: {len([kf for kf in keyframes if kf["type"] == "peak"])}'
        stats_text += f'\nValleys: {len([kf for kf in keyframes if kf["type"] == "valley"])}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def save_keyframes_info(keyframes, video_path, output_dir=None):
    """
    Save keyframes information to a text file
    
    Args:
        keyframes (list): List of keyframe information
        video_path (str): Path to the original video
        output_dir (str): Directory to save output files
    
    Returns:
        str: Path to saved keyframes file
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keyframes_file = os.path.join(output_dir, f"{video_name}_keyframes.txt")
    
    with open(keyframes_file, 'w', encoding='utf-8') as f:
        f.write(f"Keyframes Analysis for: {os.path.basename(video_path)}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total keyframes detected: {len(keyframes)}\n")
        f.write(f"Peaks: {len([kf for kf in keyframes if kf['type'] == 'peak'])}\n")
        f.write(f"Valleys: {len([kf for kf in keyframes if kf['type'] == 'valley'])}\n\n")
        
        f.write("Keyframe Details:\n")
        f.write("-" * 30 + "\n")
        f.write("Frame\tType\tScore\tProminence\n")
        f.write("-" * 30 + "\n")
        
        for kf in keyframes:
            f.write(f"{kf['frame']}\t{kf['type']}\t{kf['score']:.4f}\t{kf['prominence']:.4f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Frame List (for easy copying):\n")
        frame_list = [str(kf['frame']) for kf in keyframes]
        f.write(", ".join(frame_list) + "\n")
    
    return keyframes_file

def save_keyframe_images(frames, keyframes, video_path, output_dir=None):
    """
    Save keyframe images to disk
    
    Args:
        frames (list): List of PIL Images (all frames)
        keyframes (list): List of keyframe information
        video_path (str): Path to the original video
        output_dir (str): Directory to save keyframe images
    
    Returns:
        str: Path to keyframes directory
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keyframes_dir = os.path.join(output_dir, f"{video_name}_keyframes")
    
    # Create keyframes directory if it doesn't exist
    os.makedirs(keyframes_dir, exist_ok=True)
    
    print(f"Saving {len(keyframes)} keyframe images...")
    
    saved_count = 0
    for i, kf in enumerate(keyframes):
        frame_idx = kf['frame'] - 1  # Convert to 0-based index for frames list
        
        # Make sure frame index is valid
        if 0 <= frame_idx < len(frames):
            frame_image = frames[frame_idx]
            
            # Create filename with frame number, type, and score
            filename = f"keyframe_{kf['frame']:04d}_{kf['type']}_score{kf['score']:.3f}.jpg"
            filepath = os.path.join(keyframes_dir, filename)
            
            # Save image
            frame_image.save(filepath, 'JPEG', quality=95)
            saved_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"Saved {i + 1}/{len(keyframes)} keyframe images")
        else:
            print(f"Warning: Invalid frame index {frame_idx} for keyframe {kf['frame']}")
    
    print(f"Saved {saved_count} keyframe images to: {keyframes_dir}")
    return keyframes_dir

def create_timeline_visualization(frames, keyframes, video_path, fps=30, output_dir=None, 
                                thumb_size=(120, 80), cols=20):
    """
    Create a timeline visualization showing all keyframes in chronological order
    
    Args:
        frames (list): List of PIL Images (all frames)
        keyframes (list): List of keyframe information
        video_path (str): Path to the original video
        fps (float): Video frame rate for time calculation
        output_dir (str): Directory to save timeline image
        thumb_size (tuple): Size of each thumbnail in pixels (width, height)
        cols (int): Number of columns in the grid
    
    Returns:
        str: Path to timeline image
    """
    if not keyframes:
        print("No keyframes to create timeline")
        return None
    
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timeline_path = os.path.join(output_dir, f"{video_name}_timeline.jpg")
    
    # Calculate grid dimensions
    rows = (len(keyframes) + cols - 1) // cols
    
    # Calculate canvas size
    thumb_w, thumb_h = thumb_size
    margin = 10
    label_height = 40
    header_height = 60
    
    canvas_width = cols * (thumb_w + margin) + margin
    canvas_height = header_height + rows * (thumb_h + label_height + margin) + margin
    
    # Create canvas
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw title
    title = f"Timeline Visualization - {os.path.basename(video_path)}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((canvas_width - title_width) // 2, 10), title, fill='black', font=title_font)
    
    # Draw subtitle with stats
    subtitle = f"Total Keyframes: {len(keyframes)} | Peaks: {len([kf for kf in keyframes if kf['type'] == 'peak'])} | Valleys: {len([kf for kf in keyframes if kf['type'] == 'valley'])}"
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    draw.text(((canvas_width - subtitle_width) // 2, 35), subtitle, fill='gray', font=font)
    
    print(f"Creating timeline with {len(keyframes)} keyframes...")
    
    for i, kf in enumerate(keyframes):
        row = i // cols
        col = i % cols
        
        # Calculate position
        x = margin + col * (thumb_w + margin)
        y = header_height + row * (thumb_h + label_height + margin)
        
        # Get frame image
        frame_idx = kf['frame'] - 1  # Convert to 0-based index
        if 0 <= frame_idx < len(frames):
            frame_img = frames[frame_idx].copy()
            
            # Resize frame to thumbnail size
            frame_img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            
            # Create thumbnail background (to handle aspect ratio differences)
            thumb_bg = Image.new('RGB', thumb_size, 'lightgray')
            
            # Center the image on background
            paste_x = (thumb_size[0] - frame_img.size[0]) // 2
            paste_y = (thumb_size[1] - frame_img.size[1]) // 2
            thumb_bg.paste(frame_img, (paste_x, paste_y))
            
            # Paste thumbnail to canvas
            canvas.paste(thumb_bg, (x, y))
            
            # Draw border based on keyframe type
            border_color = 'red' if kf['type'] == 'peak' else 'orange'
            border_width = 3
            draw.rectangle([x-border_width, y-border_width, 
                          x+thumb_w+border_width-1, y+thumb_h+border_width-1], 
                         outline=border_color, width=border_width)
            
            # Draw frame information
            time_sec = kf['frame'] / fps
            minutes = int(time_sec // 60)
            seconds = time_sec % 60
            
            # Frame number and time
            frame_text = f"Frame {kf['frame']}"
            time_text = f"{minutes:02d}:{seconds:05.2f}"
            score_text = f"Score: {kf['score']:.3f}"
            type_text = f"{kf['type'].upper()}"
            
            # Draw text labels
            text_y = y + thumb_h + 5
            draw.text((x, text_y), frame_text, fill='black', font=font)
            draw.text((x, text_y + 12), time_text, fill='blue', font=font)
            draw.text((x, text_y + 24), score_text, fill='green', font=font)
            
            # Draw type with color
            type_color = 'red' if kf['type'] == 'peak' else 'orange'
            draw.text((x + thumb_w - 40, text_y), type_text, fill=type_color, font=font)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(keyframes)} thumbnails")
    
    # Draw legend
    legend_y = canvas_height - 30
    draw.text((margin, legend_y), "Legend:", fill='black', font=font)
    draw.rectangle([margin + 50, legend_y - 2, margin + 60, legend_y + 8], outline='red', width=2)
    draw.text((margin + 65, legend_y), "Peak", fill='red', font=font)
    draw.rectangle([margin + 110, legend_y - 2, margin + 120, legend_y + 8], outline='orange', width=2)
    draw.text((margin + 125, legend_y), "Valley", fill='orange', font=font)
    
    # Save timeline
    canvas.save(timeline_path, 'JPEG', quality=95)
    print(f"Timeline visualization saved to: {timeline_path}")
    
    return timeline_path

def main():
    parser = argparse.ArgumentParser(description='Analyze video using DISTS metric and detect keyframes')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process')
    parser.add_argument('--output', type=str, default=None, help='Path to save plot (optional)')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'], 
                       help='Device to use for computation')
    parser.add_argument('--prominence', type=float, default=0.05, 
                       help='Minimum prominence for peak detection (higher = more selective)')
    parser.add_argument('--distance', type=int, default=5, 
                       help='Minimum distance between peaks in frames')
    parser.add_argument('--max_keyframes', type=int, default=100, 
                       help='Maximum number of keyframes to detect (default: 100)')
    parser.add_argument('--save_keyframes', action='store_true', 
                       help='Save keyframes list to text file')
    parser.add_argument('--save_images', action='store_true', 
                       help='Save keyframe images to disk')
    parser.add_argument('--create_timeline', action='store_true', 
                       help='Create timeline visualization of all keyframes')
    parser.add_argument('--timeline_cols', type=int, default=20, 
                       help='Number of columns in timeline grid (default: 20)')
    parser.add_argument('--thumb_size', type=int, nargs=2, default=[120, 80], 
                       help='Thumbnail size for timeline (width height, default: 120 80)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    try:
        # Extract frames from video
        print("Extracting frames from video...")
        frames = extract_frames_from_video(args.video, args.max_frames)
        
        if len(frames) < 2:
            raise ValueError("Video must have at least 2 frames")
        
        # Calculate DISTS scores
        scores = calculate_dists_scores(frames, device)
        
        # Detect keyframes (peaks and valleys)
        print("Detecting keyframes...")
        peaks, valleys, keyframes = detect_peaks_and_valleys(scores, 
                                                           prominence=args.prominence, 
                                                           distance=args.distance,
                                                           max_keyframes=args.max_keyframes)
        
        # Plot results with keyframes
        video_name = os.path.basename(args.video)
        plot_dists_scores(scores, args.output, video_name, keyframes)
        
        # Save keyframes if requested
        if args.save_keyframes and keyframes:
            keyframes_file = save_keyframes_info(keyframes, args.video)
            print(f"Keyframes saved to: {keyframes_file}")
        
        # Save keyframe images if requested
        if args.save_images and keyframes:
            keyframes_dir = save_keyframe_images(frames, keyframes, args.video)
            print(f"Keyframe images saved to: {keyframes_dir}")
        
        # Create timeline visualization if requested
        if args.create_timeline and keyframes:
            # Get video FPS for timeline
            cap = cv2.VideoCapture(args.video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            timeline_path = create_timeline_visualization(
                frames, keyframes, args.video, 
                fps=fps,
                thumb_size=tuple(args.thumb_size),
                cols=args.timeline_cols
            )
            if timeline_path:
                print(f"Timeline visualization saved to: {timeline_path}")
        
        print(f"\nAnalysis complete!")
        print(f"Processed {len(frames)} frames")
        print(f"Average DISTS score: {np.mean(scores):.4f}")
        print(f"Score range: {np.min(scores):.4f} - {np.max(scores):.4f}")
        print(f"\nKeyframes detected:")
        print(f"  Total keyframes: {len(keyframes)}")
        print(f"  Peaks: {len([kf for kf in keyframes if kf['type'] == 'peak'])}")
        print(f"  Valleys: {len([kf for kf in keyframes if kf['type'] == 'valley'])}")
        
        if keyframes:
            print(f"\nKeyframe details:")
            for kf in keyframes[:10]:  # Show first 10 keyframes
                print(f"  Frame {kf['frame']:3d}: {kf['type']:6s} (score: {kf['score']:.4f}, prominence: {kf['prominence']:.4f})")
            if len(keyframes) > 10:
                print(f"  ... and {len(keyframes) - 10} more keyframes")
            
            # Print frame numbers for easy copying
            frame_numbers = [kf['frame'] for kf in keyframes]
            print(f"\nKeyframe list: {frame_numbers}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())