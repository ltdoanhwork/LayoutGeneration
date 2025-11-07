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

def plot_dists_scores(scores, save_path=None, video_name="Video"):
    """
    Plot DISTS scores over frame sequence
    
    Args:
        scores (list): DISTS scores
        save_path (str): Path to save plot (optional)
        video_name (str): Name of video for title
    """
    plt.figure(figsize=(12, 6))
    
    frame_numbers = list(range(2, len(scores) + 2))  # Start from frame 2 (comparison with frame 1)
    
    plt.plot(frame_numbers, scores, 'b-', linewidth=2, marker='o', markersize=3)
    plt.title(f'DISTS Scores Over Time - {video_name}', fontsize=14)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('DISTS Score (vs Frame 1)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    plt.axhline(y=mean_score, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_score:.4f}')
    
    # Add text box with statistics
    stats_text = f'Statistics:\nMean: {mean_score:.4f}\nStd: {std_score:.4f}\nMin: {min_score:.4f}\nMax: {max_score:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze video using DISTS metric')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process')
    parser.add_argument('--output', type=str, default=None, help='Path to save plot (optional)')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'], 
                       help='Device to use for computation')
    
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
        
        # Plot results
        video_name = os.path.basename(args.video)
        plot_dists_scores(scores, args.output, video_name)
        
        print(f"\nAnalysis complete!")
        print(f"Processed {len(frames)} frames")
        print(f"Average DISTS score: {np.mean(scores):.4f}")
        print(f"Score range: {np.min(scores):.4f} - {np.max(scores):.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())