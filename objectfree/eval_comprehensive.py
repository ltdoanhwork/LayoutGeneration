"""
Comprehensive evaluation of image regions using multiple metrics:
1. Entropy - measures complexity/information content
2. Edge Detection (Roberts/Sobel/Laplacian) - detects strokes/lines
3. Color Variance - measures color diversity
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy import ndimage
import os


class ImageRegionAnalyzer:
    """Analyze image regions using multiple quality metrics"""
    
    def __init__(self, image_path, device=None):
        """
        Initialize analyzer with image
        
        Parameters:
        - image_path: path to image file
        - device: torch device (cuda/cpu)
        """
        self.image_path = image_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load image
        self.image_color = plt.imread(image_path)
        
        # Convert to grayscale
        if len(self.image_color.shape) == 3:
            self.image_gray = np.mean(self.image_color, axis=2).astype(np.uint8)
        else:
            self.image_gray = self.image_color
            
        print(f"Loaded image: {image_path}")
        print(f"Image shape: {self.image_color.shape}")
        print(f"Using device: {self.device}")
    
    def compute_local_entropy(self, window_size=20):
        """
        Compute local entropy map using GPU
        Higher entropy = more complex/detailed regions
        """
        print(f"\n[1/4] Computing local entropy (window_size={window_size})...")
        
        image = self.image_gray
        if len(image.shape) != 2:
            raise ValueError("Image must be grayscale (2D array).")
        
        pad = window_size // 2
        image_padded = np.pad(image, pad, mode='constant')
        image_t = torch.from_numpy(image_padded).float().to(self.device)
        
        # Unfold to get sliding windows
        patches = image_t.unfold(0, window_size, 1).unfold(1, window_size, 1)
        
        # Slice to match the number of original pixels
        h, w = image.shape
        patches = patches[:h, :w, :, :]
        
        patch_size = window_size ** 2
        num_patches = h * w
        
        # Flatten patches
        patches_flat = patches.reshape(num_patches, patch_size).long()
        
        # Compute histograms using scatter_add
        bins = 256
        hist = torch.zeros(num_patches, bins, device=self.device)
        ones = torch.ones_like(patches_flat, dtype=torch.float, device=self.device)
        hist.scatter_add_(1, patches_flat, ones)
        
        # Normalize to probabilities
        hist += 1e-10
        prob = hist / hist.sum(dim=1, keepdim=True)
        
        # Compute entropy
        entropy = - (prob * torch.log2(prob)).sum(dim=1)
        
        # Reshape back to image shape
        entropy_map = entropy.reshape(h, w).cpu().numpy()
        
        print(f"   Entropy range: [{entropy_map.min():.2f}, {entropy_map.max():.2f}]")
        return entropy_map
    
    def compute_edge_density(self, method='laplacian', kernel_size=3):
        """
        Compute edge density map using various edge detection methods
        Higher value = more strokes/lines/edges
        
        Parameters:
        - method: 'roberts', 'sobel', 'laplacian', 'canny'
        - kernel_size: size of the kernel (for sobel/laplacian)
        """
        print(f"\n[2/4] Computing edge density using {method} (kernel_size={kernel_size})...")
        
        image = self.image_gray
        
        if method == 'roberts':
            # Roberts Cross operator
            roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            
            edge_x = ndimage.convolve(image.astype(float), roberts_x)
            edge_y = ndimage.convolve(image.astype(float), roberts_y)
            edge_map = np.sqrt(edge_x**2 + edge_y**2)
            
        elif method == 'sobel':
            # Sobel operator
            edge_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
            edge_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
            edge_map = np.sqrt(edge_x**2 + edge_y**2)
            
        elif method == 'laplacian':
            # Laplacian operator (second derivative)
            edge_map = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
            edge_map = np.abs(edge_map)
            
        elif method == 'canny':
            # Canny edge detection
            edge_map = cv2.Canny(image, 50, 150)
            edge_map = edge_map.astype(float)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply local averaging to get density
        edge_density = gaussian_filter(edge_map, sigma=5)
        
        print(f"   Edge density range: [{edge_density.min():.2f}, {edge_density.max():.2f}]")
        return edge_density
    
    def compute_color_variance(self, window_size=20):
        """
        Compute local color variance map
        Higher variance = more colorful/diverse regions
        """
        print(f"\n[3/4] Computing color variance (window_size={window_size})...")
        
        if len(self.image_color.shape) != 3:
            print("   Warning: Grayscale image, using intensity variance")
            image = self.image_gray
            h, w = image.shape
            
            # Compute local variance using uniform filter
            mean_map = ndimage.uniform_filter(image.astype(float), size=window_size)
            mean_sq_map = ndimage.uniform_filter(image.astype(float)**2, size=window_size)
            variance_map = mean_sq_map - mean_map**2
            
        else:
            # Color image - compute variance across all channels
            image = self.image_color
            h, w, c = image.shape
            
            variance_maps = []
            for channel in range(c):
                ch_img = image[:, :, channel]
                mean_map = ndimage.uniform_filter(ch_img.astype(float), size=window_size)
                mean_sq_map = ndimage.uniform_filter(ch_img.astype(float)**2, size=window_size)
                var_map = mean_sq_map - mean_map**2
                variance_maps.append(var_map)
            
            # Average variance across channels
            variance_map = np.mean(variance_maps, axis=0)
        
        print(f"   Color variance range: [{variance_map.min():.2f}, {variance_map.max():.2f}]")
        return variance_map
    
    def compute_combined_score(self, weights=None, sigma=10):
        """
        Compute combined quality score from all metrics
        
        Parameters:
        - weights: dict with keys 'entropy', 'edge', 'color' (default: equal weights)
        - sigma: smoothing parameter
        
        Returns:
        - combined_score: weighted combination of all metrics
        - individual_maps: dict of individual normalized maps
        """
        print(f"\n[4/4] Computing combined quality score...")
        
        if weights is None:
            weights = {'entropy': 0.4, 'edge': 0.4, 'color': 0.2}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        print(f"   Weights: {weights}")
        
        # Compute all metrics
        entropy_map = self.compute_local_entropy()
        edge_map = self.compute_edge_density(method='laplacian')
        color_map = self.compute_color_variance()
        
        # Normalize all maps to [0, 1]
        def normalize(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min < 1e-10:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)
        
        entropy_norm = normalize(entropy_map)
        edge_norm = normalize(edge_map)
        color_norm = normalize(color_map)
        
        # Combine with weights
        combined = (weights['entropy'] * entropy_norm + 
                   weights['edge'] * edge_norm + 
                   weights['color'] * color_norm)
        
        # Smooth the combined score
        combined_smooth = gaussian_filter(combined, sigma=sigma)
        
        print(f"   Combined score range: [{combined_smooth.min():.3f}, {combined_smooth.max():.3f}]")
        
        return combined_smooth, {
            'entropy': entropy_norm,
            'edge': edge_norm,
            'color': color_norm
        }
    
    def find_key_regions(self, score_map, n_regions=5, min_distance=50):
        """
        Find key regions (local maxima) in the score map
        
        Parameters:
        - score_map: 2D score map
        - n_regions: number of regions to find
        - min_distance: minimum distance between regions (in pixels)
        
        Returns:
        - regions: list of (x, y, score) tuples
        """
        print(f"\nFinding top {n_regions} key regions...")
        
        # Find local maxima
        footprint = np.ones((min_distance//3, min_distance//3))
        local_max = score_map == maximum_filter(score_map, footprint=footprint)
        
        # Threshold to select significant peaks
        threshold = 0.6 * score_map.max()
        significant_peaks = np.where((local_max) & (score_map > threshold))
        
        # Get coordinates and scores
        coords = list(zip(significant_peaks[1], significant_peaks[0]))
        scores = [score_map[y, x] for x, y in coords]
        
        # Sort by score
        regions = sorted(zip(coords, scores), key=lambda x: x[1], reverse=True)
        
        # Apply non-maximum suppression based on distance
        filtered_regions = []
        for (x, y), score in regions:
            # Check distance to all already selected regions
            too_close = False
            for (x2, y2), _ in filtered_regions:
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_regions.append(((x, y), score))
            
            if len(filtered_regions) >= n_regions:
                break
        
        print(f"   Found {len(filtered_regions)} regions")
        for i, ((x, y), score) in enumerate(filtered_regions):
            print(f"   Region {i+1}: ({x}, {y}) - score: {score:.3f}")
        
        return filtered_regions
    
    def visualize_and_save(self, output_dir='outputs_eval'):
        """
        Create comprehensive visualization of all metrics
        """
        print(f"\nGenerating visualizations...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute combined score
        combined_score, individual_maps = self.compute_combined_score()
        
        # Find key regions
        key_regions = self.find_key_regions(combined_score, n_regions=5)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(self.image_color if len(self.image_color.shape) == 3 else self.image_gray, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Entropy map
        im1 = axes[0, 1].imshow(individual_maps['entropy'], cmap='hot')
        axes[0, 1].set_title('Entropy Map\n(Complexity/Detail)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # Edge density map
        im2 = axes[0, 2].imshow(individual_maps['edge'], cmap='hot')
        axes[0, 2].set_title('Edge Density Map\n(Strokes/Lines)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # Color variance map
        im3 = axes[1, 0].imshow(individual_maps['color'], cmap='hot')
        axes[1, 0].set_title('Color Variance Map\n(Color Diversity)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Combined score map
        im4 = axes[1, 1].imshow(combined_score, cmap='hot')
        axes[1, 1].set_title('Combined Quality Score\n(Weighted Average)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        
        # Original with key regions marked
        axes[1, 2].imshow(self.image_color if len(self.image_color.shape) == 3 else self.image_gray, cmap='gray')
        for i, ((x, y), score) in enumerate(key_regions):
            axes[1, 2].plot(x, y, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2)
            axes[1, 2].text(x + 20, y, f'R{i+1}\n{score:.2f}', 
                          color='red', fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1, 2].set_title('Key Regions\n(Top Quality Areas)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 'comprehensive_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {output_path}")
        plt.close()
        
        # Save individual maps
        for name, map_data in individual_maps.items():
            map_path = os.path.join(output_dir, f'{name}_map.png')
            plt.imsave(map_path, map_data, cmap='hot')
            print(f"   Saved: {map_path}")
        
        # Save combined score
        combined_path = os.path.join(output_dir, 'combined_score.png')
        plt.imsave(combined_path, combined_score, cmap='hot')
        print(f"   Saved: {combined_path}")
        
        # Save statistics
        stats = {
            'image_path': self.image_path,
            'image_shape': self.image_color.shape,
            'metrics': {
                'entropy': {
                    'mean': float(individual_maps['entropy'].mean()),
                    'std': float(individual_maps['entropy'].std()),
                    'min': float(individual_maps['entropy'].min()),
                    'max': float(individual_maps['entropy'].max())
                },
                'edge_density': {
                    'mean': float(individual_maps['edge'].mean()),
                    'std': float(individual_maps['edge'].std()),
                    'min': float(individual_maps['edge'].min()),
                    'max': float(individual_maps['edge'].max())
                },
                'color_variance': {
                    'mean': float(individual_maps['color'].mean()),
                    'std': float(individual_maps['color'].std()),
                    'min': float(individual_maps['color'].min()),
                    'max': float(individual_maps['color'].max())
                },
                'combined_score': {
                    'mean': float(combined_score.mean()),
                    'std': float(combined_score.std()),
                    'min': float(combined_score.min()),
                    'max': float(combined_score.max())
                }
            },
            'key_regions': [
                {'position': (int(x), int(y)), 'score': float(score)}
                for (x, y), score in key_regions
            ]
        }
        
        import json
        stats_path = os.path.join(output_dir, 'analysis_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   Saved: {stats_path}")
        
        return combined_score, individual_maps, key_regions


def main():
    """Example usage"""
    image_path = '/home/serverai/ltdoanh/LayoutGeneration/samples/keyframe4check/14653_keyframes/0002_clip11_frame071_14653_11.jpg'
    
    print("="*70)
    print("COMPREHENSIVE IMAGE REGION ANALYSIS")
    print("="*70)
    
    # Create analyzer
    analyzer = ImageRegionAnalyzer(image_path)
    
    # Run analysis and save results
    combined_score, maps, regions = analyzer.visualize_and_save(output_dir='outputs_eval')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nSummary:")
    print(f"  • Entropy: measures complexity/information content")
    print(f"  • Edge Density: detects strokes/lines (using Laplacian)")
    print(f"  • Color Variance: measures color diversity")
    print(f"  • Combined Score: weighted combination of all metrics")
    print(f"\nKey Regions (sorted by quality):")
    for i, ((x, y), score) in enumerate(regions):
        print(f"  {i+1}. Position: ({x:4d}, {y:4d}) - Score: {score:.3f}")


if __name__ == "__main__":
    main()
