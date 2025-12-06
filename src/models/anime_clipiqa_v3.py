#!/usr/bin/env python3
"""
Anime CLIP-IQA Module (Version 3)

This module provides a standardized interface for assessing anime image quality
using the torchmetrics CLIP-IQA implementation with custom anime-specific prompts.

Key Features:
- Uses torchmetrics.multimodal.CLIPImageQualityAssessment for robust implementation
- Defines anime-specific quality prompts (sakuga, composition, visual quality)
- Supports batch processing for efficiency
- Compatible with existing training pipeline

Author: Version 3 Enhancement
Date: 2025-12-06
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from pathlib import Path

try:
    from torchmetrics.multimodal import CLIPImageQualityAssessment
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("[Warning] torchmetrics not available. Install with: pip install torchmetrics[multimodal]")


# Anime-specific quality prompts (positive, negative pairs)
ANIME_PROMPTS = {
    # Visual Quality Attributes
    "quality": ("High quality anime frame.", "Low quality anime frame."),
    "sharpness": ("Sharp anime frame.", "Blurry anime frame."),
    "colorfulness": ("Vibrant anime colors.", "Dull anime colors."),
    "brightness": ("Well-lit anime scene.", "Dark anime scene."),
    
    # Animation Quality (Sakuga)
    "sakuga": ("High sakuga animation frame.", "Low sakuga animation frame."),
    "motion_quality": ("Smooth animation frame.", "Choppy animation frame."),
    "dynamic": ("Dynamic anime action.", "Static anime frame."),
    
    # Composition & Cinematography
    "composition": ("Well-composed anime shot.", "Poorly-composed anime shot."),
    "cinematic": ("Cinematic anime shot.", "Plain anime shot."),
    "framing": ("Well-framed anime scene.", "Poorly-framed anime scene."),
    
    # Character & Expression
    "character_detail": ("Detailed anime character.", "Simple anime character."),
    "expression": ("Expressive anime face.", "Bland anime face."),
    
    # Narrative & Storytelling
    "story_moment": ("Important story moment.", "Filler anime frame."),
    "emotional": ("Emotional anime scene.", "Neutral anime scene."),
}


class AnimeClipIQA:
    """
    Anime-specific CLIP Image Quality Assessment module.
    
    This class wraps torchmetrics CLIPImageQualityAssessment with anime-specific
    prompts and provides a unified interface for quality scoring.
    
    Usage:
        >>> iqa = AnimeClipIQA(device="cuda")
        >>> # Score single image
        >>> scores = iqa.compute_frame_scores(image_tensor)
        >>> # Score batch of images
        >>> batch_scores = iqa.compute_batch_scores(image_batch)
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "openai/clip-vit-base-patch32",
        prompt_groups: Optional[List[str]] = None,
        data_range: float = 255.0,
    ):
        """
        Initialize Anime CLIP-IQA module.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            model_name: CLIP model variant to use
            prompt_groups: List of prompt keys to use. If None, uses default set.
            data_range: Maximum value of input images (255 for uint8, 1.0 for float)
        """
        if not TORCHMETRICS_AVAILABLE:
            raise ImportError(
                "torchmetrics with multimodal support required. "
                "Install with: pip install torchmetrics[multimodal]"
            )
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.data_range = data_range
        
        # Default prompt groups: focus on key anime quality aspects
        if prompt_groups is None:
            prompt_groups = [
                "quality", "sharpness", "colorfulness", "brightness",
                "sakuga", "cinematic", "composition"
            ]
        
        self.prompt_groups = prompt_groups
        
        # Build prompt tuple for torchmetrics
        self.prompts = tuple(ANIME_PROMPTS[key] for key in prompt_groups)
        
        # Initialize CLIP-IQA metric
        self.metric = CLIPImageQualityAssessment(
            model_name_or_path=model_name,
            data_range=data_range,
            prompts=self.prompts
        ).to(self.device)
        
        print(f"[AnimeClipIQA] Initialized with model: {model_name}")
        print(f"[AnimeClipIQA] Using {len(self.prompt_groups)} prompt groups: {prompt_groups}")
    
    def preprocess_images(
        self, 
        images: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> torch.Tensor:
        """
        Preprocess images to the format expected by CLIP-IQA.
        
        Args:
            images: Images in one of the following formats:
                - np.ndarray: (H, W, 3) BGR uint8 or (N, H, W, 3) batch
                - torch.Tensor: (C, H, W) or (N, C, H, W)
                - List[np.ndarray]: List of (H, W, 3) BGR images
        
        Returns:
            torch.Tensor: (N, 3, H, W) float tensor in range [0, data_range]
        """
        if isinstance(images, list):
            # Convert list of numpy arrays to batch
            images = np.stack(images, axis=0)
        
        if isinstance(images, np.ndarray):
            # Handle BGR to RGB conversion
            if images.ndim == 3:  # Single image (H, W, 3)
                images = images[np.newaxis, ...]  # Add batch dimension
            
            # Convert BGR to RGB
            images = images[..., ::-1].copy()  # BGR -> RGB
            
            # Convert to tensor and transpose to (N, C, H, W)
            images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:  # Single image (C, H, W)
                images = images.unsqueeze(0)  # Add batch dimension
        
        else:
            raise TypeError(f"Unsupported image type: {type(images)}")
        
        # Move to device
        images = images.to(self.device)
        
        # Ensure values are in [0, data_range]
        if images.max() <= 1.0 and self.data_range == 255.0:
            images = images * 255.0
        
        return images
    
    def compute_batch_scores(
        self, 
        images: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute quality scores for a batch of images.
        
        Args:
            images: Batch of images (see preprocess_images for formats)
        
        Returns:
            Dict mapping prompt group names to numpy arrays of scores (N,)
        """
        images_tensor = self.preprocess_images(images)
        
        with torch.no_grad():
            scores_dict = self.metric(images_tensor)
        
        # Convert to numpy and map back to prompt group names
        result = {}
        for idx, key in enumerate(self.prompt_groups):
            user_key = f"user_defined_{idx}"
            if user_key in scores_dict:
                result[key] = scores_dict[user_key].cpu().numpy()
            elif key in scores_dict:
                result[key] = scores_dict[key].cpu().numpy()
        
        return result
    
    def compute_frame_scores(
        self, 
        image: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute quality scores for a single image.
        
        Args:
            image: Single image (H, W, 3) or (3, H, W)
        
        Returns:
            Dict mapping prompt group names to float scores
        """
        batch_scores = self.compute_batch_scores(image)
        return {k: float(v[0]) for k, v in batch_scores.items()}
    
    def compute_aggregated_scores(
        self,
        images: Union[np.ndarray, torch.Tensor, List[np.ndarray]],
        aggregation: str = "mean"
    ) -> Dict[str, float]:
        """
        Compute aggregated scores across multiple images.
        
        Args:
            images: Batch of images
            aggregation: Aggregation method ('mean', 'median', 'max', 'min')
        
        Returns:
            Dict mapping prompt group names to aggregated float scores
        """
        batch_scores = self.compute_batch_scores(images)
        
        agg_fn = {
            "mean": np.mean,
            "median": np.median,
            "max": np.max,
            "min": np.min,
        }.get(aggregation, np.mean)
        
        return {k: float(agg_fn(v)) for k, v in batch_scores.items()}
    
    def get_legacy_format_scores(
        self,
        images: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Get scores in legacy format (N, 6) for backward compatibility.
        
        Legacy format: [sharpness, colorfulness, brightness, sakuga, cinematic, expression]
        
        Args:
            images: Batch of images
        
        Returns:
            np.ndarray: (N, 6) array of scores
        """
        batch_scores = self.compute_batch_scores(images)
        
        # Map to legacy indices (use defaults if not available)
        legacy_keys = ["sharpness", "colorfulness", "brightness", "sakuga", "cinematic", "composition"]
        N = len(next(iter(batch_scores.values())))
        
        result = np.zeros((N, 6), dtype=np.float32)
        for i, key in enumerate(legacy_keys):
            if key in batch_scores:
                result[:, i] = batch_scores[key]
            else:
                # Fill with neutral score if prompt not available
                result[:, i] = 0.5
        
        return result


def create_anime_clipiqa(
    device: str = "cuda",
    model_name: str = "openai/clip-vit-base-patch32",
    **kwargs
) -> AnimeClipIQA:
    """
    Factory function to create AnimeClipIQA instance.
    
    Args:
        device: Device to run on
        model_name: CLIP model variant
        **kwargs: Additional arguments for AnimeClipIQA
    
    Returns:
        AnimeClipIQA instance
    """
    return AnimeClipIQA(device=device, model_name=model_name, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("=== Anime CLIP-IQA V3 Demo ===\n")
    
    # Create dummy test images
    test_images = torch.randint(0, 256, (4, 3, 224, 224)).float()
    print(f"Test batch shape: {test_images.shape}\n")
    
    # Initialize module
    iqa = AnimeClipIQA(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Compute batch scores
    print("Computing batch scores...")
    scores = iqa.compute_batch_scores(test_images)
    
    print("\nResults:")
    for key, values in scores.items():
        print(f"  {key:20s}: mean={values.mean():.3f}, std={values.std():.3f}")
    
    # Compute single frame scores
    print("\nSingle frame scores:")
    single_scores = iqa.compute_frame_scores(test_images[0])
    for key, value in single_scores.items():
        print(f"  {key:20s}: {value:.3f}")
    
    # Legacy format
    print("\nLegacy format (N, 6):")
    legacy_scores = iqa.get_legacy_format_scores(test_images)
    print(f"  Shape: {legacy_scores.shape}")
    print(f"  First row: {legacy_scores[0]}")
