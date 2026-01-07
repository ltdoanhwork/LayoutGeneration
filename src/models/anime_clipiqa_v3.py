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
import cv2

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
        model_name: str = "ViT-B/32", # Default to OpenAI CLIP name
        prompt_groups: Optional[List[str]] = None,
        data_range: float = 255.0,
    ):
        """
        Initialize Anime CLIP-IQA module using OpenAI CLIP directly.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            model_name: CLIP model variant to use (e.g. "ViT-B/32")
            prompt_groups: List of prompt keys to use. If None, uses default set.
            data_range: Maximum value of input images (255 for uint8, 1.0 for float)
        """
        import clip
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.data_range = data_range
        
        print(f"[AnimeClipIQA] Loading OpenAI CLIP: {model_name} on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()
        
        # Default prompt groups: focus on key anime quality aspects
        if prompt_groups is None:
            prompt_groups = [
                "quality", "sharpness", "colorfulness", "brightness",
                "sakuga", "cinematic", "composition"
            ]
        
        self.prompt_groups = prompt_groups
        self.prompts_dict = {key: ANIME_PROMPTS[key] for key in prompt_groups}
        
        # Precompute text embeddings
        # prompts_dict is {key: (pos_text, neg_text)}
        # We process each key: compute embedding for [pos, neg]
        self.text_embeds = {}
        
        print("[AnimeClipIQA] Precomputing text embeddings...")
        with torch.no_grad():
            for key, (pos_text, neg_text) in self.prompts_dict.items():
                # Tokenize: [pos, neg]
                text_tokens = clip.tokenize([pos_text, neg_text]).to(self.device)
                # Encode
                text_features = self.model.encode_text(text_tokens) # (2, D)
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_embeds[key] = text_features
                
        print(f"[AnimeClipIQA] Initialized with model: {model_name}")
        print(f"[AnimeClipIQA] Using {len(self.prompt_groups)} prompt groups: {prompt_groups}")
    
    def preprocess_images(
        self, 
        images: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> torch.Tensor:
        """
        Preprocess images using CLIP's preprocessor.
        
        Args:
            images: Images in one of the following formats:
                - np.ndarray: (H, W, 3) BGR uint8 or (N, H, W, 3) batch
                - torch.Tensor: (C, H, W) or (N, C, H, W) float/byte
                - List[np.ndarray]: List of (H, W, 3) BGR images
        
        Returns:
            torch.Tensor: (N, 3, 224, 224) preprocessed batch on device
        """
        from PIL import Image
        
        # Standardize to List[PIL.Image]
        pil_images = []
        
        if isinstance(images, list):
            # List of numpy arrays
            for img in images:
                if isinstance(img, np.ndarray):
                    # BGR -> RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img
                    pil_images.append(Image.fromarray(img_rgb))
                elif isinstance(img, torch.Tensor):
                    # Tensor to PIL
                    pil_images.append(self._tensor_to_pil(img))
                    
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = images[np.newaxis, ...]
            for i in range(images.shape[0]):
                img = images[i]
                # BGR -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img
                pil_images.append(Image.fromarray(img_rgb))
                
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            for i in range(images.shape[0]):
                pil_images.append(self._tensor_to_pil(images[i]))
                
        else:
            raise TypeError(f"Unsupported image type: {type(images)}")
        
        # Apply CLIP preprocess
        # preprocess returns a tensor (3, 224, 224)
        processed = [self.preprocess(img) for img in pil_images]
        batch = torch.stack(processed).to(self.device)
        
        return batch

    def _tensor_to_pil(self, tensor: torch.Tensor):
        from PIL import Image
        # Expects (C, H, W)
        if tensor.ndim == 3:
            # If float in [0, 1] or [0, 255]
            arr = tensor.cpu().numpy()
            if arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0) # CHW -> HWC
            
            # Normalize to [0, 255] uint8
            if self.data_range == 1.0 and arr.max() <= 1.0:
                 arr = (arr * 255.0).astype(np.uint8)
            else:
                 arr = arr.astype(np.uint8)
            return Image.fromarray(arr)
        return None
    
    def compute_batch_scores(
        self, 
        images: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute quality scores for a batch of images.
        
        Args:
            images: Batch of images
        
        Returns:
            Dict mapping prompt group names to numpy arrays of scores (N,)
        """
        images_tensor = self.preprocess_images(images) # (N, 3, 224, 224)
        
        with torch.no_grad():
            # Encode images
            image_features = self.model.encode_image(images_tensor) # (N, D)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute scores for each prompt pair
            results = {}
            logit_scale = self.model.logit_scale.exp()
            
            for key, text_features in self.text_embeds.items():
                # text_features: (2, D) -> [pos, neg]
                
                # Similarity: (N, D) @ (D, 2) -> (N, 2)
                logits = logit_scale * image_features @ text_features.t()
                
                # Softmax to get probabilities
                probs = logits.softmax(dim=-1) # (N, 2)
                
                # Probability of "positive" class (index 0)
                pos_probs = probs[:, 0].cpu().numpy()
                
                results[key] = pos_probs
                
        return results
    
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
    model_name: str = "ViT-B/32",
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
