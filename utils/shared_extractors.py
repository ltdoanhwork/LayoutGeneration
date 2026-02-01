"""
Shared Utilities Module

Centralized utilities used across multiple pipelines:
- CLIPExtractor: Feature extraction using OpenAI CLIP
- MultiPromptScorer: Anime attribute scoring

Used by:
- run_inference_v11.py (DSN keyframe selection)
- objectfree_pipeline.py (character detection & retrieval)
- anime_seg_detector.py (character feature extraction)
"""

from scripts.precompute_script.precompute_all_v11 import (
    CLIPExtractor,
    MultiPromptScorer,
    normalize_and_merge_scenes,
    adaptive_stride,
    decode_scene_frames
)

__all__ = [
    'CLIPExtractor',
    'MultiPromptScorer',
    'normalize_and_merge_scenes',
    'adaptive_stride',
    'decode_scene_frames'
]

"""
Usage:

From any module:
    from utils.shared_extractors import CLIPExtractor, MultiPromptScorer
    
    # Initialize
    clip_extractor = CLIPExtractor(device='cuda:0')
    anime_scorer = MultiPromptScorer(device='cuda:0')
    
    # Extract features from frames
    clip_features = clip_extractor.extract(frames)  # (N, 512)
    anime_attrs = anime_scorer.score_frames(frames) # (N, 6)
    
    # Combine
    combined_features = np.concatenate([clip_features, anime_attrs], axis=1)  # (N, 518)
"""
