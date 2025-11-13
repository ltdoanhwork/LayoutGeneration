#!/usr/bin/env python3
"""
Advanced BBox Refinement Testing - Compare different parameters
"""

import cv2
import numpy as np
import glob
from pathlib import Path
from bbox_refinement import BBoxRefinement


def test_parameter_sensitivity():
    """Test how different parameters affect refinement"""
    
    # Get test image
    test_folder = "data/samples/keyframe4check/23670_keyframes"
    images = sorted(glob.glob(f"{test_folder}/*.jpg"))[:1]
    
    if not images:
        print("[ERROR] No test images found")
        return
    
    img_path = images[0]
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    # Single test bbox (from SAM2)
    test_bbox = [int(w*0.3), int(h*0.2), int(w*0.7), int(h*0.7)]
    
    refiner = BBoxRefinement()
    output_dir = "debug_output_refinement_comparison"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Test configurations
    configs = [
        {
            'name': 'Conservative (eta=0.95, search=10%)',
            'params': {
                'weights': (0.4, 0.4, 0.2),
                'lambda_': 10,
                'eta': 0.95,
                'search_margin': 0.1,
                'expansion_margin': 0.05,
            }
        },
        {
            'name': 'Balanced (eta=0.90, search=20%)',
            'params': {
                'weights': (0.4, 0.4, 0.2),
                'lambda_': 10,
                'eta': 0.90,
                'search_margin': 0.2,
                'expansion_margin': 0.1,
            }
        },
        {
            'name': 'Aggressive (eta=0.80, search=30%)',
            'params': {
                'weights': (0.4, 0.4, 0.2),
                'lambda_': 10,
                'eta': 0.80,
                'search_margin': 0.3,
                'expansion_margin': 0.15,
            }
        },
        {
            'name': 'Edge-focused (w_edge=0.5)',
            'params': {
                'weights': (0.3, 0.2, 0.5),
                'lambda_': 10,
                'eta': 0.90,
                'search_margin': 0.2,
                'expansion_margin': 0.1,
            }
        },
        {
            'name': 'Color-focused (w_color=0.5)',
            'params': {
                'weights': (0.25, 0.5, 0.25),
                'lambda_': 10,
                'eta': 0.90,
                'search_margin': 0.2,
                'expansion_margin': 0.1,
            }
        },
    ]
    
    results = []
    
    print(f"\n[TEST] Original bbox: {test_bbox}")
    print(f"  Size: {test_bbox[2]-test_bbox[0]} x {test_bbox[3]-test_bbox[1]}")
    print(f"\n{'='*70}")
    
    for config_idx, config in enumerate(configs):
        name = config['name']
        params = config['params']
        
        refined, heatmap, info = refiner.refine_bbox(
            image, test_bbox,
            **params
        )
        
        expansion = info['expansion_ratio']
        
        print(f"\n[Config {config_idx+1}] {name}")
        print(f"  Original: {test_bbox}")
        print(f"  Refined:  {refined}")
        print(f"  Expansion: {expansion:.2f}x ({(expansion-1)*100:.1f}%)")
        print(f"  New size: {refined[2]-refined[0]} x {refined[3]-refined[1]}")
        
        results.append({
            'config': name,
            'original_bbox': test_bbox,
            'refined_bbox': refined,
            'expansion_ratio': expansion,
        })
        
        # Visualize
        viz_path = f"{output_dir}/{config_idx:02d}_{name.replace(' ', '_').lower()}.png"
        refiner.visualize_refinement(image, test_bbox, refined, heatmap, viz_path)
    
    print(f"\n{'='*70}")
    print(f"[SUMMARY] All visualizations saved to: {output_dir}")
    print(f"\nExpansion ratio ranking:")
    for idx, r in enumerate(sorted(results, key=lambda x: x['expansion_ratio']), 1):
        print(f"  {idx}. {r['config']}: {r['expansion_ratio']:.2f}x")


def test_on_multiple_images():
    """Test refinement on all keyframes"""
    
    test_folder = "data/samples/keyframe4check/23670_keyframes"
    images = sorted(glob.glob(f"{test_folder}/*.jpg"))
    images = [f for f in images if "preview" not in f.lower()]
    
    print(f"\n[BATCH TEST] Testing on {len(images)} images")
    
    refiner = BBoxRefinement()
    output_dir = "debug_output_batch_refinement"
    Path(output_dir).mkdir(exist_ok=True)
    
    results = []
    
    for img_idx, img_path in enumerate(images[:5]):  # First 5 images
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        
        # Create multiple test bboxes
        test_bboxes = [
            [int(w*0.1), int(h*0.1), int(w*0.4), int(h*0.4)],
            [int(w*0.5), int(h*0.2), int(w*0.85), int(h*0.6)],
        ]
        
        img_name = Path(img_path).stem
        
        for bbox_idx, bbox in enumerate(test_bboxes):
            refined, heatmap, info = refiner.refine_bbox(
                image, bbox,
                weights=(0.4, 0.4, 0.2),
                lambda_=10,
                eta=0.90,
                search_margin=0.2,
                expansion_margin=0.1,
            )
            
            expansion = info['expansion_ratio']
            results.append({
                'image': img_name,
                'bbox_idx': bbox_idx,
                'expansion_ratio': expansion,
                'original_area': (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]),
                'refined_area': (refined[2]-refined[0]) * (refined[3]-refined[1]),
            })
            
            print(f"  {img_name} BBox{bbox_idx}: {expansion:.2f}x expansion")
            
            viz_path = f"{output_dir}/{img_idx:02d}_bbox{bbox_idx}_refinement.png"
            refiner.visualize_refinement(image, bbox, refined, heatmap, viz_path)
    
    # Summary
    print(f"\n[BATCH SUMMARY]")
    avg_expansion = np.mean([r['expansion_ratio'] for r in results])
    print(f"  Average expansion: {avg_expansion:.2f}x")
    print(f"  Min: {min([r['expansion_ratio'] for r in results]):.2f}x")
    print(f"  Max: {max([r['expansion_ratio'] for r in results]):.2f}x")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED BBOX REFINEMENT TESTING")
    print("="*70)
    
    # Test 1: Parameter sensitivity
    test_parameter_sensitivity()
    
    # Test 2: Batch processing
    test_on_multiple_images()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE ✓")
    print("="*70)
