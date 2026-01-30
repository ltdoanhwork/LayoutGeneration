#!/usr/bin/env python3
"""
End-to-End Video Inference Pipeline.

Takes a raw video file, performs:
1. Scene detection (pyscenedetect)
2. Frame extraction
3. CLIP feature extraction
4. Anime attribute extraction
5. Model comparison (V11, VSUMM, LLMVS)

Usage:
    python scripts/run_inference_video.py --video <path.mp4> --output_dir <dir>
"""

import sys
import os
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

import torch
import clip
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablation" / "pytorch-vsumm-reinforce"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablation" / "LLMVS"))

from src.scene_detection import create_detector, Scene
from src.models.dsn_v8 import create_dsn_v8

# Optional model imports
try:
    from models import DSN as VSUMM_DSN
except ImportError:
    VSUMM_DSN = None

try:
    from networks.model_visual import LLMVSVisual
    from llmvs_utils.configs import Config as LLMVSConfig
except ImportError:
    LLMVSVisual = None
    LLMVSConfig = None


def log(msg: str):
    print(f"[Pipeline] {msg}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Feature Extraction (from precompute_all_v11.py)
# ============================================================================

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


MULTI_PROMPTS = {
    "sharpness": [
        ("Sharp anime frame.", "Blurry anime frame."),
        ("Crisp anime artwork.", "Fuzzy anime artwork."),
        ("Clear anime image.", "Unclear anime image."),
    ],
    "colorfulness": [
        ("Vibrant anime colors.", "Dull anime colors."),
        ("Colorful anime scene.", "Desaturated anime scene."),
        ("Rich anime palette.", "Muted anime palette."),
    ],
    "brightness": [
        ("Well-lit anime scene.", "Dark anime scene."),
        ("Bright anime frame.", "Dim anime frame."),
        ("Good exposure anime.", "Underexposed anime."),
    ],
    "sakuga": [
        ("High sakuga animation frame.", "Low sakuga animation frame."),
        ("Key animation frame.", "In-between animation frame."),
        ("Fluid motion anime.", "Static anime frame."),
    ],
    "cinematic": [
        ("Cinematic anime shot.", "Plain anime shot."),
        ("Well-composed anime.", "Poorly-composed anime."),
        ("Professional anime framing.", "Amateur anime framing."),
    ],
    "expression": [
        ("Expressive anime face.", "Bland anime face."),
        ("Emotional anime character.", "Neutral anime character."),
        ("Dynamic anime expression.", "Static anime expression."),
    ],
}
ATTR_NAMES = list(MULTI_PROMPTS.keys())


class FeatureExtractor:
    """Combined CLIP feature and Anime attribute extractor."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        
        # Pre-encode attribute prompts
        self.pos_embeds = {}
        self.neg_embeds = {}
        
        with torch.no_grad():
            for attr, prompts in MULTI_PROMPTS.items():
                pos_list, neg_list = [], []
                for pos_text, neg_text in prompts:
                    pos_tok = clip.tokenize([pos_text]).to(device)
                    neg_tok = clip.tokenize([neg_text]).to(device)
                    pos_list.append(self.model.encode_text(pos_tok).float())
                    neg_list.append(self.model.encode_text(neg_tok).float())
                self.pos_embeds[attr] = torch.cat(pos_list, dim=0)
                self.neg_embeds[attr] = torch.cat(neg_list, dim=0)
        
        log("CLIP ViT-B/32 + Multi-Prompt Scorer loaded")
    
    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract L2-normalized CLIP features."""
        pil_list = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        batch = torch.stack([self.preprocess(p) for p in pil_list]).to(self.device)
        
        with torch.no_grad():
            feats = self.model.encode_image(batch).float().cpu().numpy()
        
        return l2_normalize(feats, axis=1).astype(np.float32)
    
    def extract_anime_attrs(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract anime quality attributes."""
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        batch = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        
        with torch.no_grad():
            img_feats = self.model.encode_image(batch).float()
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        
        results = {}
        for attr in ATTR_NAMES:
            pos_emb = self.pos_embeds[attr] / self.pos_embeds[attr].norm(dim=-1, keepdim=True)
            neg_emb = self.neg_embeds[attr] / self.neg_embeds[attr].norm(dim=-1, keepdim=True)
            
            pos_sim = img_feats @ pos_emb.T
            neg_sim = img_feats @ neg_emb.T
            
            logits = torch.stack([pos_sim, neg_sim], dim=-1)
            probs = torch.softmax(logits * 100, dim=-1)
            scores = probs[:, :, 0].mean(dim=1).cpu().numpy()
            results[attr] = scores
        
        return np.stack([results[attr] for attr in ATTR_NAMES], axis=1).astype(np.float32)


# ============================================================================
# Scene Detection & Frame Extraction
# ============================================================================

def normalize_and_merge_scenes(scenes: List[Scene], min_len_frames: int = 30) -> List[Scene]:
    if not scenes:
        return []
    
    norm = []
    for s in scenes:
        a, b = int(s.start_frame), int(s.end_frame)
        if b < a:
            a, b = b, a
        norm.append(Scene(a, b))
    norm.sort(key=lambda x: x.start_frame)
    
    merged = []
    for sc in norm:
        cur_len = sc.end_frame - sc.start_frame + 1
        if not merged:
            merged.append(sc)
        elif cur_len < min_len_frames:
            prev = merged[-1]
            merged[-1] = Scene(prev.start_frame, sc.end_frame)
        else:
            merged.append(sc)
    
    return merged


def adaptive_stride(scene_len: int) -> int:
    if scene_len < 100:
        return 5
    elif scene_len < 300:
        return 15
    else:
        return 25


def decode_scene_frames(
    video_path: str,
    start_frame: int,
    end_frame: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    
    frames = []
    frame_indices = []
    
    for fidx in range(start_frame, end_frame + 1, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
        frame_indices.append(fidx)
    
    cap.release()
    return frames, frame_indices


# ============================================================================
# Model Comparison
# ============================================================================

def create_keyframe_grid(frames, selected_indices, title="", max_frames=8):
    """Create visualization grid of keyframes."""
    if not frames or not selected_indices:
        return None
    
    display_indices = selected_indices[:max_frames]
    key_frames = [frames[i] for i in display_indices if i < len(frames)]
    
    if not key_frames:
        return None
    
    target_h, target_w = 180, 320
    resized = [cv2.resize(f, (target_w, target_h)) for f in key_frames]
    
    n = len(resized)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    
    grid = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
    
    for idx, frame in enumerate(resized):
        r = idx // cols
        c = idx % cols
        grid[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = frame
    
    # Add title bar
    title_bar = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return np.vstack([title_bar, grid])


def run_comparison_on_scene(
    frames: List[np.ndarray],
    feats: np.ndarray,
    anime_attrs: np.ndarray,
    models: Dict[str, Any],
    frame_indices: List[int], # Original video frame indices
    video_stem: str,
    output_dir: Path,
    budget_ratio: float = 0.05,
    device: str = "cuda",
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """Run all models on a single scene."""
    T = len(feats)
    budget = max(3, min(15, int(T * budget_ratio)))
    
    # Quality metrics
    quality = anime_attrs.mean(axis=1)
    ranks = np.argsort(np.argsort(quality))
    percentiles = ranks / max(1, T - 1)
    k10 = max(1, int(T * 0.1))
    top10_idx = set(np.argsort(quality)[-k10:])
    
    results = {"frames": T, "budget": budget, "models": {}}
    grids = []
    
    def save_individual_frames(model_name: str, selected_scene_indices: List[int]):
        model_dir = output_dir / video_stem / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        global_indices = []
        for idx in selected_scene_indices:
            if idx < len(frames):
                frame = frames[idx]
                original_idx = frame_indices[idx]
                global_indices.append(original_idx)
                
                # Zero-padded frame number (e.g. frame_001234.jpg)
                fname = f"frame_{original_idx:06d}.jpg"
                cv2.imwrite(str(model_dir / fname), frame)
        return global_indices

    # V11
    if "v11" in models:
        try:
            feats_input = np.concatenate([feats, anime_attrs], axis=1)
            feats_t = torch.from_numpy(feats_input).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs, _ = models["v11"](feats_t)
                probs = probs.squeeze(0).cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            mpr = float(np.mean(percentiles[sel_idx]))
            top10 = len(set(sel_idx) & top10_idx) / k10
            
            # Save frames
            global_idx = save_individual_frames("V11", sel_idx)
            
            results["models"]["V11"] = {
                "mpr": mpr, "top10": top10, 
                "indices": sel_idx, # relative to scene
                "global_indices": global_idx # absolute video frame indices
            }
            grids.append(create_keyframe_grid(frames, sel_idx, f"V11 (MPR={mpr:.2f})", budget))
        except Exception as e:
            log(f"V11 error: {e}")
            import traceback
            traceback.print_exc()
    
    # VSUMM
    if "vsumm" in models and models["vsumm"] is not None:
        try:
            feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs = models["vsumm"](feats_t).squeeze().cpu().numpy()
            
            sel_idx = sorted(np.argsort(probs)[-budget:].tolist())
            mpr = float(np.mean(percentiles[sel_idx]))
            top10 = len(set(sel_idx) & top10_idx) / k10
            
            # Save frames
            global_idx = save_individual_frames("VSUMM", sel_idx)

            results["models"]["VSUMM"] = {
                "mpr": mpr, "top10": top10, 
                "indices": sel_idx,
                "global_indices": global_idx
            }
            grids.append(create_keyframe_grid(frames, sel_idx, f"VSUMM (MPR={mpr:.2f})", budget))
        except Exception as e:
            log(f"VSUMM error: {e}")
    
    # LLMVS
    if "llmvs" in models and models["llmvs"] is not None:
        try:
            feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                scores = models["llmvs"](feats_t).squeeze().cpu().numpy()
            
            sel_idx = sorted(np.argsort(scores)[-budget:].tolist())
            mpr = float(np.mean(percentiles[sel_idx]))
            top10 = len(set(sel_idx) & top10_idx) / k10
            
            # Save frames
            global_idx = save_individual_frames("LLMVS", sel_idx)

            results["models"]["LLMVS"] = {
                "mpr": mpr, "top10": top10, 
                "indices": sel_idx,
                "global_indices": global_idx
            }
            grids.append(create_keyframe_grid(frames, sel_idx, f"LLMVS (MPR={mpr:.2f})", budget))
        except Exception as e:
            log(f"LLMVS error: {e}")
    
    return results, grids


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End Video Inference")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output_dir", type=str, default="demo_outputs/video_inference")
    
    # Checkpoints
    parser.add_argument("--v11_ckpt", type=str, 
        default="runs/training_v11_recerr_w0.2/best.pt")
    parser.add_argument("--vsumm_ckpt", type=str,
        default="runs/ablation_vsumm/sakuga_train/model_epoch60.pth.tar")
    parser.add_argument("--llmvs_ckpt", type=str,
        default="runs/ablation_llmvs/optionB_visual/best_model.pth")
    
    # Scene detection
    parser.add_argument("--scene_threshold", type=float, default=0.8)
    parser.add_argument("--min_scene_len", type=int, default=100)
    
    parser.add_argument("--budget_ratio", type=float, default=0.03)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        log(f"Error: Video not found: {video_path}")
        return
    
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    video_stem = video_path.stem
    log(f"Processing: {video_stem}")
    
    # ===== Step 1: Scene Detection =====
    log(f"Step 1: Scene Detection (TransNetV2, threshold={args.scene_threshold})")
    
    detector = create_detector(
        "transnetv2", 
        model_dir="src/models/TransNetV2", 
        device=args.device,
        prob_threshold=args.scene_threshold
    )
    try:
        scenes_raw = detector.detect(str(video_path))
    finally:
        detector.close()
    
    if not scenes_raw:
        log("  No scenes detected, using whole video")
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        scenes_raw = [Scene(0, total_frames - 1)]
    
    scenes = normalize_and_merge_scenes(scenes_raw, args.min_scene_len)
    log(f"  Detected {len(scenes)} scenes")
    
    # ===== Step 2: Initialize Extractors =====
    log("Step 2: Initialize Feature Extractor")
    extractor = FeatureExtractor(device=args.device)
    
    # ===== Step 3: Load Models =====
    log("Step 3: Load Models")
    models = {}
    
    # V11
    try:
        ckpt = torch.load(args.v11_ckpt, map_location="cpu")
        v11_model = create_dsn_v8(
            feat_dim=512 + 6,
            use_pcgrad=False,
            num_attn_layers=ckpt["config"].get("num_attn_layers", 2),
            gating_hidden=ckpt["config"].get("gating_hidden", 64),
            lstm_hidden=ckpt["config"].get("lstm_hidden", 128),
        ).to(args.device)
        v11_model.load_state_dict(ckpt["model_state_dict"])
        v11_model.eval()
        models["v11"] = v11_model
        log("  V11 loaded")
    except Exception as e:
        log(f"  V11 failed: {e}")
    
    # VSUMM
    if VSUMM_DSN:
        try:
            vsumm_model = VSUMM_DSN(in_dim=512, hid_dim=256, num_layers=1, cell='lstm').to(args.device)
            ckpt = torch.load(args.vsumm_ckpt, map_location="cpu")
            str_ckpt = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            clean_ckpt = {k.replace("module.", ""): v for k, v in str_ckpt.items()}
            vsumm_model.load_state_dict(clean_ckpt)
            vsumm_model.eval()
            models["vsumm"] = vsumm_model
            log("  VSUMM loaded")
        except Exception as e:
            log(f"  VSUMM failed: {e}")
    
    # LLMVS
    if LLMVSVisual and LLMVSConfig:
        try:
            config = LLMVSConfig(
                reduced_dim=2048, 
                input_dim=512, 
                model='LLMVSVisual', 
                dataset='sakuga', 
                tag='inference'
            )
            llmvs_model = LLMVSVisual(config).to(args.device)
            ckpt = torch.load(args.llmvs_ckpt, map_location="cpu")
            str_ckpt = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            clean_ckpt = {k.replace("module.", ""): v for k, v in str_ckpt.items()}
            llmvs_model.load_state_dict(clean_ckpt, strict=False)
            llmvs_model.eval()
            models["llmvs"] = llmvs_model
            log("  LLMVS loaded")
        except Exception as e:
            log(f"  LLMVS failed: {e}")
    
    # ===== Step 4: Process Each Scene =====
    log("Step 4: Process Scenes")
    all_results = []
    
    for sid, sc in enumerate(tqdm(scenes, desc="Scenes")):
        s, e = int(sc.start_frame), int(sc.end_frame)
        scene_len = e - s + 1
        stride = adaptive_stride(scene_len)
        
        # Decode frames
        frames, frame_indices = decode_scene_frames(str(video_path), s, e, stride)
        
        if len(frames) < 3:
            continue
        
        # Extract features
        feats = extractor.extract_features(frames)
        anime_attrs = extractor.extract_anime_attrs(frames)
        
        # Run comparison and save frames
        scene_results, grids = run_comparison_on_scene(
            frames, feats, anime_attrs, models,
            frame_indices=frame_indices, # Pass original indices
            video_stem=video_stem,
            output_dir=output_dir,
            budget_ratio=args.budget_ratio, 
            device=args.device
        )
        
        scene_results["scene_id"] = sid
        scene_results["start_frame"] = s
        scene_results["end_frame"] = e
        all_results.append(scene_results)
        
        # Save visualization (still useful for overview)
        if grids:
            final_img = np.vstack([g for g in grids if g is not None])
            img_path = output_dir / f"{video_stem}_scene_{sid:04d}.jpg"
            cv2.imwrite(str(img_path), final_img)
    
    # ===== Save Summary =====
    # Aggregate keys
    aggregated_keys = {}
    for model_name in ["V11", "VSUMM", "LLMVS"]:
        all_keys = []
        for r in all_results:
            if model_name in r["models"] and "global_indices" in r["models"][model_name]:
                all_keys.extend(r["models"][model_name]["global_indices"])
        if all_keys:
            aggregated_keys[model_name] = sorted(list(set(all_keys)))

    summary = {
        "video": str(video_path),
        "total_scenes": len(all_results),
        "aggregated_keys": aggregated_keys,
        "scenes": all_results,
    }
    
    # Compute averages
    for model_name in ["V11", "VSUMM", "LLMVS"]:
        mprs = [r["models"].get(model_name, {}).get("mpr", None) for r in all_results]
        mprs = [m for m in mprs if m is not None]
        if mprs:
            summary[f"{model_name}_avg_mpr"] = float(np.mean(mprs))
    
    json_path = output_dir / f"{video_stem}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # ===== Step 5: Unified Comparison Image =====
    log("Step 5: Generating Unified Comparison Image")
    try:
        create_unified_comparison(
            output_dir, video_stem, aggregated_keys
        )
    except Exception as e:
        log(f"Error creating unified image: {e}")
        import traceback
        traceback.print_exc()

    log(f"\n✅ Done! Results saved to {output_dir}")
    log(f"   Summary: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    for model_name in ["V11", "VSUMM", "LLMVS"]:
        avg = summary.get(f"{model_name}_avg_mpr", None)
        if avg:
            print(f"{model_name:10s}: MPR = {avg:.4f}")
    print("="*60)


def create_unified_comparison(output_dir: Path, video_stem: str, aggregated_keys: Dict[str, List[int]]):
    """
    Creates a single large image comparing keyframes from all models.
    Rows: V11, VSUMM, LLMVS
    """
    target_h = 180
    model_rows = []
    
    # Define order
    models = ["V11", "VSUMM", "LLMVS"]
    
    max_w = 0
    
    for model in models:
        keys = aggregated_keys.get(model, [])
        if not keys:
            continue
            
        model_dir = output_dir / video_stem / model
        images = []
        for k in keys:
            fname = f"frame_{k:06d}.jpg"
            p = model_dir / fname
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    # Resize keeping aspect ratio
                    h, w = img.shape[:2]
                    scale = target_h / h
                    new_w = int(w * scale)
                    img_resized = cv2.resize(img, (new_w, target_h))
                    images.append(img_resized)
        
        if not images:
            continue
            
        # Concatenate horizontally
        row_img = np.hstack(images)
        
        # Add label
        label_w = 100
        label_img = np.zeros((target_h, label_w, 3), dtype=np.uint8)
        cv2.putText(label_img, model, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 255), 2, cv2.LINE_AA)
                   
        full_row = np.hstack([label_img, row_img])
        model_rows.append(full_row)
        max_w = max(max_w, full_row.shape[1])

    if not model_rows:
        return

    # Pad rows to max_w and stack vertically
    padded_rows = []
    for row in model_rows:
        h, w = row.shape[:2]
        if w < max_w:
            pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        padded_rows.append(row)
        
    final_img = np.vstack(padded_rows)
    save_path = output_dir / f"{video_stem}_unified_comparison.jpg"
    cv2.imwrite(str(save_path), final_img)
    log(f"Saved unified comparison to {save_path}")


if __name__ == "__main__":
    main()

"""
python scripts/run_inference_video.py --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga_test/70207.mp4 --output_dir demo_outputs/video_verification_70207 --budget_ratio 0.05

python scripts/run_inference_video.py --video /home/serverai/ltdoanh/LayoutGeneration/data/samples/Sakuga/13926.mp4 --output_dir demo_outputs/video_verification_13926 --budget_ratio 0.05
"""