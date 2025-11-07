#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization of reconstruction medoids:
- Build embeddings for sampled frames.
- Map provided keyframes to the sampled index space (nearest by index).
- Compute nearest-medoid assignment in cosine distance (on L2-normalized feats).
- Produce:
  1) tsne.png                - t-SNE of all frames (small) and medoids (large)
  2) timeline_assignment.png - scatter of (frame_idx vs assigned medoid id)
  3) dist_heatmap.png        - heatmap of distances (frames x medoids)
  4) montages/medoid_XX.jpg  - contact sheet per medoid: medoid + nearest neighbors
- Export assignment_table.csv mapping sampled frames to (medoid_id, distance).
All comments are in English (per user requirement).
"""

from __future__ import annotations
import os
import csv
import json
import argparse
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from eval.feature_extractor import FeatureExtractor, FrameSample


# ----------------------------
# I/O helpers
# ----------------------------
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def load_scenes_json(path: str) -> List[Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = [(int(d["start_frame"]), int(d["end_frame"])) for d in data]
    scenes.sort(key=lambda x: x[0])
    return scenes


def load_keyframes_csv(path: str) -> List[int]:
    idxs: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "frame_idx" not in reader.fieldnames:
            raise ValueError("keyframes.csv must contain a 'frame_idx' column.")
        for row in reader:
            idxs.append(int(row["frame_idx"]))
    return sorted(idxs)


def frames_to_timecode(frame_idx: int, fps: float) -> str:
    t = frame_idx / max(1.0, fps)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ----------------------------
# Core computations
# ----------------------------
def sample_and_embed_all(
    video_path: str,
    extractor: FeatureExtractor,
    stride: int,
    max_frames: int,
) -> Tuple[np.ndarray, List[int]]:
    """Return (Fe_all [N,D] L2-normalized, idx_all [N])."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    idx = list(range(0, total, max(1, stride)))
    if max_frames > 0 and len(idx) > max_frames:
        sel = np.linspace(0, len(idx)-1, max_frames, dtype=int)
        idx = [idx[i] for i in sel]

    samples = [FrameSample(i) for i in idx]
    frames = extractor.load_frames(video_path, samples)
    Fe = extractor.embed_images(frames, batch_size=32)
    # Already L2-normalized by FeatureExtractor
    return Fe, idx


def map_keys_to_sample_space(keys: List[int], idx_all: List[int]) -> List[int]:
    """Map arbitrary key indices to the nearest index in the sampled set."""
    if not keys:
        return []
    if not idx_all:
        return []
    mapped: List[int] = []
    for k in keys:
        # nearest by |k - idx|
        nearest = min(idx_all, key=lambda j: abs(k - j))
        mapped.append(nearest)
    # Remove duplicates and sort
    mapped = sorted(set(mapped))
    return mapped


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Assume row-wise L2-normalized embeddings. Return (A @ B^T)."""
    return A @ B.T


def nearest_medoid_assign(
    Fe_all: np.ndarray, Fe_keys: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      - assign_ids: (N,) int - argmax over cosine similarity → nearest key id
      - assign_dist: (N,) float - cosine distance to the assigned key = 1 - max_sim
    """
    if Fe_keys.shape[0] == 0:
        N = Fe_all.shape[0]
        return np.full((N,), -1, dtype=np.int32), np.full((N,), np.nan, dtype=np.float32)
    S = cosine_sim_matrix(Fe_all, Fe_keys)      # (N, M)
    max_sim = np.max(S, axis=1)                 # (N,)
    ids = np.argmax(S, axis=1).astype(np.int32) # (N,)
    dist = 1.0 - max_sim.astype(np.float32)
    return ids, dist


# ----------------------------
# Visualizations
# ----------------------------
def plot_tsne(
    Fe_all: np.ndarray,
    Fe_keys: np.ndarray,
    assign_ids: np.ndarray,
    out_path: str,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> None:
    """t-SNE of all frames; medoids embedded in SAME space. Colors = assigned medoid id."""
    # Combine all frames and medoids so they share one t-SNE space.
    N = Fe_all.shape[0]
    M = Fe_keys.shape[0]
    if N == 0:
        return
    X = Fe_all if M == 0 else np.vstack([Fe_all, Fe_keys])  # (N+M, D)

    # Perplexity must be < number of samples. Also avoid too small perplexity.
    n_samples = X.shape[0]
    p = min(perplexity, max(5.0, min(50.0, n_samples - 1)))

    # Be compatible with older scikit-learn (no 'learning_rate="auto"')
    try:
        # Keep comments in English
        tsne = TSNE(
            n_components=2,
            perplexity=p,          # make sure p < number of samples
            max_iter=n_iter,       # <-- use max_iter instead of n_iter
            learning_rate=200.0,   # numeric for compatibility
            init="random",
            method="barnes_hut",
            random_state=42,
        )

    except TypeError:
        # Fallback for old sklearn
        tsne = TSNE(
            n_components=2,
            perplexity=p,
            n_iter=n_iter,
            init="random",
            learning_rate=200.0,
            random_state=42,
        )

    Y = tsne.fit_transform(X)               # (N+M, 2) or (N,2) if M==0
    Y_all = Y[:N]
    Y_keys = None if M == 0 else Y[N:]

    # Plot: frames colored by assigned medoid id, medoids as big X markers
    plt.figure()
    if M > 0:
        plt.scatter(Y_all[:, 0], Y_all[:, 1], s=8, c=assign_ids, alpha=0.7, linewidths=0)
        plt.scatter(Y_keys[:, 0], Y_keys[:, 1], s=120, marker="X", c=np.arange(M), edgecolors="k")
        plt.title("t-SNE: frames (colored by nearest medoid) + medoids (X)")
    else:
        plt.scatter(Y_all[:, 0], Y_all[:, 1], s=8, alpha=0.7, linewidths=0)
        plt.title("t-SNE: frames (no medoids provided)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def plot_timeline_assignment(
    idx_all: List[int],
    assign_ids: np.ndarray,
    out_path: str,
) -> None:
    """Scatter (frame_idx vs medoid_id). Shows temporal partitioning (Voronoi in time)."""
    x = np.array(idx_all, dtype=np.int64)
    y = assign_ids.astype(np.int32)
    plt.figure()
    plt.scatter(x, y, s=8, alpha=0.7)
    plt.xlabel("Frame index")
    plt.ylabel("Assigned medoid id")
    plt.title("Timeline assignment: nearest-medoid per sampled frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_distance_heatmap(
    Fe_all: np.ndarray,
    Fe_keys: np.ndarray,
    out_path: str,
) -> None:
    """Heatmap of cosine distance from each frame to each medoid."""
    if Fe_keys.shape[0] == 0 or Fe_all.shape[0] == 0:
        return
    S = cosine_sim_matrix(Fe_all, Fe_keys)  # (N, M)
    D = 1.0 - S
    plt.figure()
    plt.imshow(D, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Medoid id")
    plt.ylabel("Frame (sample order)")
    plt.title("Cosine distance heatmap (frames × medoids)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_montage_for_medoid(
    video_path: str,
    medoid_idx: int,
    medoid_frame_idx: int,
    neighbor_frame_idx: List[int],
    out_path: str,
    thumb_size: Tuple[int, int] = (240, 135),
    grid_cols: int = 5,
) -> None:
    """
    Save a contact-sheet image:
      Row 0, Col 0: the medoid frame (highlight by a thin border).
      The rest: nearest neighbor frames in ascending distance order.
    """
    ensure_dir(os.path.dirname(out_path))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    def _read_and_resize(fi: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frm = cap.read()
        if not ok or frm is None:
            return None
        return cv2.resize(frm, thumb_size, interpolation=cv2.INTER_AREA)

    tiles: List[np.ndarray] = []
    # Put medoid first
    med = _read_and_resize(medoid_frame_idx)
    if med is not None:
        tiles.append(med)
    # Then neighbors
    for fi in neighbor_frame_idx:
        im = _read_and_resize(fi)
        if im is not None:
            tiles.append(im)

    cap.release()
    if not tiles:
        return

    # Compute grid size
    n_tiles = len(tiles)
    cols = max(1, grid_cols)
    rows = int(np.ceil(n_tiles / cols))

    H, W, _ = tiles[0].shape
    gap = 6
    canvas = np.full((rows * H + (rows + 1) * gap, cols * W + (cols + 1) * gap, 3), 32, dtype=np.uint8)

    # Paste tiles
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        y0 = gap + r * (H + gap)
        x0 = gap + c * (W + gap)
        canvas[y0:y0+H, x0:x0+W] = tile

    # Optional: draw a thin rectangle around the medoid at (0,0)
    cv2.rectangle(canvas, (gap, gap), (gap + W, gap + H), (255, 255, 255), 2)

    cv2.imwrite(out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def build_montages(
    video_path: str,
    idx_all: List[int],
    Fe_all: np.ndarray,
    mapped_keys: List[int],
    Fe_keys: np.ndarray,
    assign_ids: np.ndarray,
    assign_dist: np.ndarray,
    out_dir: str,
    top_neighbors: int = 12,
) -> None:
    """Save a montage per medoid with the closest neighbors."""
    ensure_dir(out_dir)
    if len(mapped_keys) == 0:
        return

    # For each medoid m, collect frames assigned to m with their distances
    M = len(mapped_keys)
    for m in range(M):
        # indices in the sampled set assigned to m
        inds = np.where(assign_ids == m)[0].tolist()
        # sort by distance ascending
        inds.sort(key=lambda i: float(assign_dist[i]))
        # take top neighbors (excluding the medoid itself if present twice)
        neighbor_sample_inds = inds[:top_neighbors]

        # Convert sampled indices to absolute frame indices
        neighbor_frame_idx = [idx_all[i] for i in neighbor_sample_inds]
        medoid_frame_idx = mapped_keys[m]

        out_path = os.path.join(out_dir, f"medoid_{m:02d}.jpg")
        save_montage_for_medoid(
            video_path=video_path,
            medoid_idx=m,
            medoid_frame_idx=medoid_frame_idx,
            neighbor_frame_idx=neighbor_frame_idx,
            out_path=out_path,
        )

def save_frames_cache(path: str, Fe_all: np.ndarray, idx_all: List[int]) -> None:
    np.savez_compressed(path, Fe_all=Fe_all, idx_all=np.asarray(idx_all, dtype=np.int64))

def load_frames_cache(path: str) -> Tuple[np.ndarray, List[int]]:
    data = np.load(path, allow_pickle=False)
    Fe_all = data["Fe_all"]
    idx_all = data["idx_all"].tolist()
    return Fe_all, idx_all

def fit_tsne_frames_only(Fe_all: np.ndarray, perplexity: float, n_iter: int) -> np.ndarray:
    """Fit t-SNE on frames only (Fe_all) for stable 2D coords across runs."""
    if Fe_all.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    n_samples = Fe_all.shape[0]
    p = min(perplexity, max(5.0, min(50.0, n_samples - 1)))
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=p,
            max_iter=n_iter,
            learning_rate=200.0,
            init="random",
            method="barnes_hut",
            random_state=42,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=p,
            n_iter=n_iter,
            init="random",
            learning_rate=200.0,
            random_state=42,
        )
    Y_all = tsne.fit_transform(Fe_all)
    return Y_all

def plot_tsne_frames_only(
    Y_all: np.ndarray,
    assign_ids: np.ndarray,
    out_path: str,
    title: str = "t-SNE (frames only): colored by nearest medoid"
) -> None:
    """Scatter frames in fixed 2D (Y_all), colored by assigned medoid id."""
    if Y_all.shape[0] == 0:
        return
    plt.figure()
    plt.scatter(Y_all[:, 0], Y_all[:, 1], s=8, c=assign_ids, alpha=0.7, linewidths=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_tsne_with_medoids_centroids(
    Y_all: np.ndarray,
    assign_ids: np.ndarray,
    out_path: str,
    title: str = "t-SNE (frames only) + medoid centroids (approx)"
) -> None:
    """Overlay medoid 'positions' as 2D centroids of their assigned frames."""
    if Y_all.shape[0] == 0:
        return
    plt.figure()
    plt.scatter(Y_all[:, 0], Y_all[:, 1], s=8, c=assign_ids, alpha=0.7, linewidths=0)
    # draw medoid centroids
    if assign_ids.size > 0 and np.max(assign_ids) >= 0:
        M = int(np.max(assign_ids)) + 1
        for m in range(M):
            idx = np.where(assign_ids == m)[0]
            if idx.size == 0:
                continue
            centroid = Y_all[idx].mean(axis=0)
            plt.scatter([centroid[0]], [centroid[1]], s=160, marker="X", edgecolors="k")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize reconstruction medoids and assignments.")
    ap.add_argument("--video", required=True, type=str)
    ap.add_argument("--scenes_json", required=True, type=str, help="scenes.json from pipeline")
    ap.add_argument("--keyframes_csv", required=True, type=str, help="keyframes.csv with frame_idx column")
    ap.add_argument("--out_dir", required=True, type=str)

    # Embedding backbone & sampling for 'all frames'
    ap.add_argument("--backbone", type=str, default="resnet50")      # resnet50 | vit_b_16 (torchvision)
    ap.add_argument("--device", type=str, default=None)              # 'cuda' | 'cpu'
    ap.add_argument("--input_w", type=int, default=224)
    ap.add_argument("--input_h", type=int, default=224)
    ap.add_argument("--sample_stride", type=int, default=10)
    ap.add_argument("--max_frames_eval", type=int, default=300)

    # t-SNE params
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_iter", type=int, default=1000)

    # Montage
    ap.add_argument("--top_neighbors", type=int, default=12)

    # Optional caching of sampled frames & embeddings
    ap.add_argument("--frames_cache", type=str, default=None,
                help="If set, save/load Fe_all & idx_all here to ensure same frames across runs (.npz).")
    ap.add_argument("--tsne_mode", type=str, default="frames",
                    choices=["frames", "frames_plus_medoids"],
                    help="Fit t-SNE on frames only (stable across runs) or on frames+medoids (old behavior).")
    ap.add_argument("--embed_keys_direct", action="store_true", default=True,
                help="If true, embed keyframes directly (no snapping & no dedup).")
    
    args = ap.parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "montages"))

    # Load inputs
    scenes = load_scenes_json(args.scenes_json)
    keys = load_keyframes_csv(args.keyframes_csv)

    # Build extractor and embed sampled frames
    extractor = FeatureExtractor(
        backbone=args.backbone,
        device=args.device,
        input_size=(args.input_w, args.input_h),
        pretrained=True,
    )

    # 1) Fe_all, idx_all: load cache or compute
    if args.frames_cache and os.path.exists(args.frames_cache):
        Fe_all, idx_all = load_frames_cache(args.frames_cache)
        print(f"[INFO] Loaded frames cache from {args.frames_cache} -> N={Fe_all.shape[0]}")
    else:
        Fe_all, idx_all = sample_and_embed_all(
            video_path=args.video,
            extractor=extractor,
            stride=args.sample_stride,
            max_frames=args.max_frames_eval,
        )
        if args.frames_cache:
            save_frames_cache(args.frames_cache, Fe_all, idx_all)
            print(f"[INFO] Saved frames cache to {args.frames_cache} -> N={Fe_all.shape[0]}")

    # 2) Keys → Fe_keys
    keys = load_keyframes_csv(args.keyframes_csv)
    if args.embed_keys_direct:
        key_indices = keys[:]  # use original indices, no snapping, no dedup
        print(f"[INFO] embed_keys_direct=True  K_raw={len(keys)}  K_used={len(key_indices)}")
    else:
        mapped_keys = map_keys_to_sample_space(keys, idx_all)
        key_indices = mapped_keys
        print(f"[INFO] embed_keys_direct=False K_raw={len(keys)}  M_mapped_unique={len(key_indices)}")

    key_samples = [FrameSample(i) for i in key_indices]
    key_frames = extractor.load_frames(args.video, key_samples)
    Fe_keys = extractor.embed_images(key_frames, batch_size=32)

    # Assign each sampled frame to its nearest medoid (cosine distance)
    assign_ids, assign_dist = nearest_medoid_assign(Fe_all, Fe_keys)

    # Save assignment table for inspection
    assign_csv = os.path.join(args.out_dir, "assignment_table.csv")
    with open(assign_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_row", "frame_idx", "assigned_medoid_id", "cosine_distance"])
        for i in range(len(idx_all)):
            w.writerow([i, idx_all[i], int(assign_ids[i]), float(assign_dist[i])])

    # 1) t-SNE plot
    if args.tsne_mode == "frames":
        # Fit t-SNE once on frames only (stable across runs)
        Y_all = fit_tsne_frames_only(Fe_all, args.tsne_perplexity, args.tsne_iter)
        plot_tsne_frames_only(
            Y_all=Y_all,
            assign_ids=assign_ids,
            out_path=os.path.join(args.out_dir, "tsne_frames_only.png"),
            title="t-SNE (frames only): colored by nearest medoid",
        )
        # Optional: overlay medoid centroids (approximate X)
        plot_tsne_with_medoids_centroids(
            Y_all=Y_all,
            assign_ids=assign_ids,
            out_path=os.path.join(args.out_dir, "tsne_frames_only_with_centroids.png"),
            title="t-SNE (frames only) + medoid centroids (approx)",
        )
    else:
        # Old behavior: embed frames+medoids together (layout changes with M)
        plot_tsne(
            Fe_all=Fe_all,
            Fe_keys=Fe_keys,
            assign_ids=assign_ids,
            out_path=os.path.join(args.out_dir, "tsne.png"),
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_iter,
        )

    # 2) Timeline assignment
    plot_timeline_assignment(
        idx_all=idx_all,
        assign_ids=assign_ids,
        out_path=os.path.join(args.out_dir, "timeline_assignment.png"),
    )

    # 3) Distance heatmap
    plot_distance_heatmap(
        Fe_all=Fe_all,
        Fe_keys=Fe_keys,
        out_path=os.path.join(args.out_dir, "dist_heatmap.png"),
    )

    # 4) Montages per medoid
    build_montages(
        video_path=args.video,
        idx_all=idx_all,
        Fe_all=Fe_all,
        mapped_keys=key_indices,     # now these are "used keys" (direct or mapped)
        Fe_keys=Fe_keys,
        assign_ids=assign_ids,
        assign_dist=assign_dist,
        out_dir=os.path.join(args.out_dir, "montages"),
        top_neighbors=args.top_neighbors,
    )

    print(f"[DONE] Visualizations saved to: {args.out_dir}")
    print("  - tsne.png")
    print("  - timeline_assignment.png")
    print("  - dist_heatmap.png")
    print("  - montages/*.jpg")
    print("  - assignment_table.csv")


if __name__ == "__main__":
    main()

"""

python -m eval.visualize.viz_medoids \
  --video samples/Sakuga/10736.mp4 \
  --scenes_json outputs/run_psd_lpips/scenes.json \
  --keyframes_csv outputs/run_psd_lpips/keyframes.csv \
  --out_dir outputs/visualize/viz_medoids_lpips \
  --backbone resnet50 \
  --sample_stride 1 --max_frames_eval 2000 \
  --tsne_perplexity 30 --tsne_iter 1000 \
  --top_neighbors 50 \
  --frames_cache outputs/visualize/shared_cache/10736_stride1_max2000_resnet50.npz \
  --tsne_mode frames \
  --embed_keys_direct

python -m eval.visualize.viz_medoids \
  --video samples/Sakuga/10736.mp4 \
  --scenes_json outputs/run_psd_dists/scenes.json \
  --keyframes_csv outputs/run_psd_dists/keyframes.csv \
  --out_dir outputs/visualize/viz_medoids_dists \
  --backbone resnet50 \
  --sample_stride 1 --max_frames_eval 2000 \
  --tsne_perplexity 30 --tsne_iter 1000 \
  --top_neighbors 50 \
  --frames_cache outputs/visualize/shared_cache/10736_stride1_max2000_resnet50.npz \
  --tsne_mode frames \
  --embed_keys_direct

"""