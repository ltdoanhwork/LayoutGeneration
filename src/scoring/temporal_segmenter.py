# -*- coding: utf-8 -*-
"""
auto_temporal_segmenter.py
--------------------------
Auto-tuned temporal segmentation for keyframe crops using PELT.

Upgrades:
- Auto target K from N and max_len; grid-search penalty to hit K.
- Peak-boosted boundary: median filter + local z-score + NMS.
- Scene hints: weak priors from file names (scene_XXXX).
- Auto switch: boundary -> rbf if B is flat.
- Strict guards: min_len, max_len, merge tiny tails.

Usage:
  python auto_temporal_segmenter.py --dir /path/to/crops \
    --max-len 4 --min-len 3 --mode auto --save out.json

Requires:
- ruptures, numpy, pillow, scikit-learn
- scoring.get_clip_embedding, scoring.get_iqa_score
"""
import os
import re
import json
import argparse
from typing import List, Tuple, Sequence

import numpy as np
import ruptures as rpt
from PIL import Image
from sklearn.preprocessing import StandardScaler

# ---- Your scoring API (must be resolvable) ----
from ..scoring import get_clip_embedding, get_iqa_score

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ----------------------------
# Sorting & filename hints
# ----------------------------
def _is_image(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def _parse_scene_frame(name: str) -> Tuple[int, int]:
    m = re.search(r"scene_(\d+)_frame_(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"(\d+)", name)
    return (0, int(nums[-1])) if nums else (0, 0)

def list_images_sorted(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if _is_image(f)]
    files.sort(key=lambda x: _parse_scene_frame(x))
    return [os.path.join(folder, f) for f in files]

def scene_change_hints(files: Sequence[str]) -> np.ndarray:
    """
    Weak priors: return binary array H of length N-1 where name suggests scene change.
    """
    if len(files) < 2: return np.zeros(0, dtype=np.float32)
    ids = [ _parse_scene_frame(os.path.basename(f))[0] for f in files ]
    H = np.array([1.0 if ids[i] != ids[i+1] else 0.0 for i in range(len(ids)-1)], dtype=np.float32)
    return H

# ----------------------------
# Robust deltas & smoothing
# ----------------------------
def cosine_delta_embeds(emb: np.ndarray) -> np.ndarray:
    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = np.einsum("ij,ij->i", embn[:-1], embn[1:])
    return 1.0 - np.clip(sim, -1.0, 1.0)  # [N-1]

def robust_delta_1d(x: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(x))
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-8
    return np.clip((d - med) / (1.4826 * mad + 1e-8), 0.0, 6.0)

def median_filter_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1 or x.size < k: return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i+k])
    return out

def local_zscore(x: np.ndarray, win: int = 11) -> np.ndarray:
    if win <= 3 or x.size < win: return x
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    out = np.empty_like(x)
    for i in range(len(x)):
        seg = xp[i:i+win]
        mu = seg.mean()
        sd = seg.std() + 1e-8
        out[i] = (x[i] - mu) / sd
    return np.maximum(out, 0.0)

def nms_1d(x: np.ndarray, radius: int = 2, keep: int = 9999) -> np.ndarray:
    """
    1D non-maximum suppression: keep top 'keep' peaks with neighborhood 'radius'.
    Return binary mask of same length as x.
    """
    idx = np.argsort(x)[::-1]
    used = np.zeros_like(x, dtype=bool)
    keep_mask = np.zeros_like(x, dtype=bool)
    for i in idx:
        if used[i]: continue
        keep_mask[i] = True
        if keep_mask.sum() >= keep: break
        lo = max(0, i - radius); hi = min(len(x), i + radius + 1)
        used[lo:hi] = True
        used[i] = False  # allow self
    return keep_mask

# ----------------------------
# Auto Segmenter
# ----------------------------
class AutoSegmenter:
    """
    Auto-tuned PELT segmenter with:
      - boundary (1D) + rbf fallback
      - auto penalty search to match target K
      - peak-boosted boundary and scene hints
      - min_len / max_len guards
    """
    def __init__(self,
                 w_clip: float = 0.8,
                 w_iqa: float = 0.2,
                 min_len: int = 3,
                 max_len: int = 0,
                 mode: str = "auto",   # "auto" | "boundary" | "rbf"
                 kmin: int = 1,
                 kmax: int = 2,
                 hint_strength: float = 0.15,  # add to B(t) where scene id changes
                 peak_radius: int = 2):
        assert mode in ("auto", "boundary", "rbf")
        self.w_clip = float(w_clip)
        self.w_iqa = float(w_iqa)
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.mode = mode
        self.kmin = kmin
        self.kmax = kmax
        self.hint_strength = float(hint_strength)
        self.peak_radius = int(peak_radius)

    # --- features ---
    def _extract_clip_iqa(self, path: str) -> Tuple[np.ndarray, float]:
        img = Image.open(path).convert("RGB")
        clip_feat = get_clip_embedding(img).astype(np.float32)  # [D]
        iqa = float(get_iqa_score(img))
        return clip_feat, iqa

    def _extract_seq(self, files: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        clips, iqas = [], []
        for p in files:
            c, q = self._extract_clip_iqa(p)
            clips.append(c); iqas.append(q)
        return np.stack(clips, 0), np.asarray(iqas, np.float32)

    # --- boundary build ---
    def _build_B(self, clips_raw: np.ndarray, iqa_scaled: np.ndarray, hints: np.ndarray) -> np.ndarray:
        d_clip = cosine_delta_embeds(clips_raw)      # [N-1]
        d_iqa  = robust_delta_1d(iqa_scaled)         # [N-1]
        B = self.w_clip * d_clip + self.w_iqa * d_iqa

        # Peak boosting: median filter -> local zscore -> rescale
        B = median_filter_1d(B, k=5)
        Z = local_zscore(B, win=11)
        B = 0.7 * B + 0.3 * Z

        # Scene hints (weak prior)
        if hints.size:
            B = B + self.hint_strength * hints

        # Min-max normalize
        B = (B - B.min()) / (B.max() - B.min() + 1e-8)
        return B

    # --- penalty search to hit target K ---
    def _auto_penalty_1d(self, B: np.ndarray, K_target: int) -> Tuple[List[int], float]:
        """
        Grid-search penalty to get number of segments close to K_target.
        Returns (cps, chosen_pen).
        """
        N = len(B) + 1
        if N <= self.min_len: return [N-1], 1.0

        def predict(pen):
            algo = rpt.Pelt(model="l2", min_size=self.min_len).fit(B.reshape(-1,1))
            cps = algo.predict(pen=pen)
            if not cps or cps[-1] != (N-1):
                cps = list(cps) + [N-1]
            return cps

        # Build search range from data variability
        lam_min = 0.05
        lam_max = max(5.0, 10.0 * float(np.mean(B)+1e-3))
        grid = np.geomspace(lam_min, lam_max, 20)

        best_cps, best_gap, best_pen = None, 1e9, None
        for lam in grid:
            cps = predict(lam)
            K = len(cps)
            gap = abs(K - K_target)
            if gap < best_gap or (gap == best_gap and K >= K_target):
                best_cps, best_gap, best_pen = cps, gap, lam
                if gap == 0:
                    break
        return best_cps, float(best_pen if best_pen is not None else 1.0)

    # --- enforce max_len by peaks (with fallback chunking) ---
    def _split_by_maxlen(self, seg: List[int], B_full: np.ndarray) -> List[List[int]]:
        if self.max_len <= 0 or len(seg) <= self.max_len:
            return [seg]
        s, e = seg[0], seg[-1]
        L = len(seg)
        need_k = int(np.ceil(L / self.max_len))
        internal = B_full[s:e] if e > s else np.zeros(0, np.float32)
        cuts = []
        if internal.size:
            mask = nms_1d(internal, radius=self.peak_radius, keep=need_k-1)
            cuts = [s+i for i, m in enumerate(mask) if m]
        # fallback even chunking if not enough peaks
        if len(cuts) < (need_k - 1):
            extra = list(range(s + self.max_len - 1, e, self.max_len))[:(need_k - 1 - len(cuts))]
            cuts = sorted(set(cuts + extra))

        subs, prev = [], s
        for cp in cuts + [e]:
            subs.append(list(range(prev, cp+1)))
            prev = cp + 1
        # merge tiny tails
        cleaned = []
        for sub in subs:
            if cleaned and len(sub) < self.min_len:
                cleaned[-1].extend(sub)
            else:
                cleaned.append(sub)
        # hard cap if still > max_len
        final = []
        for sub in cleaned:
            if len(sub) <= self.max_len:
                final.append(sub)
            else:
                for i in range(0, len(sub), self.max_len):
                    final.append(sub[i:i+self.max_len])
        return final

    # --- main ---
    def segment(self, image_dir: str) -> Tuple[List[List[int]], List[List[str]]]:
        files = list_images_sorted(image_dir)
        if not files:
            raise ValueError(f"No images in {image_dir}")
        N = len(files)

        clips_raw, iqas = self._extract_seq(files)
        iqa_scaled = StandardScaler().fit_transform(iqas.reshape(-1,1)).ravel()
        hints = scene_change_hints(files)  # [N-1]

        # target K from N and max_len
        if self.max_len > 0:
            K_target = int(np.clip(np.round(N / self.max_len), self.kmin, self.kmax))
        else:
            K_target = self.kmin  # minimal split

        # choose mode
        mode = self.mode
        if mode == "auto": mode = "boundary"

        # first try boundary
        B = self._build_B(clips_raw, iqa_scaled, hints)
        cps, pen = self._auto_penalty_1d(B, K_target)

        segments_idx = []
        start = 0
        for cp in cps:
            seg = list(range(start, cp+1))
            if len(seg) >= self.min_len:
                segments_idx.append(seg)
                start = cp+1
        if not segments_idx:
            segments_idx = [list(range(N))]

        # fallback to rbf if still too few segments and allowed
        if mode in ("auto", "rbf") and len(segments_idx) < max(1, K_target//2):
            clips_n = clips_raw / (np.linalg.norm(clips_raw, axis=1, keepdims=True) + 1e-8)
            X = np.hstack([clips_n, iqa_scaled[:,None]])
            algo = rpt.Pelt(model="rbf", min_size=self.min_len).fit(X)
            # penalty for rbf is typically larger; scale by N/max_len
            lam_rbf = max(3.0, 1.5 * (N / max(self.max_len, self.min_len)))
            cps = algo.predict(pen=lam_rbf)
            if not cps or cps[-1] != (N-1):
                cps = list(cps) + [N-1]

            segments_idx = []
            start = 0
            for cp in cps:
                seg = list(range(start, cp+1))
                if len(seg) >= self.min_len:
                    segments_idx.append(seg)
                    start = cp+1
            if not segments_idx:
                segments_idx = [list(range(N))]
            # build a proxy B for max_len splitting
            B = np.linalg.norm(X[1:] - X[:-1], axis=1)
            B = (B - B.min()) / (B.max() - B.min() + 1e-8)

        # enforce max_len
        if self.max_len > 0:
            splitted = []
            for seg in segments_idx:
                splitted.extend(self._split_by_maxlen(seg, B_full=B))
            segments_idx = splitted

        segments_files = [[files[i] for i in seg] for seg in segments_idx]
        return segments_idx, segments_files

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Auto temporal segmentation (PELT)")
    ap.add_argument("--dir", type=str, required=True, help="Directory of cropped images")
    ap.add_argument("--mode", type=str, default="auto", choices=["auto","boundary","rbf"])
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=4)
    ap.add_argument("--w-clip", type=float, default=0.8)
    ap.add_argument("--w-iqa", type=float, default=0.2)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--save", type=str, default="")
    return ap.parse_args()

def main():
    args = parse_args()
    seg = AutoSegmenter(
        w_clip=args.w_clip,
        w_iqa=args.w_iqa,
        min_len=args.min_len,
        max_len=args.max_len,
        mode=args.mode,
        kmin=args.kmin,
        kmax=args.kmax,
    )
    idx, files = seg.segment(args.dir)
    print(f"Detected {len(idx)} segments")
    for k, seg_idx in enumerate(idx, 1):
        print(f"Segment {k:02d}: idx [{seg_idx[0]}..{seg_idx[-1]}], len={len(seg_idx)}")
        print(f"  First: {files[k-1][0]}")
        print(f"  Last : {files[k-1][-1]}")
    if args.save:
        out = {"segments_idx": idx, "segments_files": files,
               "params": {"min_len": args.min_len, "max_len": args.max_len,
                          "mode": args.mode, "w_clip": args.w_clip, "w_iqa": args.w_iqa,
                          "kmin": args.kmin, "kmax": args.kmax}}
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.save}")

if __name__ == "__main__":
    main()
