
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import numpy as np

try:
    import cv2  # optional for flow or reading images
except Exception:
    cv2 = None

from scipy.stats import wasserstein_distance


# ==============================
# Metrics (objectives to minimize)
# ==============================

def reconstruction_error(
    feats_all: np.ndarray,  # (N, D) L2-normalized
    feats_keys: np.ndarray  # (M, D) L2-normalized
) -> float:
    """
    Mean nearest-neighbor cosine distance of all frames to the keyframe set.
    Assumes L2-normalized embeddings, so cosine_sim = Fa @ Fk^T; d = 1 - cos_sim.
    """
    if feats_keys.shape[0] == 0 or feats_all.shape[0] == 0:
        return float("nan")
    S = feats_all @ feats_keys.T                  # (N, M) cosine similarity
    d = 1.0 - np.max(S, axis=1)                  # (N,) min cosine distance
    return float(np.mean(d).astype(np.float32))


def frechet_distance(
    feats_all: np.ndarray, feats_keys: np.ndarray, eps: float = 1e-6
) -> float:
    """
    Frechet-like distance (FID-style) between two Gaussian approximations.
    FD = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2})

    Notes:
      - Requires at least 2 samples per set to estimate covariance.
      - Uses symmetric sqrt via eigen-decomposition on the symmetrized product.
    """
    if feats_all.shape[0] < 2 or feats_keys.shape[0] < 2:
        return float("nan")

    mu1 = np.mean(feats_all, axis=0)
    mu2 = np.mean(feats_keys, axis=0)
    S1 = np.cov(feats_all, rowvar=False) + np.eye(feats_all.shape[1]) * eps
    S2 = np.cov(feats_keys, rowvar=False) + np.eye(feats_keys.shape[1]) * eps

    diff = mu1 - mu2
    diff2 = float(diff.dot(diff))

    cov_prod = S1.dot(S2)
    cov_prod = (cov_prod + cov_prod.T) * 0.5            # symmetrize (PSD)
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    eigvals[eigvals < 0] = 0.0
    sqrt_cov_prod = eigvecs.dot(np.diag(np.sqrt(eigvals))).dot(eigvecs.T)

    trace = np.trace(S1 + S2 - 2.0 * sqrt_cov_prod)
    return float(diff2 + trace)


# -------------------------------------------------------
# MS-SWD: Multi-Scale Sliced Wasserstein Distance (colors)
# -------------------------------------------------------
def _downsample_gaussian_pyramid(img: np.ndarray, num_scales: int = 3) -> List[np.ndarray]:
    """
    Build a small Gaussian pyramid of an image.
    img: HxWxC in [0, 255] or [0,1], any dtype.
    Returns list [img_s0, img_s1, ..., img_s{num_scales-1}]
    """
    pyr = [img]
    for _ in range(1, num_scales):
        if cv2 is not None:
            img = cv2.pyrDown(img)
        else:
            # Fallback: simple 2x downsample via slicing (no blur)
            img = img[::2, ::2]
        pyr.append(img)
    return pyr


def _sample_color_projections(num_dirs: int, seed: int = 42) -> np.ndarray:
    """
    Sample random unit directions in RGB space for 1D slicing.
    Returns (num_dirs, 3) normalized vectors.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(num_dirs, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


def _collect_color_values(frames: List[np.ndarray], scale: int) -> np.ndarray:
    """
    Collect pixel colors from a list of frames at a given pyramid scale.
    Returns array of shape (K, 3) where K is total sampled pixels across frames.
    For efficiency, random subsample pixels per frame.
    """
    rng = np.random.default_rng(1234 + scale)
    colors = []
    # Limit pixels per frame to keep runtime reasonable
    max_pixels_per_frame = 50_000
    for img in frames:
        # ensure float32 in [0,1]
        if img.dtype != np.float32:
            x = img.astype(np.float32)
            x /= (255.0 if x.max() > 1.0 else 1.0)
        else:
            x = img
            if x.max() > 1.0:
                x = x / 255.0

        # get pyramid
        pyr = _downsample_gaussian_pyramid(x, num_scales=scale + 1)
        xs = pyr[scale]  # HxWxC
        H, W, C = xs.shape
        pix = H * W
        if pix > max_pixels_per_frame:
            idx = rng.choice(pix, size=max_pixels_per_frame, replace=False)
            pts = xs.reshape(-1, C)[idx]
        else:
            pts = xs.reshape(-1, C)
        colors.append(pts)
    if len(colors) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(colors, axis=0).astype(np.float32)


def ms_swd_color(
    frames_all: List[np.ndarray],
    frames_keys: List[np.ndarray],
    num_scales: int = 3,
    num_dirs: int = 32,
    seed: int = 42
) -> float:
    """
    Multi-Scale Sliced Wasserstein Distance between ALL vs KEY frames in color space.
    - For each scale s:
        * Downsample frames via Gaussian pyramid.
        * Gather RGB pixel sets A_s and B_s.
        * For each random direction v in R^3, project colors to 1D: a = A_s v, b = B_s v
        * Compute 1D Wasserstein distance W1(a,b).
    - Return mean of W1 over directions and scales.

    Notes:
      * This is a practical, dependency-light approximation of MS-SWD as a
        perceptual color distribution discrepancy. It captures multi-scale color
        mismatches between the full video and the selected keyframes subset.
      * If you want to make it faster, lower num_dirs or downsample more aggressively.
    """
    if len(frames_all) < 1 or len(frames_keys) < 1:
        return float("nan")

    dirs = _sample_color_projections(num_dirs=num_dirs, seed=seed)
    distances = []
    for s in range(num_scales):
        A = _collect_color_values(frames_all, scale=s)  # (Ka, 3)
        B = _collect_color_values(frames_keys, scale=s) # (Kb, 3)
        if A.shape[0] == 0 or B.shape[0] == 0:
            continue
        # Optional: subsample to balance sizes for speed
        K = min(A.shape[0], B.shape[0], 200_000)
        rng = np.random.default_rng(9876 + s)
        A = A[rng.choice(A.shape[0], size=K, replace=False)]
        B = B[rng.choice(B.shape[0], size=K, replace=False)]

        for v in dirs:
            a = A @ v  # (K,)
            b = B @ v  # (K,)
            # SciPy's wasserstein_distance expects 1D arrays
            w = wasserstein_distance(a, b)
            distances.append(w)

    if len(distances) == 0:
        return float("nan")
    return float(np.mean(distances).astype(np.float32))


# =========================
# Feature & distance helpers
# =========================

def l2_normalize(feats: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(feats, axis=axis, keepdims=True)
    return feats / (n + eps)


def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cosine distance on L2-normalized embeddings: D = 1 - A @ B^T
    A: (N, D) normalized
    B: (M, D) normalized
    returns: (N, M)
    """
    return 1.0 - (A @ B.T)


# ================================
# Optional: TV-L1 flow magnitude
# ================================
def compute_tvl1_flow_magnitude(frames: List[np.ndarray]) -> np.ndarray:
    """
    Compute per-frame optical flow magnitude (forward) using OpenCV TV-L1.
    Returns array of shape (N,) with magnitude for frame t (except last).
    The last frame's magnitude is copied from t-1 for same length.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV not available. Install opencv-python.")
    N = len(frames)
    if N < 2:
        return np.zeros((N,), dtype=np.float32)

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    mags = []
    for t in range(N - 1):
        a = frames[t]
        b = frames[t + 1]
        if a.ndim == 3:
            a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.shape[2] == 3 else a[..., 0]
        else:
            a_gray = a
        if b.ndim == 3:
            b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.shape[2] == 3 else b[..., 0]
        else:
            b_gray = b
        a_gray = a_gray.astype(np.float32) / (255.0 if a_gray.max() > 1.0 else 1.0)
        b_gray = b_gray.astype(np.float32) / (255.0 if b_gray.max() > 1.0 else 1.0)

        flow = tvl1.calc(a_gray, b_gray, None)  # HxWx2
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(np.mean(mag)))

    # pad last to equal length
    if len(mags) == 0:
        return np.zeros((N,), dtype=np.float32)
    mags.append(mags[-1])
    return np.array(mags, dtype=np.float32)


# ===========================
# Contextual Bandit RL policy
# ===========================

@dataclass
class BanditConfig:
    budget_ratio: float = 0.05      # fraction of frames to select per scene
    Bmin: int = 3
    Bmax: int = 15
    eps: float = 0.1                # ε for ε-greedy
    lr: float = 1e-2                # learning rate for SGD
    tau_redundancy: float = 0.2     # threshold on cosine distance to call "redundant"
    # reward weights
    w_cov: float = 1.0
    w_div: float = 0.5
    w_motion: float = 0.2
    w_ms_swd: float = 0.2           # optional penalty weight on MS-SWD (lower is better)
    w_over_budget: float = 1.0
    w_redundancy: float = 0.8
    # MS-SWD params
    ms_swd_scales: int = 3
    ms_swd_dirs: int = 16


class EpsGreedyBandit:
    """
    Two-action contextual bandit: a ∈ {Select, Skip}.
    Linear Q approximator per action:
        Q(a|x) = w_a^T x
    Update (semi-gradient):
        w_a <- w_a + lr * (r - w_a^T x) * x
    """

    def __init__(self, dim: int, eps: float = 0.1, lr: float = 1e-2, seed: int = 0):
        self.dim = dim
        self.eps = eps
        self.lr = lr
        rng = np.random.default_rng(seed)
        # small random init helps tie-breaking
        self.w_select = rng.normal(scale=1e-3, size=(dim,))
        self.w_skip = rng.normal(scale=1e-3, size=(dim,))

    def act(self, x: np.ndarray, force_select: bool, force_skip: bool) -> str:
        if force_select:
            return "Select"
        if force_skip:
            return "Skip"

        if np.random.rand() < self.eps:
            return "Select" if np.random.rand() < 0.5 else "Skip"
        q_sel = float(self.w_select @ x)
        q_skip = float(self.w_skip @ x)
        return "Select" if q_sel >= q_skip else "Skip"

    def update(self, x: np.ndarray, action: str, reward: float):
        if action == "Select":
            pred = float(self.w_select @ x)
            self.w_select += self.lr * (reward - pred) * x
        else:
            pred = float(self.w_skip @ x)
            self.w_skip += self.lr * (reward - pred) * x


# ===========================
# Keyframe selection pipeline
# ===========================

@dataclass
class SceneData:
    """
    Data for a single scene.
      frames: list of HxWxC images (BGR or RGB). Required for MS-SWD and/or flow.
      feats:  (T, D) L2-normalized features for each frame.
      flow:   optional (T,) array of flow magnitudes. If None and frames provided,
              you can compute via compute_tvl1_flow_magnitude(frames).
    """
    frames: Optional[List[np.ndarray]]  # can be None if you skip MS-SWD and flow
    feats: np.ndarray                   # (T, D) normalized
    flow: Optional[np.ndarray] = None   # (T,)


class RLKeyframeSelector:
    """
    RL wrapper that selects keyframes for each scene with a contextual bandit.
    Reward is self-supervised and encourages lower reconstruction error,
    lower Frechet distance, lower MS-SWD, higher diversity, and good motion coverage,
    under a frame budget.

    Usage:
      selector = RLKeyframeSelector(cfg)
      indices = selector.select_for_scene(scene)
    """

    def __init__(self, cfg: BanditConfig):
        self.cfg = cfg
        # context: [novelty, motion, time_ratio, local_contrast]
        self.ctx_dim = 4
        self.bandit = EpsGreedyBandit(dim=self.ctx_dim, eps=cfg.eps, lr=cfg.lr, seed=123)

    # ---------- context helpers ----------

    @staticmethod
    def _local_contrast(frame: np.ndarray) -> float:
        """
        Simple local contrast proxy: grayscale std.
        """
        if frame is None:
            return 0.0
        if frame.ndim == 3:
            if frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if cv2 is not None else frame.mean(axis=2)
            else:
                gray = frame[..., 0]
        else:
            gray = frame
        g = gray.astype(np.float32)
        if g.max() > 1.0:
            g = g / 255.0
        return float(np.std(g))

    @staticmethod
    def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mu = x.mean()
        sd = x.std()
        return (x - mu) / (sd + eps)

    # ---------- main selection ----------

    def select_for_scene(self, scene: SceneData) -> List[int]:
        feats = scene.feats   # (T, D) normalized
        T = feats.shape[0]
        B = int(np.clip(int(np.ceil(self.cfg.budget_ratio * T)), self.cfg.Bmin, self.cfg.Bmax))

        # Prepare per-frame context ingredients
        frames = scene.frames if scene.frames is not None else [None] * T

        # motion
        if scene.flow is not None:
            motion = scene.flow.astype(np.float32)
        else:
            # If frames exist and OpenCV available, you can uncomment to compute on the fly:
            # motion = compute_tvl1_flow_magnitude(frames) if frames[0] is not None else np.zeros((T,), dtype=np.float32)
            motion = np.zeros((T,), dtype=np.float32)

        # local contrast
        contrasts = np.array([self._local_contrast(frames[t]) if frames[t] is not None else 0.0 for t in range(T)], dtype=np.float32)

        # incremental novelty cache
        # min cosine distance from each frame i to the current selected set S
        min_dist = np.ones((T,), dtype=np.float32)  # start as distance=1 (since d in [0,2] for unnormalized, but here it's 0..2; with normalized cosine distance max <= 2; 1 is a neutral init)
        S: List[int] = []

        # precompute pairwise cosine sims in blocks for speed if T is large
        # Here we compute on-demand per selected t via dot products to avoid O(T^2) upfront.
        # To accelerate, you can cache feats.T once:
        feats_T = feats.T  # (D, T)

        # bookkeeping for MS-SWD: store selected frames list
        def frames_subset(idx_list: List[int]) -> List[np.ndarray]:
            return [frames[i] for i in idx_list if frames[i] is not None]

        # baseline metrics at start (for reward deltas)
        base_rec = reconstruction_error(feats, feats[S]) if len(S) > 0 else 1.0
        base_fd = frechet_distance(feats, feats[S]) if feats.shape[0] >= 2 and len(S) >= 2 else 1.0
        base_ms = ms_swd_color(frames, frames_subset(S), self.cfg.ms_swd_scales, self.cfg.ms_swd_dirs) if frames[0] is not None and len(S) > 0 else 1.0

        # main loop
        for t in range(T):
            # Budget guards
            force_select = (T - t) < (B - len(S))
            force_skip = (len(S) >= B)

            # Build context vector x_t
            # novelty: current min distance to S (if empty, treat as high novelty)
            novelty_t = float(min_dist[t]) if len(S) > 0 else 1.0
            time_ratio = float(t) / max(1, (T - 1))
            ctx = np.array([novelty_t, motion[t], time_ratio, contrasts[t]], dtype=np.float32)

            # z-score per-scene for stability
            # (We can do a running standardization, but per-dimension global zscore per scene is fine)
            # Compute global stats once for motion/contrast; novelty changes over time so we do min-max clamp
            # Simpler: scale each component to [0,1]-ish by safe bounds:
            # - novelty in [0, 2]; clamp to [0, 1] for stability
            ctx_scaled = ctx.copy()
            ctx_scaled[0] = np.clip(ctx_scaled[0], 0.0, 1.0)
            # motion and contrast: robust scale by median absolute deviation would be better; keep simple:
            if motion.max() > 0:
                ctx_scaled[1] = motion[t] / (np.median(motion) + 1e-6)
                ctx_scaled[1] = np.clip(ctx_scaled[1], 0.0, 5.0)
            else:
                ctx_scaled[1] = 0.0
            if contrasts.max() > 0:
                ctx_scaled[3] = contrasts[t] / (np.median(contrasts) + 1e-6)
                ctx_scaled[3] = np.clip(ctx_scaled[3], 0.0, 5.0)
            # time_ratio is already [0,1]

            # Choose action
            action = self.bandit.act(ctx_scaled, force_select=force_select, force_skip=force_skip)

            if action == "Select":
                # Compute distances from all i to new candidate t: d(i,t) = 1 - feats[i]·feats[t]
                # Use dot product via feats @ feats[t]
                sims_it = feats @ feats[t]             # (T,)
                dist_it = 1.0 - sims_it               # cosine distance
                nearest_before = float(min_dist[t]) if len(S) > 0 else 1.0

                # Coverage gain: reduction in mean(min_dist) if we add t
                prev_mean = float(np.mean(min_dist))
                new_min_dist = np.minimum(min_dist, dist_it)
                new_mean = float(np.mean(new_min_dist))
                cov_gain = prev_mean - new_mean

                # Diversity gain: how far t is from current S (nearest distance before adding)
                div_gain = nearest_before

                # Redundancy penalty
                redundancy = 1.0 if nearest_before < self.cfg.tau_redundancy else 0.0

                # Budget penalty if it would exceed (should be guarded by force_skip, but keep for safety)
                over_budget = 1.0 if (len(S) + 1) > B else 0.0

                # Optional MS-SWD penalty: evaluate sparsely to save time
                # We approximate by recomputing MS-SWD every K selections or for small scenes.
                ms_penalty = 0.0
                will_compute_ms = (frames[0] is not None) and ((len(S) < 4) or ((len(S) + 1) % 5 == 0))
                if will_compute_ms:
                    S_tmp = S + [t]
                    ms_val = ms_swd_color(
                        frames, frames_subset(S_tmp),
                        num_scales=self.cfg.ms_swd_scales,
                        num_dirs=self.cfg.ms_swd_dirs
                    )
                    if not np.isnan(ms_val):
                        # reward should increase when metric decreases -> add as negative
                        # Convert to *gain* (decrease): base - new
                        ms_gain = (base_ms - ms_val) if not np.isnan(base_ms) else 0.0
                        # We'll treat "penalty" as negative weight times (−ms_gain) so positive gain increases reward.
                        ms_penalty = - ms_gain  # will be weighted by w_ms_swd
                # Motion bonus is already in context; if you want explicit reward term:
                motion_bonus = float(motion[t])

                # Total reward (positive is better)
                r = (
                    self.cfg.w_cov * cov_gain
                    + self.cfg.w_div * div_gain
                    + self.cfg.w_motion * motion_bonus
                    - self.cfg.w_redundancy * redundancy
                    - self.cfg.w_over_budget * over_budget
                    + (- self.cfg.w_ms_swd * ms_penalty)  # ms_penalty is negative of (ms_gain); so -w*penalty = +w*ms_gain
                )

                # Update policy
                self.bandit.update(ctx_scaled, action="Select", reward=r)

                # Commit selection
                S.append(t)
                min_dist = new_min_dist

                # Update baselines occasionally to reflect new subset quality
                if len(S) == 1 or (len(S) % 5 == 0):
                    feats_keys = feats[S]
                    base_rec = reconstruction_error(feats, feats_keys)
                    base_fd = frechet_distance(feats, feats_keys) if feats.shape[0] >= 2 and len(S) >= 2 else base_fd
                    if frames[0] is not None:
                        base_ms = ms_swd_color(
                            frames, frames_subset(S),
                            num_scales=self.cfg.ms_swd_scales,
                            num_dirs=self.cfg.ms_swd_dirs
                        )

            else:
                # Skip: small penalty if we are falling behind budget
                need = B - len(S)
                remain = (T - t - 1)
                r = -0.01 if remain < need * 1.2 else 0.0
                self.bandit.update(ctx_scaled, action="Skip", reward=r)

            # Early exit if budget filled
            if len(S) >= B:
                break

        # Fill remainder by novelty if under budget
        if len(S) < B:
            candidates = np.setdiff1d(np.arange(T), np.array(S, dtype=np.int32), assume_unique=True)
            # pick largest novelty
            order = np.argsort(-min_dist[candidates])
            S_extra = candidates[order[: (B - len(S))]]
            S.extend(list(map(int, S_extra)))

        return S


# ===========================
# End-to-end example driver
# ===========================
def example_run():
    """
    Minimal example to demonstrate usage on synthetic data.

    Replace this with your real:
      - frames: list of HxWxC (uint8 or float32)
      - feats:  (T, D) L2-normalized embeddings (e.g., CLIP/DINOv2 features)
      - flow:   optional (T,) flow magnitudes
    """
    # Synthetic scene with T frames and D-dim features
    T, D = 200, 128
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(T, D)).astype(np.float32)
    feats = l2_normalize(feats, axis=1)

    # Fake frames (optional). If you skip MS-SWD, you can set frames=None
    frames = [np.clip(rng.normal(loc=127, scale=40, size=(128, 128, 3)), 0, 255).astype(np.uint8) for _ in range(T)]

    # Optional flow magnitudes
    flow = None  # or: compute_tvl1_flow_magnitude(frames)

    scene = SceneData(frames=frames, feats=feats, flow=flow)

    cfg = BanditConfig(
        budget_ratio=0.06,
        Bmin=3, Bmax=15,
        eps=0.1, lr=1e-2,
        tau_redundancy=0.2,
        w_cov=1.0, w_div=0.5, w_motion=0.2,
        w_ms_swd=0.2, w_over_budget=1.0, w_redundancy=0.8,
        ms_swd_scales=3, ms_swd_dirs=16
    )

    selector = RLKeyframeSelector(cfg)
    S = selector.select_for_scene(scene)

    # Evaluate
    feats_keys = feats[S]
    rec = reconstruction_error(feats, feats_keys)
    fd = frechet_distance(feats, feats_keys)
    ms = ms_swd_color(frames, [frames[i] for i in S], num_scales=cfg.ms_swd_scales, num_dirs=cfg.ms_swd_dirs)

    print(f"Selected {len(S)} frames out of {T}:")
    print("Indices:", S)
    print(f"Reconstruction error: {rec:.4f}")
    print(f"Frechet distance    : {fd:.4f}")
    print(f"MS-SWD (color)      : {ms:.4f}")


if __name__ == "__main__":
    example_run()
