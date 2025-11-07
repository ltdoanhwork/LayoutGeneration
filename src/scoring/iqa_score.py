# ---- Optional: IQA metrics ----
from PIL import Image
from utils.process_images import _to_numpy_rgb, load_image_from_path
from typing import Union, Optional
import numpy as np
import math
import cv2
# 1) BRISQUE (no-reference)
try:
    from imquality import brisque as _brisque
    _BRISQUE_OK = True
except Exception:
    _BRISQUE_OK = False

# 2) NIQE (no-reference) via skimage
try:
    from skimage.metrics import niqe as _niqe
    from skimage.color import rgb2gray
    _NIQE_OK = True
except Exception:
    _NIQE_OK = False

# ----------------------------
# IQA scoring
# ----------------------------
def _iqa_brisque(pil_img: Image.Image) -> Optional[float]:
    """Lower is better (BRISQUE). Return None if fails."""
    if not _BRISQUE_OK:
        return None
    try:
        # imquality expects BGR ndarray
        arr = _to_numpy_rgb(pil_img)
        if cv2 is not None:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            # fallback: approximate BGR as RGB (minor effect)
            bgr = arr[:, :, ::-1]
        score = float(_brisque.score(bgr))
        return score
    except Exception:
        return None


def _iqa_niqe(pil_img: Image.Image) -> Optional[float]:
    """Lower is better (NIQE). Return None if fails."""
    if not _NIQE_OK:
        return None
    try:
        arr = _to_numpy_rgb(pil_img).astype(np.float32) / 255.0
        gray = rgb2gray(arr)
        score = float(_niqe(gray))
        return score
    except Exception:
        return None


def _iqa_heuristics(pil_img: Image.Image) -> float:
    """
    Heuristic IQA proxy (higher is better):
    combine sharpness (variance of Laplacian) + exposure balance.
    """
    arr = _to_numpy_rgb(pil_img)
    if cv2 is not None:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        # simple finite-diff gradient magnitude
        g = arr.mean(axis=2)
        dx = np.abs(np.diff(g, axis=1)).mean()
        dy = np.abs(np.diff(g, axis=0)).mean()
        sharp = float(dx + dy)

    # exposure balance: penalize too dark/too bright
    mean_luma = arr.mean()
    exposure_pen = math.exp(-((mean_luma - 128.0) ** 2) / (2 * (64.0 ** 2)))
    # normalize roughly
    return float((sharp / (sharp + 50.0)) * exposure_pen)


def get_iqa_score(image) -> float:
    """
    Returns a scalar where higher = better quality (normalize if using BRISQUE/NIQE).
    We invert BRISQUE/NIQE to a [0,1]-like score for consistency.
    """
    pil = image if isinstance(image, Image.Image) else Image.fromarray(_to_numpy_rgb(image))

    # Try BRISQUE first
    s_b = _iqa_brisque(pil)
    if s_b is not None and np.isfinite(s_b):
        # BRISQUE lower=better => map to (0,1] by 1 / (1 + score)
        return float(1.0 / (1.0 + max(0.0, s_b)))

    # Then NIQE
    s_n = _iqa_niqe(pil)
    if s_n is not None and np.isfinite(s_n):
        return float(1.0 / (1.0 + max(0.0, s_n)))

    # Fallback heuristic (already higher=better)
    return _iqa_heuristics(pil)