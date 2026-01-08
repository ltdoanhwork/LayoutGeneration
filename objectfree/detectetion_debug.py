"""
FULL PIPELINE: SAM3 + DINOv3 + ONLINE IDENTITY MANAGER
=====================================================

INPUT
-----
- 1 folder chứa ảnh (keyframes, theo thứ tự thời gian)

PIPELINE
--------
1. SAM3 text-prompt detection (STATEFUL API)
2. Quality gate (score + bbox size)
3. DINOv3 embedding
4. IdentityManager (online, cosine mean, temporal)
5. Giữ KEY CHARACTER
6. Print thống kê chi tiết

OUTPUT
------
- debug_output/
    ├── kept/          (ảnh crop được giữ – key character)
    └── rejected/      (ảnh bị loại)
"""

# =========================================================
# ========================= IMPORT ========================
# =========================================================
import os
import gc
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List
from PIL import Image
from huggingface_hub import login
from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics.pairwise import cosine_similarity

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================================================
# ========================= CONFIG ========================
# =========================================================
CONFIG = {
    "sam3": {
        "huggingface_token": os.getenv("HF_TOKEN_SAM3", ""),     # <<< TOKEN SAM3
        "text_prompt": "character",
        "score_thresh": 0.7,
    },
    "dinov3": {
        "token": os.getenv("HF_TOKEN_DINO", ""),                 # <<< TOKEN DINOv3
        "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "cache_dir": "./weights/dinov3",
    },
    "idm": {
        "sim_threshold": 0.78,
        "max_embeds_per_id": 15,
    },
    "io": {
        "input_folder": "/home/serverai/ltdoanh/LayoutGeneration/outputs/inference_v11/124161/keyframes_selected_from_video",
        "output_root": "./debug_output3",
    },
    "runtime": {
        "min_box_size": 50,
        "max_images": None,
        "verbose": True,
    }
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# ===================== LOGIN & LOAD ======================
# =========================================================
if CONFIG["sam3"]["huggingface_token"]:
    login(token=CONFIG["sam3"]["huggingface_token"])

print("[INFO] Loading SAM3...")
sam_model = build_sam3_image_model().to(DEVICE).eval()
sam_processor = Sam3Processor(sam_model)
TEXT_PROMPT = CONFIG["sam3"]["text_prompt"]
SAM_SCORE_THRESH = CONFIG["sam3"]["score_thresh"]

print("[INFO] Loading DINOv3...")
dino_model = AutoModel.from_pretrained(
    CONFIG["dinov3"]["model_name"],
    token=CONFIG["dinov3"]["token"] if CONFIG["dinov3"]["token"] else None,
    cache_dir=CONFIG["dinov3"]["cache_dir"],
    trust_remote_code=True
).to(DEVICE).eval()

dino_processor = AutoImageProcessor.from_pretrained(
    CONFIG["dinov3"]["model_name"],
    token=CONFIG["dinov3"]["token"] if CONFIG["dinov3"]["token"] else None,
    cache_dir=CONFIG["dinov3"]["cache_dir"],
    trust_remote_code=True
)

print("[INFO] Models ready")


# =========================================================
# ==================== IDENTITY MANAGER ===================
# =========================================================
class IdentityManager:
    def __init__(self, sim_threshold, max_embeds_per_id):
        self.sim_threshold = sim_threshold
        self.max_embeds = max_embeds_per_id

        self.next_id = 0
        self.embeds = {}      # id -> [embeddings]

        # statistics
        self.total_detected = 0
        self.total_kept = 0
        self.total_rejected = 0

    def _mean_similarity(self, emb_new, emb_list):
        sims = cosine_similarity(
            emb_new[None],
            np.stack(emb_list)
        )[0]
        return float(np.mean(sims))

    def assign(self, emb_new, quality_ok=True):
        self.total_detected += 1

        if not quality_ok:
            self.total_rejected += 1
            return None, 0.0, "rejected_low_quality"

        best_id = None
        best_sim = -1.0

        for assigned_id in self.embeds.keys():
            sim = self._mean_similarity(
                emb_new, self.embeds[assigned_id]
            )
            if sim > best_sim:
                best_sim = sim
                best_id = assigned_id

        if best_id is not None and best_sim >= self.sim_threshold:
            self.embeds[best_id].append(emb_new)
            if len(self.embeds[best_id]) > self.max_embeds:
                self.embeds[best_id].pop(0)

            self.total_kept += 1
            return best_id, best_sim, "matched"

        # create new identity
        new_id = self.next_id
        self.next_id += 1
        self.embeds[new_id] = [emb_new]

        self.total_kept += 1
        return new_id, 0.0, "new_id"

    def summary(self):
        return {
            "detected": self.total_detected,
            "kept": self.total_kept,
            "rejected": self.total_rejected,
            "identities": len(self.embeds),
        }


# =========================================================
# ========================= UTILS =========================
# =========================================================
def list_images(folder: Path):
    imgs = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if CONFIG["runtime"]["max_images"]:
        imgs = imgs[:CONFIG["runtime"]["max_images"]]
    return imgs


@torch.no_grad()
def sam_detect(pil_img):
    state = sam_processor.set_image(pil_img)
    out = sam_processor.set_text_prompt(state=state, prompt=TEXT_PROMPT)
    boxes = out.get("boxes", [])
    scores = out.get("scores", [])
    return boxes.detach().cpu().tolist(), scores.detach().cpu().tolist()


@torch.no_grad()
def dino_extract(crops):
    rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in crops]
    inp = dino_processor(rgb, return_tensors="pt").to(DEVICE)
    out = dino_model(**inp)
    feat = out.last_hidden_state[:, 0]
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat.cpu().numpy()


def crop_box(img, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if (x2 - x1) < CONFIG["runtime"]["min_box_size"] or (y2 - y1) < CONFIG["runtime"]["min_box_size"]:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


# =========================================================
# =========================== MAIN ========================
# =========================================================
def main():
    input_dir = Path(CONFIG["io"]["input_folder"])
    out_root = Path(CONFIG["io"]["output_root"])
    kept_dir = out_root / "kept"
    rej_dir = out_root / "rejected"
    kept_dir.mkdir(parents=True, exist_ok=True)
    rej_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images(input_dir)
    print(f"[INFO] Found {len(imgs)} images")

    idm = IdentityManager(
        sim_threshold=CONFIG["idm"]["sim_threshold"],
        max_embeds_per_id=CONFIG["idm"]["max_embeds_per_id"]
    )

    for idx, img_path in enumerate(imgs):
        pil = Image.open(img_path).convert("RGB")
        img = cv2.imread(str(img_path))

        boxes, scores = sam_detect(pil)
        if CONFIG["runtime"]["verbose"]:
            print(f"[{idx:04d}] {img_path.name} | detected={len(boxes)}")

        crops = []
        meta = []   # mỗi phần tử là dict

        for b, s in zip(boxes, scores):
            crop = crop_box(img, b)
            if crop is None:
                idm.total_detected += 1
                idm.total_rejected += 1
                continue

            crops.append(crop)
            meta.append({
                "crop": crop,
                "score": float(s),
                "bbox": b,
                "img_name": img_path.name,
            })

        if not crops:
            continue

        embs = dino_extract(crops)

        for emb, m in zip(embs, meta):
            quality_ok = m["score"] >= SAM_SCORE_THRESH

            cid, sim, action = idm.assign(
                emb_new=emb,
                quality_ok=quality_ok
            )


            if cid is not None:
                out = kept_dir / f"id{cid:03d}_{img_path.stem}.jpg"
                cv2.imwrite(str(out), m["crop"])
            else:
                out = rej_dir / f"{img_path.stem}.jpg"
                cv2.imwrite(str(out), m["crop"])

        gc.collect()
        torch.cuda.empty_cache()

    # ===== SUMMARY =====
    s = idm.summary()
    print("\n========== SUMMARY ==========")
    print(f"Total detected objects : {s['detected']}")
    print(f"Total kept (key chars) : {s['kept']}")
    print(f"Total rejected         : {s['rejected']}")
    print(f"Total identities       : {s['identities']}")
    print("================================\n")


if __name__ == "__main__":
    main()
