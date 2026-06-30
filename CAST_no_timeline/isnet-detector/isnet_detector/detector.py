"""
SimpleISNetDetector - detect foreground objects in images using ISNet.

Usage:
    from isnet_detector import SimpleISNetDetector

    detector = SimpleISNetDetector(model_path="weights/isnetis.ckpt", device="cuda:0")
    objects, mask_binary = detector.detect_objects("frame.jpg")
    detector.visualize("frame.jpg", objects, mask_binary, "output.jpg")
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.cuda import amp

from .anime_seg import AnimeSegmentation


def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def is_contained(box_small, box_large):
    x1_s, y1_s, x2_s, y2_s = box_small
    x1_l, y1_l, x2_l, y2_l = box_large
    return x1_l <= x1_s and y1_l <= y1_s and x2_l >= x2_s and y2_l >= y2_s


def union_bbox(boxes):
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def find_overlap_groups(objects, iou_threshold=0.01):
    n = len(objects)
    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        queue = [i]
        while queue:
            current = queue.pop(0)
            for j in range(n):
                if not visited[j]:
                    iou = compute_iou(objects[current]["bbox"], objects[j]["bbox"])
                    contained = is_contained(objects[j]["bbox"], objects[current]["bbox"]) or \
                                is_contained(objects[current]["bbox"], objects[j]["bbox"])
                    if iou > iou_threshold or contained:
                        visited[j] = True
                        group.append(j)
                        queue.append(j)
        if len(group) > 1:
            groups.append(sorted(group))
    return groups


def remove_contained_objects(objects):
    n = len(objects)
    to_remove = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(n):
            if i != j and j not in to_remove:
                if is_contained(objects[i]["bbox"], objects[j]["bbox"]):
                    to_remove.add(i)
                    break
    return [o for i, o in enumerate(objects) if i not in to_remove], len(to_remove)


class SimpleISNetDetector:
    """
    Pure ISNet-based foreground detector (no grid cells, no complex dependencies).

    Args:
        model_path: Path to isnetis.ckpt checkpoint file.
        device: torch device string, e.g. "cuda:0" or "cpu".
        img_size: Internal inference resolution (default 1024).
    """

    def __init__(self, model_path=None, device="cuda:0", img_size=1024, use_u2net=False):
        self.device = torch.device(device)
        self.img_size = img_size
        self.use_amp = device != "cpu"
        self.use_u2net = use_u2net
        self._u2net_saliency_fn = None

        # Always load ISNet for detection
        print(f"[SimpleISNetDetector] Loading ISNet on {device}")
        if model_path and os.path.exists(model_path):
            print(f"[SimpleISNetDetector] Loading ISNet weights from {model_path}")
            self.model = AnimeSegmentation.try_load(
                net_name="isnet_is",
                ckpt_path=model_path,
                map_location=str(device),
                img_size=img_size,
            )
        else:
            print("[SimpleISNetDetector] Creating ISNet model without pretrained weights")
            self.model = AnimeSegmentation(net_name="isnet_is", img_size=img_size)

        self.model = self.model.to(self.device)
        self.model.eval()

        if use_u2net:
            print(f"[SimpleISNetDetector] Using U2Net backend for segmentation")
            try:
                # Import CAST's U2Net saliency (already has pretrained weights)
                import sys
                cast_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                if cast_root not in sys.path:
                    sys.path.insert(0, cast_root)
                from utils.saliency import compute_saliency_hybrid
                self._u2net_saliency_fn = compute_saliency_hybrid
                print("[SimpleISNetDetector] U2Net saliency loaded successfully")
            except ImportError as e:
                print(f"[SimpleISNetDetector] WARNING: Could not import U2Net saliency: {e}")
                print("[SimpleISNetDetector] Falling back to ISNet for segmentation")
                self.use_u2net = False
        
        print("[SimpleISNetDetector] Models ready")

    @torch.no_grad()
    def _get_isnet_mask(self, input_img: np.ndarray) -> np.ndarray:
        """Return soft saliency map using ISNet in shape (H, W, 1), values in [0, 1]."""
        input_img = (input_img / 255).astype(np.float32)
        h0, w0 = input_img.shape[:2]
        S = self.img_size
        if h0 > w0:
            h, w = S, int(S * w0 / h0)
        else:
            h, w = int(S * h0 / w0), S
        ph, pw = S - h, S - w

        canvas = np.zeros([S, S, 3], dtype=np.float32)
        canvas[ph // 2: ph // 2 + h, pw // 2: pw // 2 + w] = cv2.resize(input_img, (w, h))
        tensor = torch.from_numpy(canvas.transpose(2, 0, 1)[np.newaxis]).float().to(self.device)

        if self.use_amp:
            with amp.autocast():
                pred = self.model(tensor)
            pred = pred.to(torch.float32)
        else:
            pred = self.model(tensor)

        pred = pred.cpu().numpy()[0].transpose(1, 2, 0)
        pred = pred[ph // 2: ph // 2 + h, pw // 2: pw // 2 + w]
        pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
        return pred

    @torch.no_grad()
    def get_mask(self, input_img: np.ndarray) -> np.ndarray:
        """Return soft saliency map in shape (H, W, 1), values in [0, 1]."""
        if self.use_u2net and self._u2net_saliency_fn is not None:
            # Use CAST's proven U2Net saliency
            mask = self._u2net_saliency_fn(
                input_img, prefer_u2net=True, fast_only=False,
                center_bias_strength=0.0, threshold=0.0
            )
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]
            return mask.astype(np.float32)

        # Original ISNet path
        input_img = (input_img / 255).astype(np.float32)
        h0, w0 = input_img.shape[:2]
        S = self.img_size
        if h0 > w0:
            h, w = S, int(S * w0 / h0)
        else:
            h, w = int(S * h0 / w0), S
        ph, pw = S - h, S - w

        canvas = np.zeros([S, S, 3], dtype=np.float32)
        canvas[ph // 2: ph // 2 + h, pw // 2: pw // 2 + w] = cv2.resize(input_img, (w, h))
        tensor = torch.from_numpy(canvas.transpose(2, 0, 1)[np.newaxis]).float().to(self.device)

        if self.use_amp:
            with amp.autocast():
                pred = self.model(tensor)
            pred = pred.to(torch.float32)
        else:
            pred = self.model(tensor)

        pred = pred.cpu().numpy()[0].transpose(1, 2, 0)
        pred = pred[ph // 2: ph // 2 + h, pw // 2: pw // 2 + w]
        pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
        return pred

    def detect_objects(
        self,
        image_path,
        threshold=0.5,
        min_area=1000,
        merge_kernel=21,
        dilate_iter=2,
        erode_iter=1,
        pre_filter_area=100,
        adaptive_morph=False,
    ):
        """
        Detect foreground objects in an image.

        Returns:
            objects: list of dicts with keys bbox, pixel_area, bbox_area, size, confidence
            mask_binary: binary mask (uint8 H×W)
        """
        img_np = np.array(Image.open(image_path).convert("RGB"))
        H, W = img_np.shape[:2]

        mask = self._get_isnet_mask(img_np)
        mask_raw = (mask[:, :, 0] > threshold).astype(np.uint8)

        # Pre-filter tiny noise
        if pre_filter_area > 0:
            cnts, _ = cv2.findContours(mask_raw * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_filtered = np.zeros_like(mask_raw)
            for cnt in cnts:
                if cv2.contourArea(cnt) >= pre_filter_area:
                    cv2.drawContours(mask_filtered, [cnt], -1, 1, -1)
        else:
            mask_filtered = mask_raw

        # Adaptive morphology
        actual_dilate, actual_erode = dilate_iter, erode_iter
        if adaptive_morph:
            fg_ratio = mask_filtered.sum() / (H * W)
            if fg_ratio < 0.01:
                actual_dilate = max(dilate_iter, 3)
                actual_erode = max(1, erode_iter - 1)
            elif fg_ratio > 0.3:
                actual_dilate = max(1, dilate_iter - 1)
                actual_erode = actual_dilate

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_kernel, merge_kernel))
        mask_binary = cv2.dilate(mask_filtered * 255, kernel, iterations=actual_dilate)
        mask_binary = cv2.erode(mask_binary, kernel, iterations=actual_erode)

        if mask_binary.sum() == 0 and mask_filtered.sum() > 0:
            mask_binary = cv2.dilate(mask_filtered * 255, kernel, iterations=actual_dilate)

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            pixel_area = cv2.contourArea(cnt)
            if pixel_area >= min_area:
                confidence = np.mean(mask_binary[y: y + h, x: x + w]) / 255.0
                objects.append({
                    "bbox": (x, y, x + w, y + h),
                    "pixel_area": int(pixel_area),
                    "bbox_area": w * h,
                    "size": (w, h),
                    "confidence": float(confidence),
                })
        objects.sort(key=lambda o: o["pixel_area"], reverse=True)
        return objects, mask_binary

    def visualize(self, image_path, objects, mask_binary, output_path):
        """Overlay mask + draw colored bboxes on the image and save to output_path."""
        img_np = np.array(Image.open(image_path).convert("RGB"))
        vis = img_np.copy()

        # Overlay segmentation mask as semi-transparent cyan layer
        if mask_binary is not None and mask_binary.max() > 0:
            mask_bool = mask_binary > 0
            overlay = vis.copy()
            overlay[mask_bool] = (0, 220, 180)
            vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)

        COLORS = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            (0, 128, 255), (128, 255, 0),
        ]
        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj["bbox"]
            color = COLORS[i % len(COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            label = f"#{i+1} s={obj['confidence']:.2f} px={obj['pixel_area']}"
            cv2.putText(vis, label, (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
