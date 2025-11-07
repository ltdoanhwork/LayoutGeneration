# All comments are in English as requested.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T


@dataclass(frozen=True)
class FrameSample:
    frame_idx: int
    scene_id: Optional[int] = None

class TorchBackbone(nn.Module):
    """A thin wrapper to get a global-pooled embedding from torchvision models."""
    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        name = name.lower()
        if name == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
            self.backbone = nn.Sequential(*list(net.children())[:-2])  # up to conv5_x
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = feat_dim
        elif name in ("vit_b_16", "vit-b-16"):
            net = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT if pretrained else None)
            self.backbone = net
            self.out_dim = net.heads.head.in_features  # 768
        else:
            raise ValueError(f"Unknown backbone: {name}")
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "resnet50":
            f = self.backbone(x)      # (B, C, H', W')
            f = self.pool(f).flatten(1)
            return f                   # (B, 2048)
        elif self.name.startswith("vit"):
            # ViT forward returns class logits; we extract penultimate features
            # Trick: call forward_features if available
            if hasattr(self.backbone, "forward_features"):
                f = self.backbone.forward_features(x)  # (B, C, H', W') or dict
                if isinstance(f, dict) and "x" in f:
                    f = f["x"]
                if f.ndim == 3:
                    f = f[:, 0]  # CLS
                return f
            else:
                # Fallback to full forward then strip head
                raise RuntimeError("This torchvision ViT does not expose forward_features.")
        else:
            raise RuntimeError("Unreachable")


class FeatureExtractor:
    """
    Video frame sampler + embedding extractor.
    Uses torchvision backbones (default: resnet50) with ImageNet normalization.
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (224, 224),
        pretrained: bool = True,
    ):
        dev = device if device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.input_size = input_size
        self.model = TorchBackbone(backbone, pretrained=pretrained).to(self.device).eval()
        self.transform = T.Compose([
            T.ToTensor(),  # [0,1], CHW
            T.Resize(self.input_size, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed_images(self, bgr_images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized embeddings (N, D) as float32 numpy array."""
        if not bgr_images:
            return np.zeros((0, self.model.out_dim), dtype=np.float32)

        # Convert BGR->RGB and apply transforms
        def _prep(img: np.ndarray) -> torch.Tensor:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = T.functional.to_pil_image(rgb)
            ten = self.transform(pil)  # (3,H,W)
            return ten

        tensors = [ _prep(im) for im in bgr_images ]
        embs: List[torch.Tensor] = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i+batch_size], dim=0).to(self.device)
            f = self.model(batch)  # (B, D)
            f = torch.nn.functional.normalize(f, dim=1)
            embs.append(f.detach().cpu())
        Fe = torch.cat(embs, dim=0).numpy().astype(np.float32)
        return Fe

    def sample_frames_by_stride(
        self, video_path: str, stride: int, max_frames: Optional[int] = None
    ) -> List[FrameSample]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idx = list(range(0, total, max(1, stride)))
        if max_frames and len(idx) > max_frames:
            sel = np.linspace(0, len(idx)-1, max_frames, dtype=int)
            idx = [idx[i] for i in sel]
        cap.release()
        return [FrameSample(i) for i in idx]

    def load_frames(self, video_path: str, samples: Sequence[FrameSample], resize_to: Optional[Tuple[int,int]]=None) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        out: List[np.ndarray] = []
        for s in samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s.frame_idx)
            ok, f = cap.read()
            if not ok or f is None:
                continue
            if resize_to is not None:
                f = cv2.resize(f, resize_to, interpolation=cv2.INTER_AREA)
            out.append(f)
        cap.release()
        return out
