# src/scene_detection/transnetv2_backend.py
import os
import torch
from typing import List
from .interface import Scene, Scenes, SceneDetector
from .registry import register_detector
from src.models.TransNetV2.inference_pytorch.transnetv2_pytorch import TransNetV2

@register_detector("transnetv2")
class TransNetV2Detector(SceneDetector):
    """
    Detect scenes using TransNetV2 PyTorch model.
    Parameters:
      - weights_path: path to PyTorch model weights file
      - prob_threshold: boundary threshold (default 0.5)
      - device: device to run inference on ('cuda' or 'cpu', default: 'cuda' if available)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        model_dir = self.params.get("model_dir")
        if not model_dir:
            raise ValueError("Missing 'model_dir' for TransNetV2Detector.")

        device = self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        weights_path = os.path.join(os.path.expanduser(model_dir), "inference_pytorch", "transnetv2-pytorch-weights.pth")

        self._model = TransNetV2()
        state_dict = torch.load(weights_path)
        self._model.load_state_dict(state_dict)
        self._model.eval().to(device)
        self._device = device
            
        

    def detect(self, video_path: str) -> Scenes:
        self._lazy_load()
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Resize frame to 27x48 pixels (required input size for TransNetV2)
            frame = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            return []

        # Convert frames to PyTorch tensor
        with torch.no_grad():
            # shape: batch x frames x height x width x channels
            video_tensor = torch.from_numpy(np.stack(frames)).unsqueeze(0)
            video_tensor = video_tensor.to(self._device, dtype=torch.uint8)
            
            # Run inference
            single_frame_pred, all_frame_pred = self._model(video_tensor)
            
            # Process predictions
            single_frame_pred = torch.sigmoid(single_frame_pred)
            all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"])
            
            # Convert to numpy for further processing
            y = single_frame_pred.squeeze().cpu().numpy()

        thr = float(self.params.get("prob_threshold", 0.5))
        boundaries = np.where(y > thr)[0].tolist()

        scenes: List[Scene] = []
        prev = 0
        T = len(frames)
        for b in boundaries:
            if b > prev:
                scenes.append(Scene(prev, b))
                prev = b + 1
        if prev < T:
            scenes.append(Scene(prev, T - 1))
        return scenes

    def close(self) -> None:
        # Nothing more needed for TF SavedModel; kept for future use.
        self._model = None
