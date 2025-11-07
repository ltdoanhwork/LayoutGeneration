# src/scene_detection/pyscenedetect_backend.py
from typing import List
from .interface import Scene, Scenes, SceneDetector
from .registry import register_detector

@register_detector("pyscenedetect")
class PySceneDetectDetector(SceneDetector):
    """
    Detect scenes using PySceneDetect ContentDetector.
    Common parameters:
      - threshold: float = 27.0
    """
    def detect(self, video_path: str) -> Scenes:
        try:
            from scenedetect import VideoManager, SceneManager
            from scenedetect.detectors import ContentDetector
            import cv2
        except Exception as e:
            raise RuntimeError("PySceneDetect is not installed. pip install scenedetect") from e

        threshold = float(self.params.get("threshold", 27.0))

        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()
        video_manager.release()

        # Convert timecode to frame index
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        scenes: List[Scene] = []
        for start_time, end_time in scene_list:
            s = int(start_time.get_seconds() * fps)
            e = int(end_time.get_seconds() * fps) - 1
            if e < s: e = s
            scenes.append(Scene(s, e))

        # Fallback: if no scenes detected, return the full video
        if not scenes:
            # Don't read total_frames here to keep backend lightweight; pipeline can fallback later
            pass
        return scenes
