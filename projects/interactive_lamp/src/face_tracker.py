"""FaceTracker: wraps MediaPipe FaceLandmarker.

Takes an RGB frame (from Camera, already converted) and returns the detection
result for the first face -- landmarks (presence), blendshapes (expression), and
the head-pose matrix -- or None if no face is found. Interpreting those outputs
into emotion / engagement lives elsewhere; this class only runs the model.
"""

import time

import mediapipe as mp
from cv2.typing import MatLike
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

from config import (
    MODEL_PATH,
    MAX_NUM_FACES,
    MIN_FACE_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

class FaceTracker:
    def __init__(self) -> None:
        self._base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        self._options = vision.FaceLandmarkerOptions(
            base_options=self._base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=MAX_NUM_FACES,
            min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=True,                 # -> emotion signal
            output_facial_transformation_matrixes=True,   # -> head pose (looking at me?)
        )
        self._detector = vision.FaceLandmarker.create_from_options(self._options)
        self._last_timestamp_ms = -1

    def process(self, rgb_frame: MatLike) -> FaceLandmarkerResult | None:
        """Run detection on an RGB frame. Return the result, or None if no face."""
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(self._last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
        self._last_timestamp_ms = timestamp_ms
        result = self._detector.detect_for_video(mp_frame, timestamp_ms)
        if not result.face_landmarks:
            return None
        return result
