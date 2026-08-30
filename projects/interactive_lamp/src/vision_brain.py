from typing import NamedTuple

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import OBJECT_MODEL_PATH, OBJECT_MAX_RESULTS, OBJECT_SCORE_THRESHOLD


class Detection(NamedTuple):
    label: str
    score: float
    box: tuple[int, int, int, int]


class VisionBrain:
    def __init__(self) -> None:
        base_options = python.BaseOptions(model_asset_path=str(OBJECT_MODEL_PATH))
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            max_results=OBJECT_MAX_RESULTS,
            score_threshold=OBJECT_SCORE_THRESHOLD,
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray) -> list[Detection]:
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(mp_frame)
        return [
            Detection(
                label=d.categories[0].category_name,
                score=d.categories[0].score,
                box=(
                    d.bounding_box.origin_x,
                    d.bounding_box.origin_y,
                    d.bounding_box.width,
                    d.bounding_box.height,
                ),
            )
            for d in result.detections
        ]

    def main_object(self, rgb_frame: np.ndarray) -> Detection | None:
        objects = [d for d in self.detect(rgb_frame) if d.label != "person"]
        if not objects:
            return None
        return max(objects, key=lambda d: d.score)
