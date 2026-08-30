from collections import deque
from datetime import datetime
from typing import NamedTuple

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    OBJECT_MODEL_PATH,
    OBJECT_MAX_RESULTS,
    OBJECT_SCORE_THRESHOLD,
    OBJECT_MEMORY_SIZE,
)


class Detection(NamedTuple):
    label: str
    score: float
    box: tuple[int, int, int, int]


class Memory(NamedTuple):
    label: str
    at: datetime
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
        self._memory: deque[Memory] = deque(maxlen=OBJECT_MEMORY_SIZE)

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

    def remember(self, det: Detection) -> None:
        self._memory.append(Memory(label=det.label, at=datetime.now(), box=det.box))

    def seen(self, query: str) -> Memory | None:
        lowered = query.lower()
        collapsed = lowered.replace(" ", "")
        for item in reversed(self._memory):
            name = item.label.replace("_", " ")
            if name in lowered or name.replace(" ", "") in collapsed:
                return item
        return None
