from collections import Counter
from typing import NamedTuple

import numpy as np

from perception.vision_brain import Detection, VisionBrain
from config import LOOK_CONFIRM_FRAMES


class LookResult(NamedTuple):
    label: str | None


class LookBehavior:
    def __init__(self, vision: VisionBrain, frames: int = LOOK_CONFIRM_FRAMES) -> None:
        self._vision = vision
        self._frames = frames
        self._frames_left = 0
        self._votes: list[Detection] = []
        self._result: LookResult | None = None

    @property
    def active(self) -> bool:
        return self._frames_left > 0

    def start(self) -> None:
        self._votes = []
        self._result = None
        self._frames_left = self._frames

    def cancel(self) -> None:
        self._votes = []
        self._result = None
        self._frames_left = 0

    def step(self, frame: np.ndarray | None) -> None:
        if not self.active:
            return
        if frame is not None:
            detection = self._vision.main_object(frame)
            if detection is not None:
                self._votes.append(detection)
        self._frames_left -= 1
        if not self.active:
            self._result = LookResult(label=self._settle())

    def take_result(self) -> LookResult | None:
        result, self._result = self._result, None
        return result

    def _settle(self) -> str | None:
        if not self._votes:
            return None
        label = Counter(d.label for d in self._votes).most_common(1)[0][0]
        latest = next(d for d in reversed(self._votes) if d.label == label)
        self._vision.remember(latest)
        return label
