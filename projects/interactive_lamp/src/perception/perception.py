import numpy as np

from perception.face_tracker import FaceTracker
from perception.face_reader import FaceReader, FaceReading


class FacePerception:
    def __init__(self) -> None:
        self._tracker = FaceTracker()
        self._reader = FaceReader()

    def read(self, frame: np.ndarray | None, dt: float) -> FaceReading | None:
        if frame is None:
            return None
        return self._reader.read(self._tracker.process(frame), dt)
