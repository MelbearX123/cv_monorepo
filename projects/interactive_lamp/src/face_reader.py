import math
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

from poses import Mood
from config import (
    ENGAGE_YAW,
    DISENGAGE_YAW,
    ENGAGE_DWELL,
    DISENGAGE_DWELL,
    SMILE_THRESHOLD,
    SAD_THRESHOLD,
)


class FaceReader:
    def __init__(
        self,
        engage_yaw: float = ENGAGE_YAW,
        disengage_yaw: float = DISENGAGE_YAW,
        engage_dwell: float = ENGAGE_DWELL,
        disengage_dwell: float = DISENGAGE_DWELL,
    ) -> None:
        self._engage_yaw = engage_yaw
        self._disengage_yaw = disengage_yaw
        self._engage_dwell = engage_dwell
        self._disengage_dwell = disengage_dwell

        self._is_engaged = False
        self._looking_time = 0.0
        self._away_time = 0.0

    def read(self, result: FaceLandmarkerResult | None, dt: float) -> tuple[bool, Mood]:
        looking = self._is_looking(result)
        is_engaged = self._update_engagement(looking, dt)
        mood = self._read_emotion(result)
        return is_engaged, mood

    def _is_looking(self, result: FaceLandmarkerResult | None) -> bool:
        if result is None or not result.facial_transformation_matrixes:
            return False
        matrix = result.facial_transformation_matrixes[0]
        yaw = math.atan2(matrix[0][2], matrix[2][2])
        limit = self._disengage_yaw if self._is_engaged else self._engage_yaw
        return abs(yaw) < limit

    def _update_engagement(self, looking: bool, dt: float) -> bool:
        if looking:
            self._looking_time += dt
            self._away_time = 0.0
        else:
            self._away_time += dt
            self._looking_time = 0.0
        if not self._is_engaged and self._looking_time >= self._engage_dwell:
            self._is_engaged = True
        elif self._is_engaged and self._away_time >= self._disengage_dwell:
            self._is_engaged = False
        return self._is_engaged

    def _read_emotion(self, result: FaceLandmarkerResult | None) -> Mood:
        if result is None or not result.face_blendshapes:
            return Mood.NEUTRAL
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        smile = max(
            scores.get("mouthSmileLeft", 0.0), scores.get("mouthSmileRight", 0.0)
        )
        sad = scores.get("browInnerUp", 0.0)  # brow raise is sad signal
        if smile > SMILE_THRESHOLD:
            return Mood.HAPPY
        if sad > SAD_THRESHOLD:
            return Mood.SAD
        return Mood.NEUTRAL
