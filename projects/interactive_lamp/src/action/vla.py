from enum import StrEnum
from typing import NamedTuple

import numpy as np

from perception.vision_brain import VisionBrain
from body.lamp_body import LampBody
from body.lamp_gesture import LampGesture
from body.poses import Joints, Mood, poses
from config import (
    POINT_DURATION,
    POINT_NECK_YAW_SPAN,
    POINT_PITCH_LEVEL,
    POINT_PITCH_SPAN,
    POINT_NECK_YAW_SIGN,
    POINT_CENTER_TOLERANCE,
    POINT_MAX_CORRECTIONS,
)


class VLAState(StrEnum):
    IDLE = "idle"
    MOVING = "moving"
    REOBSERVE = "reobserve"
    DONE = "done"


class PointResult(NamedTuple):
    label: str
    found: bool
    centered: bool


class VLAManager:
    def __init__(self, lamp: LampBody, vision: VisionBrain, mood: Mood) -> None:
        self._lamp = lamp
        self._vision = vision
        self._gesture = LampGesture(lamp, mood=mood)
        self._state = VLAState.IDLE
        self._goal_label: str | None = None
        self._live = False
        self._corrections = 0
        self._result: PointResult | None = None

    @property
    def active(self) -> bool:
        return self._state not in (VLAState.IDLE, VLAState.DONE)

    def point_at_label(self, text: str, frame: np.ndarray) -> None:
        # live goal: detect objects in this frame, pick the one whose label
        # appears in `text` (the utterance), move toward its box, arm re-observe.
        # if no detected object is named in `text` -> finish as not-found.
        raise NotImplementedError

    def point_at_box(self, box: tuple[int, int, int, int]) -> None:
        # memory recall: point where it was, no re-observe.
        raise NotImplementedError

    def step(self, dt: float, frame: np.ndarray) -> None:
        # advance the pointing motion; on arrival, re-detect (live only),
        # correct once if off-center, then settle to DONE.
        raise NotImplementedError

    def take_result(self) -> PointResult | None:
        # hand the finished PointResult to main once, then reset to IDLE.
        raise NotImplementedError

    def _start_move(self, box: tuple[int, int, int, int], frame_shape) -> None:
        # compute target joints from the box and hand them to the gesture player.
        raise NotImplementedError

    def _target_joints(self, box, frame_shape) -> dict[str, float]:
        cx, cy = self._box_center_norm(box, frame_shape)
        # map normalized center -> neck yaw (pan) + head pitch (tilt), hold the rest.
        raise NotImplementedError

    def _box_center_norm(self, box, frame_shape) -> tuple[float, float]:
        # box = (x, y, w, h) px; return center in [-1, 1] x/y frame coords.
        raise NotImplementedError
