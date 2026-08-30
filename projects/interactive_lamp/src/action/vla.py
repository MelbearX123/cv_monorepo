import re
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from perception.vision_brain import VisionBrain
from body.lamp_body import LampBody
from body.lamp_gesture import LampGesture
from body.poses import Joints, Mood, poses
from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    POINT_DURATION,
    POINT_NECK_YAW_LEVEL,
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
        self._frame_shape: tuple[int, ...] | None = None
        self._aim_center: tuple[float, float] = (0.0, 0.0)

    @property
    def active(self) -> bool:
        return self._state != VLAState.IDLE

    def point_at_label(self, text: str, frame: np.ndarray) -> None:
        self._frame_shape = frame.shape
        self._live = True
        self._corrections = 0

        lowered = text.lower()
        matches = [
            d
            for d in self._vision.detect(frame)
            if self._label_spoken(d.label, lowered)
        ]
        if not matches:
            self._goal_label = None
            self._result = PointResult(label="", found=False, centered=False)
            return

        target = max(matches, key=lambda d: d.score)
        self._goal_label = target.label
        self._start_move(target.box, frame.shape)

    def point_at_box(self, box: tuple[int, int, int, int]) -> None:
        self._live = False
        self._goal_label = None
        self._corrections = 0
        self._result = None
        self._start_move(box, self._frame_shape or (CAMERA_HEIGHT, CAMERA_WIDTH))

    def step(self, dt: float, frame: np.ndarray) -> None:
        self._gesture.step(dt)
        if self._gesture.active:
            return
        if not self._live:
            self._state = VLAState.IDLE
            return
        self._resolve(frame)

    def take_result(self) -> PointResult | None:
        if self._result is None:
            return None
        result = self._result
        self._reset()
        return result

    @staticmethod
    def _label_spoken(label: str, lowered: str) -> bool:
        phrase = re.escape(label.replace("_", " "))
        return re.search(rf"\b{phrase}\b", lowered) is not None

    def _resolve(self, frame: np.ndarray) -> None:
        matches = [d for d in self._vision.detect(frame) if d.label == self._goal_label]
        if not matches:
            self._finish(centered=False)
            return
        target = max(matches, key=lambda d: d.score)
        cx, cy = self._box_center_norm(target.box, self._frame_shape)
        drift = max(abs(cx - self._aim_center[0]), abs(cy - self._aim_center[1]))
        if drift <= POINT_CENTER_TOLERANCE or self._corrections >= POINT_MAX_CORRECTIONS:
            self._finish(centered=drift <= POINT_CENTER_TOLERANCE)
            return
        self._corrections += 1
        self._start_move(target.box, self._frame_shape)

    def _finish(self, centered: bool) -> None:
        self._result = PointResult(self._goal_label or "", found=True, centered=centered)
        self._state = VLAState.IDLE

    def _reset(self) -> None:
        self._state = VLAState.IDLE
        self._goal_label = None
        self._corrections = 0
        self._result = None

    def _start_move(self, box: tuple[int, int, int, int], frame_shape) -> None:
        self._state = VLAState.MOVING
        self._aim_center = self._box_center_norm(box, frame_shape)
        joints = self._target_joints(box, frame_shape)
        self._gesture.play_target(joints, poses["engage"]["light"], POINT_DURATION)

    def _target_joints(self, box, frame_shape) -> dict[str, float]:
        nx, ny = self._box_center_norm(box, frame_shape)
        held = poses["engage"]["joints"]
        return {
            Joints.BASE_YAW: held[Joints.BASE_YAW],
            Joints.SHOULDER_PITCH: held[Joints.SHOULDER_PITCH],
            Joints.ELBOW_PITCH: held[Joints.ELBOW_PITCH],
            Joints.NECK_YAW: POINT_NECK_YAW_LEVEL - POINT_NECK_YAW_SIGN * nx * POINT_NECK_YAW_SPAN,  # mirror
            Joints.HEAD_PITCH: POINT_PITCH_LEVEL + ny * POINT_PITCH_SPAN,
        }

    def _box_center_norm(self, box, frame_shape) -> tuple[float, float]:
        x, y, w, h = box
        height, width = frame_shape[0], frame_shape[1]
        nx = (x + w / 2) / width * 2 - 1
        ny = (y + h / 2) / height * 2 - 1
        return nx, ny
