import pytest

from action.vla import PointResult, VLAManager
from body.poses import Gesture, Joints, Mood, poses
from config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    POINT_DURATION,
    POINT_HOLD,
    POINT_NECK_YAW_LEVEL,
    POINT_NECK_YAW_SIGN,
    POINT_NECK_YAW_SPAN,
    POINT_PITCH_LEVEL,
    POINT_PITCH_SPAN,
)
from fakes import FakeVision
from perception.vision_brain import Detection

DT = 0.05
SIZE = 40


def detection_at(nx: float, ny: float, label: str = "cup") -> Detection:
    cx = (nx + 1) / 2 * CAMERA_WIDTH
    cy = (ny + 1) / 2 * CAMERA_HEIGHT
    box = (round(cx - SIZE / 2), round(cy - SIZE / 2), SIZE, SIZE)
    return Detection(label, 0.9, box)


class ScriptedVision(FakeVision):
    def __init__(self, *frames: list[Detection]) -> None:
        super().__init__()
        self._frames = list(frames)

    def detect(self, frame) -> list[Detection]:
        if len(self._frames) > 1:
            return self._frames.pop(0)
        return self._frames[0]


def run(vla: VLAManager, frame, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        vla.step(DT, frame)


def arrive(vla: VLAManager, frame) -> None:
    run(vla, frame, POINT_DURATION + 0.2)


def head(lamp) -> dict[str, float]:
    return lamp.get_joints([Joints.NECK_YAW, Joints.HEAD_PITCH])


def test_unseen_label_reports_missing_without_moving(lamp, frame):
    vla = VLAManager(lamp, FakeVision([detection_at(0, 0, "cup")]), Mood.NEUTRAL)
    vla.point_at_label("point at the book", frame)
    assert not vla.active
    assert vla.take_result() == PointResult("", found=False, centered=False)


def test_centered_object_aims_at_level_pose(lamp, frame):
    vla = VLAManager(lamp, FakeVision([detection_at(0, 0)]), Mood.NEUTRAL)
    vla.point_at_label("point at the cup", frame)
    assert vla.active

    arrive(vla, frame)

    assert head(lamp)[Joints.NECK_YAW] == pytest.approx(POINT_NECK_YAW_LEVEL)
    assert head(lamp)[Joints.HEAD_PITCH] == pytest.approx(POINT_PITCH_LEVEL)
    assert vla.take_result() == PointResult("cup", found=True, centered=True)


@pytest.mark.parametrize("nx, ny", [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)])
def test_box_position_maps_to_neck_and_head(lamp, frame, nx, ny):
    vla = VLAManager(lamp, FakeVision([detection_at(nx, ny)]), Mood.NEUTRAL)
    vla.point_at_label("point at the cup", frame)
    arrive(vla, frame)

    expected_yaw = POINT_NECK_YAW_LEVEL - POINT_NECK_YAW_SIGN * nx * POINT_NECK_YAW_SPAN
    expected_pitch = POINT_PITCH_LEVEL + ny * POINT_PITCH_SPAN
    assert head(lamp)[Joints.NECK_YAW] == pytest.approx(expected_yaw, abs=0.01)
    assert head(lamp)[Joints.HEAD_PITCH] == pytest.approx(expected_pitch, abs=0.01)


def test_result_is_delivered_once(lamp, frame):
    vla = VLAManager(lamp, FakeVision([detection_at(0, 0)]), Mood.NEUTRAL)
    vla.point_at_label("point at the cup", frame)
    arrive(vla, frame)
    assert vla.take_result() is not None
    assert vla.take_result() is None


def test_full_cycle_returns_to_engage_pose(lamp, frame):
    vla = VLAManager(lamp, FakeVision([detection_at(0.5, 0.5)]), Mood.NEUTRAL)
    vla.point_at_label("point at the cup", frame)

    run(vla, frame, POINT_DURATION + POINT_HOLD + POINT_DURATION + 0.5)

    assert not vla.active
    engage = poses[Gesture.ENGAGE]["joints"]
    for joint, angle in lamp.get_joints(engage.keys()).items():
        assert angle == pytest.approx(engage[joint], abs=0.01)


def test_drifting_object_is_corrected_once_then_reported_off_center(lamp, frame):
    vision = ScriptedVision(
        [detection_at(0, 0)], [detection_at(0.6, 0)], [detection_at(-0.6, 0)]
    )
    vla = VLAManager(lamp, vision, Mood.NEUTRAL)
    vla.point_at_label("point at the cup", frame)

    arrive(vla, frame)
    assert vla.take_result() is None

    arrive(vla, frame)
    assert vla.take_result() == PointResult("cup", found=True, centered=False)


def test_ignores_new_request_while_pointing(lamp, frame):
    vision = FakeVision([detection_at(0, 0, "cup"), detection_at(0.5, 0, "book")])
    vla = VLAManager(lamp, vision, Mood.NEUTRAL)
    vla.point_at_label("point at the cup", frame)
    vla.point_at_label("point at the book", frame)

    arrive(vla, frame)

    assert vla.take_result().label == "cup"


def test_cancel_stops_without_a_result(lamp, frame):
    vla = VLAManager(lamp, FakeVision([detection_at(0, 0)]), Mood.NEUTRAL)
    vla.point_at_label("point at the cup", frame)
    run(vla, frame, POINT_DURATION / 2)

    vla.cancel()

    assert not vla.active
    assert vla.take_result() is None


def test_pointing_at_remembered_box_holds_without_reporting(lamp, frame):
    vla = VLAManager(lamp, FakeVision(), Mood.NEUTRAL)
    vla.point_at_box(detection_at(0.5, 0).box)
    assert vla.active

    arrive(vla, frame)

    assert vla.active
    assert vla.take_result() is None
