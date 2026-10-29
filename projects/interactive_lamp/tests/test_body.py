import pytest

from body.easing import ease_in_out, ease_in_out_back
from body.lamp_gesture import LampGesture
from body.poses import Gesture, Joints, Mood, gestures, poses

DT = 0.05


def run(gesture: LampGesture, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        gesture.step(DT)


def assert_at_pose(lamp, pose_name: str) -> None:
    pose = poses[pose_name]
    for joint, angle in lamp.get_joints(pose["joints"].keys()).items():
        assert angle == pytest.approx(pose["joints"][joint], abs=1e-6)
    assert lamp.get_light() == pytest.approx(pose["light"])


@pytest.mark.parametrize("ease", [ease_in_out, ease_in_out_back])
def test_easing_starts_at_zero_and_ends_at_one(ease):
    assert ease(0.0) == pytest.approx(0.0)
    assert ease(1.0) == pytest.approx(1.0)


def test_joints_are_clamped_to_model_limits(lamp):
    lamp.set_joints({Joints.NECK_YAW: 5.0, Joints.HEAD_PITCH: -5.0})
    assert lamp.get_joints([Joints.NECK_YAW])[Joints.NECK_YAW] == pytest.approx(1.35)
    assert lamp.get_joints([Joints.HEAD_PITCH])[Joints.HEAD_PITCH] == pytest.approx(
        -0.9
    )


def test_light_is_clipped_to_unit_range(lamp):
    lamp.set_light([2.0, -1.0, 0.5])
    assert lamp.get_light() == pytest.approx([1.0, 0.0, 0.5])


def test_gesture_reaches_its_pose_and_finishes(lamp):
    gesture = LampGesture(lamp, mood=Mood.NEUTRAL)
    gesture.play(Gesture.ENGAGE)
    assert gesture.active

    run(gesture, 1.0)

    assert not gesture.active
    assert_at_pose(lamp, "engage")


def test_sad_gestures_are_slower(lamp):
    gesture = LampGesture(lamp, mood=Mood.SAD)
    gesture.play(Gesture.ENGAGE)

    run(gesture, 1.0)
    assert gesture.active

    run(gesture, 0.7)
    assert not gesture.active


def test_multi_segment_gesture_plays_every_segment(lamp):
    gesture = LampGesture(lamp, mood=Mood.NEUTRAL)
    gesture.play(Gesture.NOD)
    segments = len(gestures[Gesture.NOD])

    run(gesture, segments - 0.5)
    assert gesture.active

    run(gesture, 0.6)
    assert not gesture.active
    assert_at_pose(lamp, gestures[Gesture.NOD][-1])


def test_play_target_moves_only_the_given_joints(lamp):
    base_before = lamp.get_joints([Joints.BASE_YAW])
    gesture = LampGesture(lamp, mood=Mood.NEUTRAL)
    gesture.play_target({Joints.NECK_YAW: 0.5}, [1.0, 0.0, 0.0], duration=0.5)

    run(gesture, 0.6)

    assert lamp.get_joints([Joints.NECK_YAW])[Joints.NECK_YAW] == pytest.approx(0.5)
    assert lamp.get_joints([Joints.BASE_YAW]) == base_before
    assert lamp.get_light() == pytest.approx([1.0, 0.0, 0.0])
