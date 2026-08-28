from collections.abc import Iterable

from poses import poses, gestures, Gesture
from easing import EASING, DURATION_SCALE, ease_in_out


class LampGesture:
    """Plays mood-modulated gestures on a LampBody, one non-blocking step at a time."""

    def __init__(self, lampBody, mood) -> None:
        self._lampBody = lampBody
        self._mood = mood
        self._ease = EASING.get(mood, ease_in_out)  # mood's timing curve
        self._duration_scale = DURATION_SCALE.get(mood, 1.0)  # mood's speed multiplier

        self._sequence: list[dict] = []  # target keyframe bundles, in order
        self._seg_index = 0  # current segment within the sequence
        self._start: dict | None = None  # start bundle of the current segment
        self._elapsed = 0.0  # seconds into the current segment
        self._duration = 1.0  # seconds per segment (after mood scaling)
        self._active = False

    @property
    def active(self) -> bool:
        """True while a gesture is still playing."""
        return self._active

    def play(self, gesture_name: Gesture, duration: float = 1.0) -> None:
        """Begin a gesture from the lamp's current state. Mood scales the duration."""
        self._sequence = [poses[name] for name in gestures[gesture_name]]
        self._seg_index = 0
        self._elapsed = 0.0
        self._duration = duration * self._duration_scale
        self._start = self._capture_bundle(self._sequence[0]["joints"].keys())
        self._active = True

    def step(self, dt: float) -> None:
        """Advance the current gesture by dt seconds and apply it to the body."""
        if not self._active:
            return
        self._elapsed += dt
        target = self._sequence[self._seg_index]
        self._apply(self.transition(self._start, target, self._elapsed, self._duration))
        if self._elapsed >= self._duration:
            self._advance_segment(target)

    def transition(
        self, start: dict, target: dict, elapsed: float, duration: float = 1.0
    ) -> dict:
        """Blend two keyframe bundles at the mood's easing, returning {joints, light}."""
        t = self._ease(min(elapsed / duration, 1.0))
        joints = {
            joint: self._lerp(start["joints"][joint], target["joints"][joint], t)
            for joint in target["joints"]
        }
        light = [self._lerp(s, e, t) for s, e in zip(start["light"], target["light"])]
        return {"joints": joints, "light": light}

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _capture_bundle(self, joint_names: Iterable[str]) -> dict:
        """Snapshot the lamp's current joints + light as a start keyframe bundle."""
        return {
            "joints": self._lampBody.get_joints(joint_names),
            "light": self._lampBody.get_light(),
        }

    def _apply(self, bundle: dict) -> None:
        """Push a blended keyframe bundle onto the body."""
        self._lampBody.set_joints(bundle["joints"])
        self._lampBody.set_light(bundle["light"])

    def _advance_segment(self, finished_target: dict) -> None:
        """Move to the next segment, or end the gesture if the sequence is done."""
        self._seg_index += 1
        if self._seg_index >= len(self._sequence):
            self._active = False
        else:
            self._start = finished_target  # arrived here; next segment starts from it
            self._elapsed = 0.0
