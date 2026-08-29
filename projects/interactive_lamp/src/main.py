import time

import cv2
import mujoco

from camera import Camera
from face_tracker import FaceTracker
from face_reader import FaceReader
from character_state import CharacterState
from lamp_body import LampBody
from lamp_gesture import LampGesture
from poses import Gesture, poses
from config import (
    LAMP_MODEL_PATH,
    RENDER_WIDTH,
    RENDER_HEIGHT,
    VIEW_LOOKAT,
    VIEW_DISTANCE,
    VIEW_AZIMUTH,
    VIEW_ELEVATION,
)


def make_view() -> mujoco.MjvCamera:
    view = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(view)
    view.lookat[:] = VIEW_LOOKAT
    view.distance = VIEW_DISTANCE
    view.azimuth = VIEW_AZIMUTH
    view.elevation = VIEW_ELEVATION
    return view


def main() -> None:
    tracker = FaceTracker()
    reader = FaceReader()
    state = CharacterState()
    lamp = LampBody(LAMP_MODEL_PATH)
    lamp.set_joints(poses[Gesture.DISENGAGE]["joints"])
    lamp.set_light(poses[Gesture.DISENGAGE]["light"])
    gesture = LampGesture(lamp, mood=state.mood)
    view = make_view()

    prev_engaged = False

    with Camera() as cam, mujoco.Renderer(
        lamp.model, height=RENDER_HEIGHT, width=RENDER_WIDTH
    ) as renderer:
        prev = time.perf_counter()
        while True:
            now = time.perf_counter()
            dt = now - prev
            prev = now

            frame = cam.get_frame()
            if frame is not None:
                result = tracker.process(frame)
                is_engaged, mood = reader.read(result, dt)
                state.is_engaged = is_engaged
                state.mood = mood

                if is_engaged and not prev_engaged:
                    gesture = LampGesture(lamp, mood=state.mood)
                    gesture.play(Gesture.ENGAGE)
                elif not is_engaged and prev_engaged:
                    gesture = LampGesture(lamp, mood=state.mood)
                    gesture.play(Gesture.DISENGAGE)
                prev_engaged = is_engaged

            gesture.step(dt)
            renderer.update_scene(lamp.data, camera=view)
            bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("lamp (q to quit)", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
