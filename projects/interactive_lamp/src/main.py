import math
import time

import cv2
import mujoco

from camera import Camera
from face_tracker import FaceTracker
from face_reader import FaceReader
from character_state import CharacterState
from speech import SpeechManager, SpeechState
from lamp_body import LampBody
from lamp_gesture import LampGesture
from poses import Gesture, poses
from perception import read_face
from viewer import make_view, draw
from config import (
    LAMP_MODEL_PATH,
    RENDER_WIDTH,
    RENDER_HEIGHT,
    THINK_COLOR,
    THINK_PULSE_HZ,
)


def main() -> None:
    tracker = FaceTracker()
    reader = FaceReader()
    state = CharacterState()
    speech = SpeechManager()

    lamp = LampBody(LAMP_MODEL_PATH)
    lamp.set_joints(poses[Gesture.DISENGAGE]["joints"])
    lamp.set_light(poses[Gesture.DISENGAGE]["light"])
    gesture = LampGesture(lamp, mood=state.mood)
    view = make_view()

    prev_engaged = False
    prev_speech_state = SpeechState.IDLE

    with Camera() as cam, mujoco.Renderer(
        lamp.model, height=RENDER_HEIGHT, width=RENDER_WIDTH
    ) as renderer:
        prev = time.perf_counter()
        while True:
            now = time.perf_counter()
            dt = now - prev
            prev = now

            reading = read_face(cam, tracker, reader, dt)
            if reading is not None:
                state.is_engaged, state.mood = reading
                if state.is_engaged and not prev_engaged:
                    gesture = LampGesture(lamp, mood=state.mood)
                    gesture.play(Gesture.ENGAGE)
                    speech.listen()
                elif not state.is_engaged and prev_engaged:
                    gesture = LampGesture(lamp, mood=state.mood)
                    gesture.play(Gesture.DISENGAGE)
                prev_engaged = state.is_engaged

            gesture.step(dt)
            speech.step()

            if speech.state == SpeechState.TRANSCRIBING:
                pulse = 0.5 + 0.5 * math.sin(2 * math.pi * THINK_PULSE_HZ * now)
                lamp.set_light([c * (0.35 + 0.65 * pulse) for c in THINK_COLOR])

            if state.is_engaged:
                heard = speech.take_transcript()
                if heard:
                    speech.say(heard)  # echo to test for now
                    pass
                if (
                    prev_speech_state == SpeechState.SPEAKING
                    and speech.state == SpeechState.IDLE
                ):
                    speech.listen()
                prev_speech_state = speech.state

            draw(renderer, lamp, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
