import math
import time

import cv2
import mujoco

from perception.camera import Camera
from perception.face_tracker import FaceTracker
from perception.face_reader import FaceReader
from perception.perception import read_face
from perception.vision_brain import VisionBrain
from action.vla import VLAManager
from dialogue.character_state import CharacterState
from dialogue.response import ResponseManager
from speech.speech import SpeechManager, SpeechState
from body.lamp_body import LampBody
from body.lamp_gesture import LampGesture
from body.poses import Gesture, poses
from body.viewer import make_view, draw
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
    response = ResponseManager()
    vision = VisionBrain()

    lamp = LampBody(LAMP_MODEL_PATH)
    lamp.set_joints(poses[Gesture.DISENGAGE]["joints"])
    lamp.set_light(poses[Gesture.DISENGAGE]["light"])
    gesture = LampGesture(lamp, mood=state.mood)
    vla = VLAManager(lamp, vision, state.mood)
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

            # Perception
            frame = cam.get_frame()
            reading = read_face(frame, tracker, reader, dt)
            if reading is not None:
                state.is_engaged, state.mood = reading
                if state.is_engaged and not prev_engaged:
                    gesture = LampGesture(lamp, mood=state.mood)
                    gesture.play(Gesture.ENGAGE)
                    greeting = response.opening(state.mood)
                    if greeting:
                        speech.say(greeting)
                elif not state.is_engaged and prev_engaged:
                    gesture = LampGesture(lamp, mood=state.mood)
                    gesture.play(Gesture.DISENGAGE)
                prev_engaged = state.is_engaged

            # Motion
            if vla.active:
                vla.step(dt, frame)
            else:
                gesture.step(dt)
            speech.step()

            # Speak
            result = vla.take_result()
            if result is not None:
                speech.say(response.point_report(result))

            # Light
            if speech.state == SpeechState.TRANSCRIBING:
                pulse = 0.5 + 0.5 * math.sin(2 * math.pi * THINK_PULSE_HZ * now)
                lamp.set_light([c * (0.35 + 0.65 * pulse) for c in THINK_COLOR])
            elif speech.state == SpeechState.SPEAKING:
                lamp.set_light(poses[Gesture.ENGAGE]["light"])

            heard = speech.take_transcript()
            if heard:
                print(f"[heard] {heard!r}")
                if response.wants_point(heard):
                    if frame is not None and not vla.active:
                        vla.point_at_label(heard, frame)
                elif response.wants_recall(heard):
                    mem = vision.seen(heard)
                    if mem is not None and not vla.active:
                        vla.point_at_box(mem.box)
                    speech.say(response.recall(mem))
                elif response.wants_look(heard):
                    det = vision.main_object(frame) if frame is not None else None
                    if det:
                        vision.remember(det)
                    speech.say(response.look(det.label if det else None))
                else:
                    speech.say(response.respond(heard, state.mood))
            elif (
                state.is_engaged and speech.state == SpeechState.IDLE and not vla.active
            ):
                speech.listen()
                print("[listening...]")

            draw(renderer, lamp, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
