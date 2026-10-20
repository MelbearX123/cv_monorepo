import logging
import time

import cv2
import mujoco

from audio.audio import AudioManager
from body.lamp_body import LampBody
from body.viewer import DetectionOverlay, draw, make_view
from character import Character
from config import DEBUG_VIEW, LAMP_MODEL_PATH, RENDER_HEIGHT, RENDER_WIDTH
from perception.camera import Camera
from perception.perception import FacePerception
from perception.vision_brain import VisionBrain
from speech.speech import SpeechManager


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    lamp = LampBody(LAMP_MODEL_PATH)
    vision = VisionBrain()
    face = FacePerception()
    speech = SpeechManager()
    view = make_view()
    overlay = DetectionOverlay(vision)

    with Camera() as cam, AudioManager() as audio, mujoco.Renderer(
        lamp.model, height=RENDER_HEIGHT, width=RENDER_WIDTH
    ) as renderer:
        character = Character(lamp, face, vision, speech, audio)
        prev = time.perf_counter()
        while True:
            now = time.perf_counter()
            dt, prev = now - prev, now

            frame = cam.get_frame()
            character.tick(dt, frame)

            draw(renderer, lamp, view)
            if DEBUG_VIEW and frame is not None:
                overlay.show(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
