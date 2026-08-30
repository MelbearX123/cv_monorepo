import mujoco
import cv2
import numpy as np
from PIL import Image
from body.lamp_body import LampBody
from body.lamp_gesture import LampGesture
from config import LAMP_MODEL_PATH, ROOT

WIDTH, HEIGHT = 480, 360
FPS = 20
DT = 1.0 / FPS
BASE_DURATION = 0.6
CHAIN = ["disengage", "engage", "shake", "nod", "disengage"]
MOODS = ["neutral", "happy", "sad"]

out_path = ROOT / "docs" / "lamp_moods.gif"

lamp = LampBody(LAMP_MODEL_PATH)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.05, 0.0, 0.35]
cam.distance = 1.8
cam.azimuth = 130
cam.elevation = -20

frames: list[Image.Image] = []
with mujoco.Renderer(lamp.model, height=HEIGHT, width=WIDTH) as renderer:
    for mood in MOODS:
        gesture = LampGesture(lamp, mood=mood)
        for name in CHAIN:
            gesture.play(name, duration=BASE_DURATION)
            while gesture.active:
                gesture.step(DT)
                renderer.update_scene(lamp.data, camera=cam)
                rgb = renderer.render().copy()
                cv2.putText(
                    rgb,
                    mood,
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                frames.append(Image.fromarray(rgb))

frames[0].save(
    out_path,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / FPS),
    loop=0,
    optimize=True,
)
print(f"wrote {out_path}  ({len(frames)} frames)")
