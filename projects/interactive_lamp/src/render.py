import mujoco
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from lamp_body import LampBody
from lamp_gesture import LampGesture

WIDTH, HEIGHT = 480, 360
FPS = 20
DT = 1.0 / FPS                 # fixed timestep -> smooth, deterministic playback
BASE_DURATION = 0.6            # seconds per segment, before mood scaling
CHAIN = ["disengage", "engage", "shake", "nod", "disengage"]
MOODS = ["neutral", "happy", "sad"]

model_path = Path(__file__).parent / "lamp.xml"
out_path = Path(__file__).parent / "lamp_moods.gif"

lamp = LampBody(model_path)

# External orbit camera framing the whole lamp + floor.
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.05, 0.0, 0.35]
cam.distance = 1.8
cam.azimuth = 130
cam.elevation = -20

frames: list[Image.Image] = []
with mujoco.Renderer(lamp.model, height=HEIGHT, width=WIDTH) as renderer:
    for mood in MOODS:
        gesture = LampGesture(lamp, mood=mood)          # mood is fixed per player
        for name in CHAIN:
            gesture.play(name, duration=BASE_DURATION)
            while gesture.active:
                gesture.step(DT)                        # advance one fixed tick
                renderer.update_scene(lamp.data, camera=cam)
                rgb = renderer.render().copy()          # RGB uint8
                cv2.putText(rgb, mood, (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                frames.append(Image.fromarray(rgb))

# Save as an animated GIF (plays inline anywhere; no codec/ffmpeg needed).
frames[0].save(
    out_path,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / FPS),   # ms per frame
    loop=0,
    optimize=True,
)
print(f"wrote {out_path}  ({len(frames)} frames)")
