import time
import mujoco
import cv2
from pathlib import Path
from lamp_body import LampBody
from lamp_gesture import LampGesture

model_path = Path(__file__).parent / "lamp.xml"
lamp = LampBody(model_path)
gesture = LampGesture(lamp, mood="neutral")

# External orbit camera
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.05, 0.0, 0.35]
cam.distance = 1.8
cam.azimuth = 130
cam.elevation = -20

# Play a multi-keyframe gesture through the non-blocking player.
gesture.play("shake", duration=0.5)

with mujoco.Renderer(lamp.model, height=480, width=640) as renderer:
    prev = time.perf_counter()
    while gesture.active:
        now = time.perf_counter()
        dt = now - prev
        prev = now

        gesture.step(dt)  # advance the gesture by real elapsed time

        renderer.update_scene(lamp.data, camera=cam)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.imshow("lamp", frame)
        if cv2.waitKey(16) == 27:  # Esc quits early
            break

    cv2.waitKey(0)  # hold the final frame until a key is pressed
cv2.destroyAllWindows()
