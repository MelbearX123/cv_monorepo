import mujoco
import cv2
from pathlib import Path
from lamp_body import LampBody

model_path = Path(__file__).parent / "lamp.xml"
lamp = LampBody(model_path)

lamp.set_joints(
    {
        "shoulder_pitch_joint": 0.35,
        "elbow_pitch_joint": -0.90,
        "head_pitch_joint": -0.30,
    }
)
lamp.set_light([1.0, 0.90, 0.70])

with mujoco.Renderer(lamp.model, height=480, width=640) as renderer:
    renderer.update_scene(lamp.data)
    image = renderer.render()

bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imshow("lamp", bgr_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
