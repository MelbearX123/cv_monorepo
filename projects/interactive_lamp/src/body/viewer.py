import cv2
import mujoco

from body.lamp_body import LampBody
from config import (
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


def draw(renderer: mujoco.Renderer, lamp: LampBody, view: mujoco.MjvCamera) -> None:
    renderer.update_scene(lamp.data, camera=view)
    bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    cv2.imshow("lamp (q to quit)", bgr)


def draw_detections(frame_rgb, detections) -> None:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        x, y, w, h = d.box
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            bgr, f"{d.label} {d.score:.2f}", (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    cv2.imshow("camera (debug)", bgr)
