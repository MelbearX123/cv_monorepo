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
