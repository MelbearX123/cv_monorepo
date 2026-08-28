import mujoco
import numpy as np
from collections.abc import Iterable
from pathlib import Path

from config import LIGHT_NAME


class LampBody:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._model = mujoco.MjModel.from_xml_path(str(self._model_path.resolve()))
        self._data = mujoco.MjData(self._model)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    def set_joints(self, targets: dict[str, float]) -> None:
        for name, angle in targets.items():
            joint = self._model.joint(name)
            low, high = joint.range
            if low < high:  # clamp to the joint's limits when it has them
                angle = min(max(angle, low), high)
            self._data.qpos[joint.qposadr[0]] = angle
        mujoco.mj_forward(m=self._model, d=self._data)

    def get_joints(self, names: Iterable[str]) -> dict[str, float]:
        return {
            name: float(self._data.qpos[self._model.joint(name).qposadr[0]])
            for name in names
        }

    def set_light(self, rgb_values: list[float]) -> None:
        light_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_LIGHT, LIGHT_NAME
        )
        R, G, B = np.clip(a=rgb_values, a_min=0, a_max=1.0)
        self._model.light_diffuse[light_id] = [R, G, B]

    def get_light(self) -> list[float]:
        light_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_LIGHT, LIGHT_NAME
        )
        return [float(c) for c in self._model.light_diffuse[light_id]]
