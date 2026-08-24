import mujoco
import numpy as np
from pathlib import Path


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
            self._data.qpos[self._model.joint(name).qposadr[0]] = angle
        mujoco.mj_forward(m=self._model, d=self._data)

    def set_light(self, rgb_values: list[float]) -> None:
        light_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_LIGHT, "lamp_light"
        )
        R, G, B = np.clip(a=rgb_values, a_min=0, a_max=1.0)
        self._model.light_diffuse[light_id] = [R, G, B]
