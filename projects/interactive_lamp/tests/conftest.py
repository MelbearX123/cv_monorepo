import numpy as np
import pytest

from body.lamp_body import LampBody
from config import CAMERA_HEIGHT, CAMERA_WIDTH, LAMP_MODEL_PATH


@pytest.fixture
def lamp() -> LampBody:
    return LampBody(LAMP_MODEL_PATH)


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
