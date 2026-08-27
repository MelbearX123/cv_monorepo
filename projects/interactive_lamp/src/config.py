from pathlib import Path

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# --- Lamp ---
LIGHT_NAME = "lamp_light"   # name of the controllable light in lamp.xml

# --- MediaPipe Face Landmarker ---
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
MAX_NUM_FACES = 1
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
