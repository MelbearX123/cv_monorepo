from pathlib import Path

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# --- Lamp ---
LIGHT_NAME = "lamp_light"  # name of the controllable light in lamp.xml
LAMP_MODEL_PATH = Path(__file__).resolve().parent / "lamp.xml"
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

# --- Render view (orbit camera framing the lamp) ---
VIEW_LOOKAT = [0.05, 0.0, 0.35]
VIEW_DISTANCE = 2.3
VIEW_AZIMUTH = 20
VIEW_ELEVATION = -20

# --- MediaPipe Face Landmarker ---
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
MAX_NUM_FACES = 1
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- Face reader: engagement (radians / seconds) ---
ENGAGE_YAW = 0.45
DISENGAGE_YAW = 0.6
ENGAGE_DWELL = 0.5
DISENGAGE_DWELL = 1.0

# --- Face reader: emotion (blendshape score, 0..1) ---
SMILE_THRESHOLD = 0.4
SAD_THRESHOLD = 0.5
