from pathlib import Path

# Project root (one level above src/); data assets live here, not in src/.
ROOT = Path(__file__).resolve().parent.parent

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# --- Lamp ---
LIGHT_NAME = "lamp_light"
LAMP_MODEL_PATH = ROOT / "assets" / "lamp.xml"
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

# --- Render view (orbit camera framing the lamp) ---
VIEW_LOOKAT = [0.05, 0.0, 0.35]
VIEW_DISTANCE = 2.3
VIEW_AZIMUTH = 20
VIEW_ELEVATION = -20

# --- MediaPipe Face Landmarker ---
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
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

# --- Speech: audio I/O (Hz / channels) ---
STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
VAD_FRAME = 512

# --- Speech: models ---
WHISPER_MODEL = "base"
WHISPER_COMPUTE = "int8"
WHISPER_DEVICE = "cpu"
KOKORO_LANG = "a"  # 'a' = American English
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 0.85

# --- Speech: turn-taking / VAD (probability + seconds) ---
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE = 0.7
VAD_SPEECH_PAD = 0.2
MAX_RECORD_SECONDS = 10.0
THINK_COLOR = [0.4, 0.7, 1.0]  # light blue
THINK_PULSE_HZ = 1.2

# --- Vision: MediaPipe object detector (local, free, real-time) ---
OBJECT_MODEL_PATH = ROOT / "models" / "efficientdet_lite0.tflite"
OBJECT_MAX_RESULTS = 5
OBJECT_SCORE_THRESHOLD = 0.4
OBJECT_MEMORY_SIZE = 5

# --- Response: check-in pacing (seconds) ---
MOOD_CHECK_COOLDOWN = 300.0
