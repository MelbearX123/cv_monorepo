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
RENDER_WIDTH = 960
RENDER_HEIGHT = 720

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
OBJECT_MODEL_PATH = ROOT / "models" / "efficientdet_lite2.tflite"
OBJECT_MAX_RESULTS = 5
OBJECT_SCORE_THRESHOLD = 0.4
OBJECT_MEMORY_SIZE = 5
LOOK_CONFIRM_FRAMES = 10  # frames to sample after "look at this" before naming the object

# --- Debug: live camera + detection overlay window (testing only) ---
DEBUG_VIEW = True
DEBUG_DETECT_EVERY = 5  # run the overlay detector every N frames to keep the loop responsive

# --- Response: check-in pacing (seconds) ---
MOOD_CHECK_COOLDOWN = 300.0

# --- VLA: point-at-object (radians / normalized frame coords) ---
POINT_DURATION = 0.8
POINT_HOLD = 8.0 
POINT_NECK_YAW_LEVEL = 0.0
POINT_NECK_YAW_SPAN = 0.6
POINT_PITCH_LEVEL = -0.3
POINT_PITCH_SPAN = 0.3
POINT_NECK_YAW_SIGN = 1.0
POINT_CENTER_TOLERANCE = 0.15
POINT_MAX_CORRECTIONS = 1

# --- Audio: character SFX + music ---
AUDIO_SAMPLE_RATE = 44100
AUDIO_DIR = ROOT / "assets" / "audio"
AUDIO_POWER_ON = AUDIO_DIR / "power_on.mp3"
AUDIO_POWER_OFF = AUDIO_DIR / "power_off.mp3"
AUDIO_MUSIC = AUDIO_DIR / "music.mp3"
AUDIO_MUSIC_DUCK = 0.25
AUDIO_DUCK_STEP = 0.05
AUDIO_POWER_OFF_GAIN = 0.4  #lower volume
