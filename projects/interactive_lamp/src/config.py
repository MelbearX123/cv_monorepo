from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

MAX_NUM_FACES = 1
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
