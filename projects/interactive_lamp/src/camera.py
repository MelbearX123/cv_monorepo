import cv2

from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


class Camera:
    """Owns the webcam. The single source of frames for face detection and the VLM."""

    def __init__(
        self,
        device_index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
    ) -> None:
        self._capture = cv2.VideoCapture(device_index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            raise RuntimeError("Camera not available")

    def get_frame(self):
        """Return the most recent frame as an RGB image, or None if the read failed."""
        success, frame = self._capture.read()
        if not success:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame

    def release(self) -> None:
        """Free the camera device. Call once on shutdown, or the camera stays locked."""
        self._capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
