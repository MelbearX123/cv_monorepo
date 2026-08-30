from perception.camera import Camera
from perception.face_tracker import FaceTracker
from perception.face_reader import FaceReader
from body.poses import Mood


def read_face(
    cam: Camera, tracker: FaceTracker, reader: FaceReader, dt: float
) -> tuple[bool, Mood] | None:
    frame = cam.get_frame()
    if frame is None:
        return None
    result = tracker.process(frame)
    return reader.read(result, dt)
