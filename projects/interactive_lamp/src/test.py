"""Throwaway test: read raw face signals off the webcam.

Run to (1) verify which yaw formula tracks left/right head turn, and
(2) see real smile/frown blendshape values for tuning FaceReader.
Press 'q' to quit.
"""

import math

import cv2

from camera import Camera
from face_tracker import FaceTracker


def main() -> None:
    tracker = FaceTracker()
    with Camera() as cam:
        while True:
            rgb = cam.get_frame()
            if rgb is None:
                continue

            result = tracker.process(rgb)
            if result is None:
                lines = ["no face"]
            else:
                matrix = result.facial_transformation_matrixes[0]
                yaw = math.atan2(matrix[0][2], matrix[2][2])
                scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
                smile = max(
                    scores.get("mouthSmileLeft", 0.0),
                    scores.get("mouthSmileRight", 0.0),
                )
                frown = max(
                    scores.get("mouthFrownLeft", 0.0),
                    scores.get("mouthFrownRight", 0.0),
                )
                brow = scores.get(
                    "browInnerUp", 0.0
                )  # inner-brow raise = reliable "sad" signal
                lines = [
                    f"yaw:   {yaw:+.2f}   (turn head L/R -> should swing)",
                    f"smile: {smile:.2f}",
                    f"frown: {frown:.2f}   (weak - ignore)",
                    f"brow:  {brow:.2f}   (make a sad/worried face)",
                ]

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # back to BGR for imshow
            for i, text in enumerate(lines):
                cv2.putText(
                    bgr,
                    text,
                    (16, 36 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("face test (q to quit)", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
