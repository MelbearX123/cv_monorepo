from cv2.typing import MatLike
import numpy as np

# This function will take multiple imgs and merge them into one


def mergeLayers(layers: list[MatLike]) -> MatLike | None:
    if not layers:
        print("No layers were selected")
        return
    canvas = np.zeros_like(layers[0])

    for layer in layers:
        painted = layer[:, :, 3] > 0
        canvas[painted] = layer[painted]

    return canvas
