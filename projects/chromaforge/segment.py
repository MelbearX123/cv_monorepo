import cv2
from cv2.typing import MatLike
import numpy as np


# Splits the picture into k different segments and colours each segment its most occuring colour
def colourSegment(img: MatLike, k: int) -> MatLike:
    # Use a bilateral blur to slightly blur while preserving edges
    bilateral_blur = cv2.bilateralFilter(src=img, d=9, sigmaColor=20, sigmaSpace=150)

    # Convert to LAB to match human perception
    img_LAB = cv2.cvtColor(bilateral_blur, cv2.COLOR_BGR2LAB)

    # Use K-means to split the image into K different colours
    k_means_img = (img_LAB.astype(np.float32)).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(
        data=k_means_img,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    # Convert centers back to 8-bit colors
    centers = np.uint8(centers)

    # Map the labels to the center colors to reconstruct the flattened image
    quantized_flattened = centers[labels.flatten()]

    # Reshape back into the original 3D image dimensions
    reshaped_img = quantized_flattened.reshape(img.shape)
    final_img = cv2.cvtColor(reshaped_img, cv2.COLOR_LAB2BGR)
    return final_img
