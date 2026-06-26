import cv2
from cv2.typing import MatLike
import numpy as np


# Mean-shift filtering merges pixels that are close in both position and
# colour into spatially-coherent blobs. K-means then caps the result to k
# colours, merging similar blobs together. Because k-means runs on every
# pixel (not just the unique colours), frequently-occurring colours pull
# the cluster centres toward them automatically - frequency weighting
# falls out for free, no need to weight anything by hand.
def colourSegment(img: MatLike, sp: int, sr: int, k: int) -> tuple[MatLike, MatLike]:
    # sp = spatial window radius, sr = colour window radius
    bilateral_blurred = cv2.bilateralFilter(src=img, d=9, sigmaColor=150, sigmaSpace=150)
    mean_shifted = cv2.pyrMeanShiftFiltering(bilateral_blurred, sp, sr)

    # Convert to LAB to match human perception
    img_LAB = cv2.cvtColor(mean_shifted, cv2.COLOR_BGR2LAB)

    # Use k-means to cap the result to k colours
    k_means_img = img_LAB.astype(np.float32).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        data=k_means_img,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    centers = np.uint8(centers)
    quantized_flattened = centers[labels.flatten()]

    reshaped_img = quantized_flattened.reshape(img.shape)
    final_img = cv2.cvtColor(reshaped_img, cv2.COLOR_LAB2BGR)

    reshaped_labels = labels.reshape(img.shape[:2])
    return final_img, reshaped_labels