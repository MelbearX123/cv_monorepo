import cv2
import numpy as np

#Given an image, blur it and make it colour blocky
img = cv2.imread('./projects/chromaforge/strawberry.jpg')

bBlurred = cv2.bilateralFilter(src=img, ksize=(11,11), sigmaX=5)
threshold1 = 50
threshold2 = 150
greyed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(greyed, threshold1, threshold2, 3)

cv2.imshow('test', greyed)
cv2.imshow('test2', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Bilateral filter (blur that preserves edges — critical, not Gaussian)
# K-means in LAB color space (not RGB — LAB matches human color perception much better)
# Morphological cleanup (remove small islands, smooth jagged borders)
# Optionally: edge-aware smoothing to clean up the region boundaries