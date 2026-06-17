import numpy as np
import cv2

# Numpy matrix ops on images: add, subtract, multiply images element-wise.
img1 = cv2.imread('./projects/cv_learning/strawberry.jpg')
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

added = img1 + img2
subbed = img1 - img2
mult = img1 * 2

# cv2.imshow('added', added)
# cv2.imshow('subbed', subbed)
# cv2.imshow('mult', mult)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.getRotationMatrix2D(center, angle, scale) — build an affine matrix.
width = img1.shape[1]
height = img1.shape[0]
center = (width / 2, height / 2)
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)

# cv2.warpAffine(img, M, (w,h)) — apply it.
rotated_image = cv2.warpAffine(img1, matrix, (width, height))
# cv2.imshow('modified', rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.resize() with INTER_LINEAR vs INTER_NEAREST — when interpolation matters.
#INTER_NEAREST, finds colour of closest pixel and copies it, blocky but extremely fast
#INTER_LINEAR, takes WA of closest 2x2 pixels, blended but slightly slower
nearest = cv2.resize(src=img1, dsize=(1000, 1000), interpolation=cv2.INTER_NEAREST)
linear = cv2.resize(src=img1, dsize=(1000, 1000), interpolation=cv2.INTER_LINEAR)

# cv2.imshow('nearest', nearest)
# cv2.imshow('linear', linear)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.flip(), cv2.transpose() — simple geometric transforms.
#flip -> hflip: 1, vflip: 0, both: -1
#transpose -> finds transpose of img (H, W, C) to (W, H , C) -> rotates the image 90 degrees CC and mirrors it
#  horizontally.
flipWay = cv2.flip(img1, -1)
transposeWay = cv2.transpose(img1)

# cv2.imshow('flipWay', flipWay)
# cv2.imshow('transposeWay', transposeWay)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Build a function that rotates an image around an arbitrary point.
def rotateArbitratyPoint(src: Matlike, point: tuple, degrees: float):
  height, width = src.shape[:2]
  matrix = cv2.getRotationMatrix2D(point, degrees, 1.0)
  rotated_image = cv2.warpAffine(src, matrix, (width, height))
  return rotated_image
