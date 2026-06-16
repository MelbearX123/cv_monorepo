import cv2
import numpy as np
from matplotlib import pyplot as plt

# cv2.imread(), cv2.imshow(), cv2.waitKey() — load and display an image.
img = cv2.imread('./cv_learning/strawberry.jpg')

# cv2.imshow('Strawberry!', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Check image.shape, image.dtype — understand what you have.
# print(img.shape)
# print(img.dtype)

# cv2.cvtColor(img, cv2.COLOR_BGR2RGB/GRAY/HSV) — convert between spaces.
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# cv2.imshow('rgb', rgb)
# cv2.imshow('grey', grey)
# cv2.imshow('hsv', hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.split() and cv2.merge() — separate and recombine channels.
b, g, r = cv2.split(img)
# cv2.imshow('b', b)
# cv2.imshow('g', g)
# cv2.imshow('r', r)

merged = cv2.merge([b, g, r])
# cv2.imshow('merged',  merged)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.calcHist() and matplotlib to visualise histograms per channel.
bhist = cv2.calcHist([img], [0], None, [256], [0,256])
plt.plot(bhist, color='b')

ghist = cv2.calcHist([img], [1], None, [256], [0,256])
plt.plot(ghist, color='g')

rhist = cv2.calcHist([img], [2], None, [256], [0,256])
plt.plot(rhist, color='r')
plt.title('Colour channel histogram')
plt.show()

# cv2.imwrite() — save your processed images.
cv2.imwrite('output.jpg', g) #Dont want clutter, image deleted