import numpy as np
import cv2

# cv2.filter2D(img, -1, kernel) with a custom numpy array kernel.
custom = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=np.float32) / 18
img = cv2.imread("./projects/cv_learning/strawberry.jpg")
cImg = cv2.filter2D(img, -1, custom)
# cv2.imshow('custom kernel', cImg)

# cv2.GaussianBlur(img, (ksize,ksize), sigmaX) — note ksize must be odd.
gblurred = cv2.GaussianBlur(img, (9, 9), 0)  # let it decide a sigma
# cv2.imshow('gaussian blur', gblurred)

# cv2.blur() (box), cv2.medianBlur() (good for salt-and-pepper noise).
noisy_img = img.copy()
num_noise_pixels = 1000

# Add White "Salt" specks
for _ in range(num_noise_pixels // 2):
    y = np.random.randint(0, noisy_img.shape[0])
    x = np.random.randint(0, noisy_img.shape[1])
    noisy_img[y, x] = 255

# Add Black "Pepper" specks
for _ in range(num_noise_pixels // 2):
    y = np.random.randint(0, noisy_img.shape[0])
    x = np.random.randint(0, noisy_img.shape[1])
    noisy_img[y, x] = 0
box_blurred = cv2.blur(noisy_img, (5, 5))
median_blurred = cv2.medianBlur(noisy_img, 5)
comparison = np.hstack((box_blurred, median_blurred))
# cv2.imshow('Box Blur | Median Blur', comparison)
# cv2.imshow('Noisy', noisy_img)

# cv2.bilateralFilter() — edge-preserving smoothing; understand why it's slow.
# Bilateral fitler slow because not 2D separable, must do N^2 operations
bFilter = cv2.bilateralFilter(noisy_img, 9, 100, 100)
gFilter = cv2.GaussianBlur(noisy_img, (9, 9), 0)

# Visualise side-by-side: Gaussian vs bilateral on a noisy image.
# cv2.imshow('Bilateral', bFilter)
# cv2.imshow('Gaussian', gFilter)

# Create a sharpening kernel manually and apply it.
sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpened_img = cv2.filter2D(img, -1, sharpening)
# cv2.imshow('sharpened', sharpened_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
