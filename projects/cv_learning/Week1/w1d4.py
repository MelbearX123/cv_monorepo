import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("./projects/cv_learning/strawberry.jpg")

# cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) for Gx; same with (0,1) for Gy.
Gx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
Gy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

# Compute gradient magnitude with np.sqrt(Gx**2 + Gy**2) and convert to uint8.
mag = np.sqrt(Gx**2, Gy**2)
mag_uint8 = np.uint8(mag)

# cv2.Scharr() — a more accurate derivative kernel than Sobel.
Gx2 = cv2.Scharr(src=img, ddepth=cv2.CV_64F, dx=1, dy=0)
Gy2 = cv2.Scharr(src=img, ddepth=cv2.CV_64F, dx=0, dy=1)

# cv2.Laplacian(img, cv2.CV_64F) after Gaussian blur.
g_blur = cv2.GaussianBlur(src=img, ksize=(3, 3), sigmaX=1)
laplacian = cv2.Laplacian(g_blur, cv2.CV_64F)

# cv2.imshow('laplacian', laplacian)

# Visualise Gx, Gy, magnitude, and direction as separate heatmaps.
direction = np.arctan2(Gy, Gx)

# plt.figure(figsize=(12, 10))

# plt.subplot(2, 2, 1)
# plt.imshow(Gx, cmap='coolwarm')
# plt.title('Horizontal Gradients (Gx)')
# plt.colorbar(label='Gradient Intensity')

# plt.subplot(2, 2, 2)
# plt.imshow(Gy, cmap='coolwarm')
# plt.title('Vertical Gradients (Gy)')
# plt.colorbar(label='Gradient Intensity')

# plt.subplot(2, 2, 3)
# plt.imshow(mag, cmap='inferno')
# plt.title('Gradient Magnitude (Edge Strength)')
# plt.colorbar(label='Edge Strength')

# plt.subplot(2, 2, 4)
# plt.imshow(direction, cmap='twilight')
# plt.title('Gradient Direction (Orientation Angle)')
# plt.colorbar(label='Angle in Radians')

# plt.tight_layout()
# plt.show()

# Compare Sobel and Scharr on the same image.
scharr_mag = np.sqrt(Gx2**2 + Gy2**2)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(mag, cmap="inferno")
axes[0].set_title("Sobel Magnitude")
axes[0].axis("off")

axes[1].imshow(scharr_mag, cmap="inferno")
axes[1].set_title("Scharr Magnitude (More Precise/Continuous)")
axes[1].axis("off")

plt.tight_layout()
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
