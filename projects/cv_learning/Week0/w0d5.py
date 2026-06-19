import torch
import numpy as np
import cv2
import math
import matplotlib
from matplotlib import pyplot as plt

# Setup: create a virtual environment, install opencv-python numpy matplotlib torch torchvision.
# Verify torch.cuda.is_available() returns True.
# print(torch.cuda.is_available())
# print(torch.__version__)


# Write a show_images(images, titles, cols=3) utility function using matplotlib subplots —
# you'll import this in every project.
def show_images(images, titles, cols=3):
    # Shows a list of images using subplots
    n_images = len(images)
    rows = math.ceil(n_images / cols)

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)  # i+1 because index is 1-based
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")  # hides the x/y axis ticks, cleaner look

    plt.tight_layout()  # stops subplots overlapping each other
    plt.show()


# Deliberately create 3 common bugs: wrong dtype (float64 > 1.0 to imshow), wrong channel order (BGR to imshow),
# wrong shape (2D vs 3D). Debug each by printing shape/dtype/min/max.
og = cv2.imread("./cv_learning/strawberry.jpg").astype(np.uint8)

bug_1 = og.astype(np.float64)
fix_1 = bug_1.astype(np.uint8)

bug_2 = og.copy()
fix_2 = cv2.cvtColor(bug_2, cv2.COLOR_BGR2RGB)

bug_3 = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(fix_1)
# axes[0].set_title("Bug 1: Blown Out (Float > 1)")
# axes[1].imshow(fix_2)
# axes[1].set_title("Bug 2: Inverted Colors (BGR)")
# axes[2].imshow(bug_3, cmap='gray')
# axes[2].set_title("Bug 3: Greyscale")

# plt.axis('off')
# plt.show()

# Load any image, deliberately corrupt it (wrong dtype, divide by wrong value) and practice restoring it using
#  only .astype(), np.clip(), and cv2.cvtColor().

corrupted = cv2.cvtColor(og.copy(), cv2.COLOR_BGR2RGB)  # Hidden swap
corrupted = corrupted.astype(np.float32) / 128.0  # Maximum value is now around 2.0

# Get it back to uint8 and BGR
restore = corrupted * 128.0
restore = restore.astype(np.uint8)

# plt.imshow(restore)
# plt.axis('off')
# plt.show()


# Write a debug_img(img, label='') function that prints shape, dtype, min, max, and shows the image —
# use this throughout the course.
def debug(img, label=""):
    print(f"Shape: {img.shape}")
    print(f"dtype: {img.dtype}")
    print(f"min: {img.min()}")
    print(f"max: {img.max()}")
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title(label)


# Write requirements.txt for your environment. Practice: delete the environment, recreate it
# from requirements.txt in under 2 minutes
# Find the versions
print(f"np: {np.__version__}")
print(f"cv2: {cv2.__version__}")
print(f"torch: {torch.__version__}")
print(f"plt: {matplotlib.__version__}")
