import torch
import matplotlib as plt
from matplotlib import pyplot
import numpy as np
import cv2

# Setup: create a virtual environment, install opencv-python numpy matplotlib torch torchvision. 
# Verify torch.cuda.is_available() returns True.

print(torch.cuda.is_available())
print(torch.__version__)

# Write a show_images(images, titles, cols=3) utility function using matplotlib subplots — 
# you'll import this in every project.
def show_images(images, titles, cols=3):
  pyplot.subplot

# Deliberately create 3 common bugs: wrong dtype (float64 > 1.0 to imshow), wrong channel order (BGR to imshow), 
# wrong shape (2D vs 3D). Debug each by printing shape/dtype/min/max.

# Load any image, deliberately corrupt it (wrong dtype, divide by wrong value) and practice restoring it using
#  only .astype(), np.clip(), and cv2.cvtColor().

# Write a debug_img(img, label='') function that prints shape, dtype, min, max, and shows the image —
# use this throughout the course.

# Write requirements.txt for your environment. Practice: delete the environment, recreate it 
# from requirements.txt in under 2 minutes.