import numpy as np
import cv2
import time

# Brightness adjustment: load an image as float32, add 50 to all channels using broadcasting, 
# clip to [0,255], convert back to uint8. Compare with original.
og = cv2.imread('strawberry.jpg')
img = og.astype(dtype=np.float32)
modified = img + 50
modified = modified.clip(min=0, max=255).astype(np.uint8)

# cv2.imshow('Original', og)
# cv2.imshow('Modified', modified)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Channel scaling: multiply the R channel by 1.5, leave G and B unchanged — use a (3,) 
# broadcast array. Watch what happens without clipping.
scaling = np.array([1, 1, 1.5])
multiplied_R = og * scaling
overflow = multiplied_R.astype(np.uint8)
clipped = multiplied_R.clip(min=0, max=255).astype(np.uint8)

# cv2.imshow('Original', og)
# cv2.imshow('Overflowed', overflow)
# cv2.imshow('Clipped', clipped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Create a red mask: pixels where the R channel > 150 AND G < 100 AND B < 100. 
# Set those pixels to (0,255,0). Use boolean indexing.
# Need to isolate each channel, and then put together
blue = og[:,:,0]
green = og[:,:,1]
red = og[:,:,2]

red_mask = (red > 150) & (green < 100) & (blue < 100)
masked = og.copy()
masked[red_mask] = (0, 255, 0)

# cv2.imshow('Original', og)
# cv2.imshow('Red mask', masked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# np.where: given two images of the same size, use np.where to combine them — 
# take pixels from image A where a mask is True, from image B elsewhere.
some_mask = (red > 100) & (green > 125) & (blue < 167)
some_mask = some_mask[:, :, np.newaxis]
combination = np.where(some_mask, og, overflow)

# cv2.imshow('Original', og)
# cv2.imshow('Overflowed', overflow)
# cv2.imshow('Combo', combination)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Stack 4 versions of an image side by side using np.hstack — build the display grid pattern you'll 
# use all of week 1.
i1 = og.copy()
i2 = modified.copy()
i3 = masked.copy()
i4 = cv2.GaussianBlur(og, (15,15), 0)

stacked = np.hstack((i1, i2, i3, i4))

# cv2.imshow('Stacked', stacked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Vectorisation benchmark: convert an image to grayscale using (1) a nested for-loop over pixels, 
# (2) numpy dot product. Time both. Record the speedup.

# gray = 0.114*B + 0.587*G + 0.299*R
h, w = og.shape[:2]
gray = np.zeros(shape=(h,w), dtype=np.uint8)
loop_start = time.time()
for y in range(h):
  for x in range(w):
    b, g, r = og[y, x]
    gray[y,x] = 0.114*b + 0.587*g + 0.299*r
loop_time = time.time() - loop_start

prod_start = time.time()
weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
gray_dot = (og.copy().astype(np.float32) @ weights).astype(np.uint8)
prod_time = time.time() - prod_start

# cv2.imshow(f'Loop: {loop_time:.4f}', gray)
# cv2.imshow(f'Dot: {prod_time:.4f}', gray_dot)
# cv2.waitKey(0)
# cv2.destroyAllWindows()