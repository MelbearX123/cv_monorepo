import numpy as np

# Create a (480, 640, 3) uint8 array of zeros — a black image. Set a 100×100 region in the
# centre to red (255,0,0 in RGB — remember OpenCV is BGR). Display it.
arr = np.zeros(dtype=np.uint8, shape=(480, 640, 3))
center_x = 640 // 2
center_y = 480 // 2
arr[center_y - 50 : center_y + 50, center_x - 50 : center_x + 50, 2] = 255

# Create a float32 array, multiply by 255, convert to uint8 with .astype(np.uint8).
arr1 = np.random.rand(dtype=np.float32).astype(np.uint8)
arr2 = arr1 * 255

# Now do it in reverse. Notice how values out of [0,255] get clipped or wrap.
arr1 = np.random.rand(480, 640, 3).astype(np.float32)
arr2 = (arr1 * 255).astype(np.uint8)

# Given img.shape = (720, 1280, 3): write the slice to crop the top-left quarter, the centre 100×100 region,
# and the bottom half.
crop = np.zeros(shape=(720, 1280, 3))
quarter_x = 1280 // 4
quarter_y = 720 // 4
# Note: y-coords 0 is top!
top_left = crop[:quarter_y, :quarter_x, :]

center_x = 1280 // 2
center_y = 720 // 2

center_100 = crop[center_y - 50 : center_y + 50, center_x - 50 : center_x + 50, :]

bottom_half = crop[720 // 2 :, :, :]


# np.zeros, np.ones, np.full, np.eye — create each, print shape and dtype.
zeros = np.zeros(shape=(3, 2, 3))
ones = np.ones(shape=(3, 2, 3))
full = np.full(shape=(3, 2, 3), fill_value=67)
eye = np.eye(N=3)

print([zeros.shape, zeros.dtype])
print([ones.shape, ones.dtype])
print([full.shape, full.dtype])
print([eye.shape, eye.dtype])

# Flatten a (480,640,3) array to 1D, then reshape back. Understand what .flatten() vs
# .ravel() vs .reshape(-1) each do.
# .flatten() creates a new copy of the array
# .ravel() creates a reference to parent array (changes also affect parent) when possible, saves memory
# .reshape(-1) same as ravel, but more general can be used with other param
toFlatten = np.zeros(shape=(480, 640, 3))
flatted = toFlatten.ravel()
restored = flatted.reshape(480, 640, 3)

# np.mean(img, axis=0), axis=1, axis=2 — for each, predict the shape before running it, then check.

# axis = 0 should mean the average of all rows
# axis = 1should be the average of all columns
# axis = 2 should mean average of all colour channels
img = np.zeros(shape=(480, 640, 3))
print(np.mean(img, axis=0).shape)  # (640, 3) collapses rows, shape loses 480
print(np.mean(img, axis=1).shape)  # (480, 3) collapses columns, shape loses 640
print(np.mean(img, axis=2).shape)  # (480, 640) collapses channels, shape loses 3
