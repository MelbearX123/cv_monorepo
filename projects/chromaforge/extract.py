import cv2
from cv2.typing import MatLike
import numpy as np

#This function will take an image that has already had its colours segmented
#and extract each layer onto a transparent png. It will return all layers

def colourExtract(img: MatLike, labels: MatLike) -> list[MatLike]:
  layers = []
  # Split the image so we can merge it with an alpha channel
  b, g, r = cv2.split(img)

  #Loop through each colour and create an alpha mask for each
  for i in np.unique(labels):
    mask = (labels == i)
    alpha = mask.astype(np.uint8) * 255
    layer = cv2.merge([b, g, r, alpha])
    layers.append(layer)
  return layers