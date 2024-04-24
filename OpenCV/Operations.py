import cv2
import numpy as np

# Read colored image
img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)

# Shape of the image
print(img.shape)

# Pick a pixel
px = img[55,55]

# Alter the pixel
img[55,55] = [255,255,255]
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ROI: Region of image
roi = img[100:150, 100:150]
img[100:150, 100:150] = [255,255,255]
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Re-read the image
img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)

# Copy and Paste a portion
watch_face = img[37:111, 107:194]
img[0:74, 0:87] = watch_face
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
