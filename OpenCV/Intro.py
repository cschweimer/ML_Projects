import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale for simplification
img = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE)

### Show image with cv2
cv2.imshow('Image', img)
# press any key to close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
# save the image
cv2.imwrite('watch1.png', img)

### Show image with matplotlib
#plt.imshow(img, cmap='gray', interpolation='bicubic')
#plt.show()
