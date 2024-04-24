import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Table.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgModel = np.zeros((1,65), np.float64)
fgModel = np.zeros((1,65), np.float64)

# rectangle of the foreground (determined manually)
# rect = (start_x, start_y, width, height)
rect = (25, 80, 1250, 425)

# Grab cut method
cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()
