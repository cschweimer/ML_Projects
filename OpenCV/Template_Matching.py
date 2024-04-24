import cv2
import numpy as np

img_rgb = cv2.imread('opencv-template-matching-python-tutorial.jpg')
img_rgb = cv2.imread('Table.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('opencv-template-for-matching.jpg',0)
template = cv2.imread('Ball.jpg',0)
# get width and height of the template
w, h = template.shape[::-1]

print('Width {}, Height {}'.format(w, h))

# Matches
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

# Draw a yellow rectangle at all locations where the matching is higher than the threshold
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0]+w, pt[1]+h), (0,255,255), 1)
    
cv2.imshow('detected', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
