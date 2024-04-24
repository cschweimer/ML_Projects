import numpy as np
import cv2

img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)

# Draw a white line on the image from (0,0) to (150,150) with width 15
# opencv is BGR (blue, green, red)
cv2.line(img, (0,0), (150,150), (255,255,255), 15)

# Draw a green rectangle, top left and bottom right
cv2.rectangle(img, (15,25), (200,150), (0,255,0), 5)

# Draw a red circle, center, radius, width, or -1 is filling the circle
cv2.circle(img, (100,63), 55, (0,0,255), -1)

# Draw a yellow polygon, connect final point to first point
pts = np.array([[10,5], [20,30], [70,20], [50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3)

# Write in purple, with size 1 and thickness 2
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV Tuts!', (0,130), font, 1, (255,0,255), 2, cv2.LINE_AA)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
