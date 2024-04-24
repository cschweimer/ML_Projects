import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('Ball.jpg',0)
img2 = cv2.imread('Table.jpg',0)

# Similartiy Detector
orb = cv2.ORB_create()

# Keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# Find keypoints and descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Find matches and sort them based on distance
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()
