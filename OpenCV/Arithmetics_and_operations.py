# Put one image into another one
import cv2
import numpy as np

img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')

'''
# overlay the images
add = img1 + img2
add = cv2.add(img1,img2) #adds all pixel values -> lots of white
weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
cv2.imshow('add', weighted)
'''

img2 = cv2.imread('mainlogo.png')
cv2.imshow('logo', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# here bgr colors
rows,cols,channels = img2.shape
# put img2 in the top left corner of img1
roi = img1[0:rows, 0:cols]

# convert image 2 to grayscale
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray image', img2gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Thresholding
# make the logo black and white and reverse it (>=220 -> 255, else ->0)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('black and white image', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# inverse of the mask
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('black and white image reversed', mask_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# background of image 1 in the area of interest: python logo is black (pixel=0) 
# foreground of image 2, everything but the logo is black
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

cv2.imshow('background image 1', img1_bg)
cv2.imshow('foreground image 2', img2_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Replace the top left in the original image with the newly created part
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
