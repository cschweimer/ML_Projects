# Filter out one color from an image or a video
# Example: Green Screen Operation

import cv2
import numpy as np

# Get the webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # HSV: Hue, Saturation, Value; other way to define colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # finding the right values is a bit trial and error
    lower_green = np.array([40,50,50])
    upper_green = np.array([150,255,250])
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    # Is the pixel in the range or not? 
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # bitwise operation
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Define a kernel
    kernel = np.ones((15,15), np.float32)/225
    smoothed = cv2.filter2D(res, -1, kernel)
    
    # Gaussian blur
    blur = cv2.GaussianBlur(res, (15,15), 0)
    
    # Median blur
    median = cv2.medianBlur(res, 15)
    
    # Bilateral blur
    bilateral = cv2.bilateralFilter(res, 15, 75, 75)
    
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    #cv2.imshow('smoothed', smoothed)
    cv2.imshow('blur', blur)
    cv2.imshow('median', median)
    cv2.imshow('bilateral', bilateral)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()
