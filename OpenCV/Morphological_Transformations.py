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

    # Erosion and Dilation
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    
    # Opening and Closing
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow('frame', frame)
    cv2.imshow('res', res)
    #cv2.imshow('erosion', erosion)
    #cv2.imshow('dilation', dilation)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()
