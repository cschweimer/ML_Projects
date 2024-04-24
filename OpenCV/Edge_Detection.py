# Edge Detection and Gradients

import cv2
import numpy as np

# Get the webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Laplacian Gradient
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    
    # Sobel Gradient (with directional intensity in x- or y-direction)
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    
    # Canny Edge Detection: smaller numbers -> more noise
    edges = cv2.Canny(frame, 100, 100)
    edges_2 = cv2.Canny(frame, 10, 10)
    
    cv2.imshow('original', frame)
    #cv2.imshow('Laplacian', laplacian)
    #cv2.imshow('sobelx', sobelx)
    #cv2.imshow('sobely', sobely)
    cv2.imshow('Edges', edges)
    #cv2.imshow('Edges_2', edges_2)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()
