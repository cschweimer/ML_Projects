import numpy as np
import cv2

# first webcam in the system
cap = cv2.VideoCapture(0)

# Ouput the file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    
    # display two videos
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    
    # Press q to stop video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()

