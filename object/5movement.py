# Import Libraries
import dlib
import glob
import numpy as np
import cv2
import os
import sys
import time

import matplotlib.pyplot as plt
import pyautogui as pyg
import shutil

# Set these thresholds accordingly.
 
# If hand size is larger than this then up, button is triggered
size_up_th = 80000
 
# If hand size is smaller than this then down key is triggered
size_down_th = 25000
 
# If the center_x location is less than this then left key is triggered
left = 320
 
# If the center_x location is greater than this then right key is triggered
right = 320
 
# Load our trained detector 
detector = dlib.simple_object_detector('test/Right_Head_Detector.svm')
 
# Set the window to normal
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
 
# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
 
# Setting the downscaling size, for faster detection
# If you're not getting any detections then you can set this to 1
scale_factor = 2.0
 
# Initially the size of the hand and its center x point will be 0
size, center_x = 0,0
 
# Initialize these variables for calculating FPS
fps = 0
frame_counter = 0
start_time = time.time()
 
# Set the while loop
while(True):
     
    # Read frame by frame
    ret, frame = cap.read()
     
    if not ret:
        break
     
    # Laterally flip the frame
    frame = cv2.flip( frame, 1 )
     
    # Calculate the Average FPS
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))
     
    # Create a clean copy of the frame
    copy = frame.copy()  
     
    # Downsize the frame.
    new_width = int(frame.shape[1]/scale_factor)
    new_height = int(frame.shape[0]/scale_factor)
    resized_frame = cv2.resize(copy, (new_width, new_height))
     
    # Detect with detector
    detections = detector(resized_frame)
     
    # Set Default values
    text = 'No Hand Detected'
    center_x = 0
    size = 0
 
    # Loop for each detection.
    for detection in (detections):    
         
        # Since we downscaled the image we will need to resacle the coordinates according to the original image.
        x1 = int(detection.left() * scale_factor )
        y1 =  int(detection.top() * scale_factor )
        x2 =  int(detection.right() * scale_factor )
        y2 =  int(detection.bottom()* scale_factor )
         
        # Calculate size of the hand. 
        size = int( (x2 - x1) * (y2-y1) )
         
        # Extract the center of the hand on x-axis.
        center_x = int(x1 + (x2 - x1) / 2)
         
        # Draw the bounding box of the detected hand
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0), 2 )
         
        # Now based on the size or center_x location set the required text
        if center_x > right:
            text = 'Right'
 
        elif center_x < left:
            text = 'Left'
 
        elif size > size_up_th:
            text = 'Up'
 
        elif size < size_down_th:
            text = 'Down'
             
        else:
            text = 'Neutral'
             
    # Now we should draw lines for left/right threshold
    cv2.line(frame, (left,0),(left, frame.shape[0]),(25,25,255), 2)
    cv2.line(frame, (right,0),(right, frame.shape[0]),(25,25,255), 2)    
 
    # Display Center_x value and size.
    cv2.putText(frame, 'Center: {}'.format(center_x), (500, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (233, 100, 25), 1)
    cv2.putText(frame, 'size: {}'.format(size), (500, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (233, 100, 25))
 
    # Finally display the text showing which key should be triggered
    cv2.putText(frame, text, (220, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (33, 100, 185), 2)
 
    # Display the image
    cv2.imshow('frame',frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Relase the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()