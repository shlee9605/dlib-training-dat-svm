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

# Load our trained detector 
# detector = dlib.simple_object_detector(file_name)
 
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
 
# load up all the detectors using this function.
hand_detector = dlib.fhog_object_detector("./test/Object_Detector.svm")
head_detector = dlib.fhog_object_detector("./test/Right_Head_Detector.svm") 
 
# Now insert all detectors in a list
detectors = [hand_detector, head_detector]
 
# Create a list of detector names in the same order
names = ['Hand Detected', 'Head Detected']
 
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
     
    # Perform the Detection
    # Beside's boxes you will also get confidence scores and ID's of the detector
    [detections, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, resized_frame, 
    upsample_num_times=1)
     
    # Loop for each detected box
    for i in range(len(detections)):    
         
        # Since we downscaled the image we will need to resacle the coordinates according to the original image.
        x1 = int(detections[i].left() * scale_factor )
        y1 =  int(detections[i].top() * scale_factor )
        x2 =  int(detections[i].right() * scale_factor )
        y2 =  int(detections[i].bottom()* scale_factor )
         
        # Draw the bounding box with confidence scores and the names of the detector
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0), 2 )
        cv2.putText(frame, '{}: {:.2f}%'.format(names[detector_idxs[i]], confidences[i]*100), (x1, y2+20), 
        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),2)
 
        # Calculate size of the hand. 
        size = int( (x2 - x1) * (y2-y1) )
         
        # Extract the center of the hand on x-axis.
        center_x = int(x1 + (x2 - x1) / 2)
     
    # Display FPS and size of hand
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),2)
 
    # This information is useful for when you'll be building hand gesture applications
    cv2.putText(frame, 'Center: {}'.format(center_x), (540, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.putText(frame, 'size: {}'.format(size), (540, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
     
    # Display the image
    cv2.imshow('frame',frame)
                   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Relase the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()