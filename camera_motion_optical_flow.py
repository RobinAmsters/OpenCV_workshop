#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 12:59:37 2018

@author: peter

Attempt to define camera movement continuously
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.01,
                       minDistance = 7
                       ,blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              30, 0.03))

color = (0,255,0)
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#print(p0)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame) 
#while True:
while True:
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                           **lk_params)
# =============================================================================
#     print(p1)
#     print(st)
#     print(err)
# =============================================================================
    # Select good points if st op point == 0, no optical flow was found
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # if there are less than 5 points left, stop the program.
    if len(good_new) < 1:
        break
    # draw the tracks and calculate if we are going left or right, up or down
    px_horizontal = []
    px_vertical = []
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color, 2)
        frame = cv2.circle(frame,(a,b),5,color,-1)
        
        px_horizontal.append(c-a)
        px_vertical.append(d-b)
        
    median_horizontal = np.median(px_horizontal)
    mean_horizontal = np.mean(px_horizontal)
    median_vertical = np.median(px_vertical)
    mean_vertical = np.mean(px_vertical)


    if mean_horizontal > 5:
        print('turn to the right')
    if mean_horizontal < -5:
        print('turn to the left')
    if mean_vertical > 5:
        print('turn down')  
    if mean_vertical < -5:
        print('turn up')  
    img = cv2.add(frame,mask)
    #adding a red dot in the middle of the frame. 
    frame = cv2.circle(frame,(640,0),5,(0,0,255),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
