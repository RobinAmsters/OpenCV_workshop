#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:26:53 2017

File to detect the position and orientation of a charuco board in a video file.

@author: Robin Amsters
@email: robin.amsters@kuleuven.be

Original files: 
    - http://www.philipzucker.com/aruco-in-opencv/
    - https://www.element14.com/community/thread/58118/l/need-python-opencv-code-help?displayFullThread=true
"""


import cv2
import cv2.aruco as aruco
import numpy as np
import pickle

import matplotlib.pyplot as plt
import file_select_gui as gui
from mpl_toolkits.mplot3d import Axes3D


cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text
markerSize = 0.033  # Size of markers in physical world [m]
# cam_params_file = gui.get_file_path("Select camera parameters file").name
cam_params_file = '../example_data/cam_params.pckl'
cal = pickle.load(open( cam_params_file, "rb" ))

cMat = cal[0]
# dist = cal[2]
dist = cal[1][0]

# 4x4 board
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(3,3,.07,.035,dictionary)

pose_0_set = False
pose_0 = ()
 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Conver to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    parameters =  aruco.DetectorParameters_create()

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if (ids is not None and len(ids) >= 4):
        corners, ids, rejectedCorners, recoveredIDs = aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=rejectedImgPoints, cameraMatrix=cMat, distCoeffs=dist, parameters=parameters) # refine corners, does not seeem to have any effect ?

        retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board, cameraMatrix=cMat, distCoeffs=dist)

        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix=cMat, distCoeffs=dist)
        rvec = np.array([rvec.item(0), rvec.item(1), rvec.item(2)])
        tvec = np.array([tvec.item(0), tvec.item(1), tvec.item(2)])

        # show information on image
        frameWithMarkers = aruco.drawDetectedMarkers(frame, corners) # Draw marker borders
        cv2.putText(frameWithMarkers, "ID: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)  # Show detected IDs
        aruco.drawAxis(frameWithMarkers, cMat, dist, rvec, tvec, 0.1) #Draw Axis

    else:  
        # Display: no IDs
        cv2.putText(frame, "No IDs", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        frameWithMarkers = frame
    

    # Display the resulting frame
    cv2.imshow('frame',frameWithMarkers)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Stop when q is pressed
        break
    
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()