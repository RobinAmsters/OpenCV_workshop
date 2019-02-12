#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:22:41 2019

@author: peter
"""

import cv2
import numpy as np

filename = 'chessboard.png'
#filename = 'simple.jpg'

img = cv2.imread(filename)
img_1 = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('chessboard.png',img_1)
cv2.imshow('Harris corner detection',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()