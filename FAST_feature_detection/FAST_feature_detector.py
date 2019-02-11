#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:32:33 2019

@author: peter
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

color = (0,255,0)

img2 = cv.imread('simple.jpg')
img3 = img2
img = cv.imread('simple.jpg',0)
# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)

px_ref = np.array([x.pt for x in kp], dtype=np.float32)

mask =np.zeros_like(img)

for elem in px_ref:
    mask = cv.circle(img2,(elem[0],elem[1]),1,color,-1)

img2 = cv.add(img2,mask)
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv.imshow('fast_true.png',img2)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()





# =============================================================================

# =============================================================================
# Disable nonmaxSuppression
# =============================================================================
# =============================================================================
# # fast.setNonmaxSuppression(0)
# # kp = fast.detect(img,None)
# # print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
# # px_ref = np.array([x.pt for x in kp], dtype=np.float32)
# # 
# # mask =np.zeros_like(img)
# # 
# # for elem in px_ref:
# #     mask2 = cv.circle(img3,(elem[0],elem[1]),1,color,-1)
# # img3 = cv.add(img2,mask2)
# # cv.imwrite('fast_false.png',img3)
# # if cv.waitKey(0) & 0xff == 27:
# #     cv.destroyAllWindows()
# =============================================================================
# 
# =============================================================================
