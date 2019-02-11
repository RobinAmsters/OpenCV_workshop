#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:40:49 2019

@author: peter
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''
    #copy the input and convert to float
    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)    
    
    #Initialise with the initial pose estimation
    Transf_mat = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    
    src = cv2.transform(src, Transf_mat[0:2])
    
    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint  
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        #Compute the transformation between the current source
        #and destination cloudpoint
        transformation = cv2.estimateAffinePartial2D(src, dst[0, indices.T], False)
        print(transformation)

ang = np.linspace(-np.pi/2, np.pi/2, 320)
a = np.array([ang, np.sin(ang)])
th = np.pi/2
rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
b = np.dot(rot, a) + np.array([[0.2], [0.3]])

#run the ICP
M2 = icp(a, b, [0.1,  0.33, np.pi/2.2], 30)


plt.figure()
plt.plot(b[0],b[1])
#plt.plot(res[0].T[0], res[0].T[1], 'r.')
plt.plot(a[0], a[1])
plt.show()