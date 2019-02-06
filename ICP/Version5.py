#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 09:07:37 2019

@author: peter
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors



def icp(ref_cloud, new_cloud, init_pose=(0,0,0), no_iterations=13):
        '''
        The Iterative Closest Point estimator.
        Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
        their relative pose and the number of iterations
        Returns the affine transform that transforms
        the cloudpoint a to the cloudpoint b.
        Note:
        '''
        ref_cloud = np.array([ref_cloud.T], copy=True).astype(np.float32)
        new_cloud = np.array([new_cloud.T], copy=True).astype(np.float32)  
    
        #Initialise with the initial pose estimation
        Transf_mat = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])
    
        ref_cloud_t = cv2.transform(ref_cloud, Transf_mat[0:2])
 
        for i in range(no_iterations):
            #Find the nearest neighbours between the current source and the
            #destination cloudpoint  
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(new_cloud[0])
            distances, indices = nbrs.kneighbors(ref_cloud_t[0])
            
            transformation = cv2.estimateAffinePartial2D(ref_cloud_t, new_cloud[0,indices.T])

            #Transform the previous source and update the
            #current source cloudpoint
            ref_cloud_t = cv2.transform(new_cloud, transformation[0])
       
            #Save the transformation from the actual source cloudpoint
            #to the destination
            Transf_mat = np.dot(Transf_mat, np.vstack((transformation[0],[0,0,1])))

            
        return Transf_mat[0:2]        


ang = np.linspace(-np.pi/2, np.pi/2, 320)
a = np.array([ang, np.sin(ang)])
th = np.pi/2
rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
b = np.dot(rot, a) + np.array([[0.2], [0.3]])


#run ICP
M2 = icp(a, b, [0.1,  0.33, np.pi/2.2])

print(M2)

#Plot the result
src = np.array([a.T]).astype(np.float32)
res = cv2.transform(src, M2)
plt.figure()
plt.plot(b[0],b[1],'b')
plt.plot(a[0],a[1],'g')
plt.show()
plt.figure()
plt.plot(res[0].T[0], res[0].T[1], 'r')
plt.plot(b[0], b[1],'b')
plt.show()