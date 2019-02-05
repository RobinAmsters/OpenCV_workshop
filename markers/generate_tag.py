"""
Generate and save aruco markers

@author: Robin Amsters
@email: robin.amsters@kuleuven.be

Original file: http://www.philipzucker.com/aruco-in-opencv/
"""
import cv2
import cv2.aruco as aruco

imageSize = 700 # Image size in pixels
n_markers = 3 # Number of markers to generate


for i in range(n_markers):
    markerID = i

    # Get dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    
    # Generate and save marker
    img = aruco.drawMarker(aruco_dict, markerID, imageSize) # dictionary, id, image size
    cv2.imwrite("marker_" + str(markerID) + '_' +  str(imageSize) +"_pixels.png", img) # Save image
    
    # Show marker untill key is pressed
    cv2.imshow('Marker ' + str(markerID) ,img)
    cv2.waitKey(750)
    cv2.destroyAllWindows()