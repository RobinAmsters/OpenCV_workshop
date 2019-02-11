import os
import cv2
import numpy as np
from PIL import Image


recognizer = cv2.face.LBPHFaceRecognizer_create()
path = '/home/peter/Documents/Python/FaceRecegnition/Standard_face_recognition/DataSet'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L'); #is PIL image
        faceNp=np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print (ID)
        IDs.append(ID)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces

    
IDs, faces = getImagesWithID(path)
recognizer.train(faces, IDs)
recognizer.write('recognizer/trainingData.xml')
cv2.destroyAllWindows()
