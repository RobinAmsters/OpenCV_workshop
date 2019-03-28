import cv2
import numpy as np


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('/home/peter/Documents/Python/FaceRecegnition/Standard_face_recognition/recognizer/trainingData.xml')

id=0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    ret, img=cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w,y+h), (0,0,255),2)
        id,conf =rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id = 'Peter'
            cv2.putText(img, str(id) ,(x,y), font, 1,(0,255,0),2,cv2.LINE_AA)
        elif(id ==2):
            id = 'Bart'
            cv2.putText(img, str(id) ,(x,y), font, 1,(0,0,255),2,cv2.LINE_AA)
        elif(id ==3):
            id = 'Martijn'
            cv2.putText(img, str(id) ,(x,y), font, 1,(0,255,255),2,cv2.LINE_AA)
        elif(id ==4):
            id = 'Jeroen'
            cv2.putText(img, str(id) ,(x,y), font, 1,(255,255,255),2,cv2.LINE_AA)
        elif(id ==5):
            id = 'David'
            cv2.putText(img, str(id) ,(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("Face",img);
    if (cv2.waitKey(1) == ord ('q')):
        break;

cam.release()
cv2.destroyAllWindows()
