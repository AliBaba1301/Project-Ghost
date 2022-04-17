import imghdr
from djitellopy import tello
import cv2

# Method for drawing a box around a given face 
def findFace(img):
    face_cascade = cv2.CascadeClassifier('haarscascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,1.2,4)

    faceCentreList = []
    faceAreaList = []

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,115,126), 2)

        # Ensures that the closest face is the one being followed
        cx = x + w/2
        cy = y + h/2
        area = w*h
        faceAreaList.append(area)
        faceCentreList.append([cx,cy])

        if len(faceAreaList) != 0:
            i = faceAreaList.index(max(faceAreaList))
            return img , [faceCentreList[i],faceAreaList]
        else:
            return img,[[0,0],0] 