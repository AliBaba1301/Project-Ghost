import imghdr
from djitellopy import tello
import cv2

def findFace(img):
    face_cascade = cv2.CascadeClassifier('haarscascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,1.2,4)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,115,126), 2)
    return img  