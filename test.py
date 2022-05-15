import cv2
from objectDetection import *

wanted_images = []


img = cv2.imread('test.png')
predictor, cfg = detr2_get_predictor()
d2_img = detectron2_detection(img,wanted_images,predictor, cfg)
cv2.imshow('test',d2_img)
cv2.waitKey(1000000)