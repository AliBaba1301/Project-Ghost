import cv2
from objectDetection import *
import os
from main import *


# TEST connection 
drone = connect_to_drone()
stream_control(drone, 'on')
stream_control(drone, 'off')

# Test objectDetection as well as ability to save wanted_images
wanted_images = ['APPLE','FORK']
predictor, cfg = detr2_get_predictor()
output_dir = 'test_data/saved_images/'


test_dir = os.path.join(os.getcwd(), 'test_data')
for file in os.listdir(test_dir):
    if file.endswith('.jpg'):
        
        img = cv2.imread(os.path.join(test_dir, file))
        
        d2_img = detectron2_detection(img,wanted_images,predictor, cfg)
        cv2.imwrite(output_dir+'detr2_'+file, d2_img)

        ssdv3_img = ssdv3_detection(img,wanted_images)
        cv2.imwrite(output_dir+'ssdv3_'+file, ssdv3_img)