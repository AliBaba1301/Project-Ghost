from distutils.command.clean import clean
from djitellopy import Tello
import cv2
from objectDetection import *
import numpy as np
import time
from faceTracking import *

flight_mode = 1  # 0 to turn motors on 1 for testing
image_h = 360
image_w = 480
pid = [0.5, 0.5, 0]
p_error_w, p_error_h = 0, 0

with open('coco.names','rt') as f:
    class_names = f.read().split('\n')
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weights_path,config_path)
net.setInputSize(image_w, image_h)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


# Connecting to the Tello Drone
def connect_to_drone():
    drone = Tello()
    drone.connect()
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0
    print('connected')
    return(drone)

# Method to turn stream on or off


def stream_control(drone, command):
    if command == 'on':
        drone.streamon()
    else:
        drone.streamoff()


def main():
    global flight_mode, p_error_w, p_error_h, p_error_area
    drone = connect_to_drone()
    print(drone.get_battery())
    stream_control(drone, 'on')

    while True:

        # Fly
        if flight_mode == 0:
            drone.takeoff()
            flight_mode = 1

        frames = drone.get_frame_read()
        clean_img =  cv2.resize(frames.frame, (image_w, image_h))
        frame_od = cv2.resize(frames.frame, (image_w, image_h))
        
        frame = cv2.resize(frames.frame, (image_w, image_h))
        vid_stream, info = findFace(frame)

        classIds, confs, bbox = net.detect(frame_od,confThreshold = 0.6, nmsThreshold =0.2 ) #Only classifies objects once and if certainty is above 60%

        for i in range(len(classIds)):
            box = bbox[i]
            classId = classIds[i] -1
            conf = confs[i]

            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(frame_od, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
            cv2.putText(frame_od,f'{class_names[classId].upper()} ~ {round(conf * 100,2)}',(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            drone.send_rc_control(0,0,0,0)
        
        output_concat = np.concatenate((clean_img, vid_stream, frame_od), axis=1) 

        cv2.imshow('window', output_concat)

        # p_error_w, p_error_h = trackFace(
        #     drone, info, image_w, image_h, pid, p_error_w, p_error_h)
        


        # Testing to see if drone can perform basic commands when assessing for damage.
        # if flight_mode == 0:
        #     drone.takeoff()
        #     drone.rotate_counter_clockwise(90)
        #     drone.rotate_clockwise(90)
        #     drone.move_left(35)
        #     drone.move_right(35)
        #     drone.move_up(35)
        #     drone.move_down(35)
        #     drone.move_forward(35)
        #     drone.move_back(35)

        print(drone.get_battery())

        if cv2.waitKey(1) and 0xFF == ord('a'):
            drone.land()
            stream_control(drone, 'off')


if __name__ == '__main__':
    main()
