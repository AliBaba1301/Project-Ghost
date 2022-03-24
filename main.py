from gettext import find
from djitellopy import Tello
import cv2
import numpy as np
import time
from faceTracking import *

image_h = 360
image_w = 480
flight_mode = 0 #0 to turn motors on 1 for testing

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
def stream_control(drone,command):
    if command == 'on':
        drone.streamon()
    else:
        drone.streamoff()

def main():
    drone =  connect_to_drone()
    print(drone.get_battery())
    stream_control(drone,'on')

    while True:
        frames = drone.get_frame_read()
        frame = cv2.resize(frames.frame,image_w,image_h)
        vid_stream = findFace(frame)

        cv2.imshow('window',vid_stream)  


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
        
       

        if cv2.waitKey(1) and 0xFF == ord('q'):
            drone.land()
            stream_control(drone,'off')
    
            break


   



if __name__ == '__main__':
    main()