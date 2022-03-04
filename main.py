# import pandas as pd
# import numpy as np
# import keras
# import tensorflow as tf
from djitellopy import Tello
import cv2

image_h = 240
image_w = 320
flight_mode = 1 #0 to turn motors on 1 for testing

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
    print(drone.get_battery)
    stream_control(drone,'on')




if __name__ == '__main__':
    main()