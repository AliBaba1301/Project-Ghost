from djitellopy import Tello
import cv2
import numpy as np
import time
from faceTracking import *

flight_mode = 0  # 0 to turn motors on 1 for testing
image_h = 360
image_w = 480
pid = [0.5, 0.5, 0]
p_error_w, p_error_h, p_error_area = 0, 0, 0


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
        frame = cv2.resize(frames.frame, (image_w, image_h))
        vid_stream, info = findFace(frame)
        output_concat = np.concatenate((frame, vid_stream), axis=1) 

        cv2.imshow('window', output_concat)

        p_error_w, p_error_h , p_error_area= trackFace(
            drone, info, image_w, image_h, pid, p_error_w, p_error_h, p_error_area)
        


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
