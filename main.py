from djitellopy import Tello
import cv2
import numpy
import time


image_h = 600
image_w = 800
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
        frame = frames.frame
        vid_stream = cv2.resize(frame,(image_w,image_h))
          

        if flight_mode == 0:
            drone.takeoff()
            drone.rotate_counter_clockwise(90)
            drone.rotate_clockwise(90)
            drone.move_left(35)
            drone.move_right(35)
        
        cv2.imshow('window',vid_stream)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stream_control(drone,'off')
            drone.land()
            break


   



if __name__ == '__main__':
    main()