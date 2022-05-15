from distutils.command.clean import clean
from djitellopy import Tello
import cv2
from objectDetection import *
import numpy as np
import time
from faceTracking import *

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


def damage_assessment(drone):
    #  Testing to see if drone can perform basic commands when assessing for damage.
    drone.takeoff()
    drone.rotate_counter_clockwise(90)
    drone.rotate_clockwise(90)
    drone.move_left(35)
    drone.move_right(35)
    drone.move_up(35)
    drone.move_down(35)
    drone.move_forward(35)
    drone.move_back(35)


def low_battery(drone):
    stream_control(drone, 'off')
    drone.move_up(50)
    drone.flip_back()
    drone.land()


def main():
    flight_mode = 1  # 0 to turn motors on 1 for testing
    damage = 0  # 1 to run damage assessment
    image_h = 360
    image_w = 480
    pid = [0.5, 0.5, 0]
    p_error_w, p_error_h = 0, 0
    # Getting List of image types the user would like to save if they appear in the drone feed
    wanted_images = []
    with open('coco.names', 'rt') as f:
        class_names = f.read().split('\n')
    class_names_sorted = sorted(class_names)
    # changing input colours so user can read clearly
    wants_images = (input(
        '\033[1;34;47m Would you like to save the images of a certain type? (Y/N) ')).upper()
    if wants_images == 'Y':
        # Showing all available images types
        print('The available images class names are:')
        for i in range(len(class_names_sorted)):
            if i == 0:
                output = ''
            elif i % 13 == 0:
                print(output)
                output = ''
            output += ' : ' + class_names_sorted[i].capitalize()
            if i == len(class_names_sorted) - 1:
                print(output)
        # taking input of user requested types
        while wants_images == 'Y':
            tag = input('Name a type of image to save: ')
            if tag.lower() in class_names:
                wanted_images.append(tag.upper())
            else:
                print('Class not available!')
            wants_images = (
                input('Would you like to save more image classes? (Y/N) ')).upper()
    print(f'You have selected: {wanted_images}')

    # reset the terminal colours
    print('\033[0;37;40m')

    drone = connect_to_drone()
    print(drone.get_battery())
    stream_control(drone, 'on')
    # predictor, cfg = detr2_get_predictor()

    while True and damage == 0:
        # Fly
        if flight_mode == 0:
            drone.takeoff()
            flight_mode = 1

        frames = drone.get_frame_read()
        clean_img = cv2.resize(frames.frame, (image_w, image_h))
        frame_yolo = cv2.resize(frames.frame, (image_w, image_h))
        # frame_detr = cv2.resize(frames.frame, (image_w, image_h))
        image_yolo = yolo_detection(frame_yolo, wanted_images)
        # image_detr = detectron2_detection(
        # frame_detr, wanted_images, predictor, cfg)
        frame = cv2.resize(frames.frame, (image_w, image_h))
        vid_stream, info = findFace(frame)

        output_concat = np.concatenate(
            (clean_img, vid_stream, image_yolo), axis=1)

        cv2.imshow('window', output_concat)

        p_error_w, p_error_h = trackFace(
            drone, info, image_w, image_h, pid, p_error_w, p_error_h)

        if drone.get_battery() <= 10:
            low_battery(drone)

        print(drone.get_battery())

        if cv2.waitKey(1) and 0xFF == ord('a'):
            drone.land()
            stream_control(drone, 'off')

    # To be run after crash, This is in order to validate drone still has full functionality
    if damage == 1:
        damage_assessment(drone)


if __name__ == '__main__':
    main()
