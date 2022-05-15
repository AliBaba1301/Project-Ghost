import imghdr
import numpy as np
from djitellopy import Tello
import cv2

# Method for drawing a box around a given face


def findFace(img):
    face_cascade = cv2.CascadeClassifier(
        'haarscascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.2, 4)

    faceCentreList = []
    faceAreaList = []

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 115, 126), 2)
        cv2.circle(img, (x + w//2, y + h//2), 2, (255, 115, 126), 0)

        # Ensures that the closest face is the one being followed
        cx = x + w/2
        cy = y + h/2
        face_width = w
        area = w*h
        faceAreaList.append(area)
        faceCentreList.append([cx, cy])

        if len(faceAreaList) != 0:
            i = faceAreaList.index(max(faceAreaList))
            return img, [faceCentreList[i], faceAreaList[i], face_width]

    return img, [[0, 0], 0, 0]


def trackFace(drone, info, w, h, pid, p_error_w, p_error_h):

    distance = ''
    # PID Controller - Used for smoothing drone transitions
    optimal_w = w//2
    error_w = info[0][0] - optimal_w
    speed_w = pid[0] * error_w + pid[1] * (error_w - p_error_w)
    # keeps the speed withins range of drone
    speed_w = int(np.clip(speed_w, -100, 100))

    optimal_h = h//2
    error_h = info[0][1] - optimal_h
    speed_h = pid[0] * error_h + pid[1] * (error_h - p_error_h)
    # keeps the speed withins range of drone
    speed_h = int(np.clip(speed_h, -100, 100))

    speed_z = 0
    proximity = int(info[2] / 2.6)
    target_prox = 50
    if proximity > target_prox:
        speed_z = -15
    if proximity < target_prox:
        speed_z = 15

    if info[0][0] != 0 and info[0][1] != 0:
        drone.yaw_velocity = speed_w
        drone.up_down_velocity = (speed_h*-1)
        drone.for_back_velocity = speed_z
        print(proximity)
    else:
        drone.for_back_velocity = 0
        drone.left_right_velocity = 0
        drone.up_down_velocity = 0
        drone.yaw_velocity = 0
        drone.speed = 0
        error = 0

    if drone.send_rc_control:
        drone.send_rc_control(drone.left_right_velocity, drone.for_back_velocity,
                              drone.up_down_velocity, drone.yaw_velocity)

    # stop the drone moving for/back forever
    if drone.for_back_velocity > 0 or drone.for_back_velocity < 0:
        drone.for_back_velocity = 0
        drone.left_right_velocity = 0
        drone.up_down_velocity = 0
        drone.yaw_velocity = 0
        drone.send_rc_control(drone.left_right_velocity, drone.for_back_velocity,
                              drone.up_down_velocity, drone.yaw_velocity)

    return error_w, error_h
