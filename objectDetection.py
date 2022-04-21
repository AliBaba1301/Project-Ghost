import cv2

image_h = 360
image_w = 480

with open('coco.names','rt') as f:
    class_names = f.read().split('\n')
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weights_path,config_path)
net.setInputSize(image_w, image_h)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

def yolo_detection(img,image_w, image_h,drone):
    classIds, confs, bbox = net.detect(img,confThreshold = 0.6, nmsThreshold =0.2 ) #Only classifies objects once and if certainty is above 60%
    
    for i in range(len(classIds)):
        box = bbox[i]
        classId = classIds[i]
        conf = confs[i]

        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(frame_od, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        cv2.putText(frame_od,f'{class_names[classId-1].upper()} ~ {round(conf * 100,2)}',(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


    drone.send_rc_control(0,0,0,0)

    return img