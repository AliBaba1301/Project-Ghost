import cv2

image_h = 360
image_w = 480

with open('coco.names', 'rt') as f:
    class_names = f.read().split('\n')
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(image_w, image_h)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def yolo_detection(img):
    # Only classifies objects once and if certainty is above 60%
    classIds, confs, bbox = net.detect(
        img, confThreshold=0.6, nmsThreshold=0.2)

    for i in range(len(classIds)):
        box = bbox[i]
        classId = classIds[i] - 1
        conf = confs[i]

        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, h+y), (255, 115, 126), thickness=2)
        cv2.putText(img, f'{class_names[classId].upper()} ~ {round(conf * 100,2)}', (box[0]+10, box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (126, 115, 255), 1)

    return img
