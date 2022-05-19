import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
import cv2
import time
import numpy as np


image_h = 360
image_w = 480
with open('coco.names', 'rt') as f:
    class_names = f.read().split('\n')

######## SSD SETUP ########
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(image_w, image_h)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def ssdv3_detection(img, wanted_images):
    # Only classifies objects once and if certainty is above 60%

    classIds, confs, bbox = net.detect(
        img, confThreshold=0.6, nmsThreshold=0.1)

    save = False
    for i in classIds:
        if class_names[i-1].upper() in wanted_images:
            save = True

    if save:
        filename = 'saved_images/ssdv3' + str(time.time()) + '.jpg'
        cv2.imwrite(filename, img)

    for i in range(len(classIds)):
        box = bbox[i]
        classId = classIds[i] - 1
        conf = confs[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, h+y), (255, 115, 126), thickness=1)
        cv2.putText(img, f'{class_names[classId].upper()} ~ {round(conf * 100,2)}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)

    return img


################################
######## DETECTRON SETUP ########
detr_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def detr2_get_predictor():
    model_path = "detectron2/FCNNR50FPN/faster_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_path)
    cfg.MODEL.DEVICE = "cpu"
    # set threshold for this model same as for ssdv3 detection
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.WEIGHTS = 'detectron2/FCNNR50FPN/model_final_280758.pkl'
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def detectron2_detection(img, wanted_images, predictor, cfg):
    outputs = predictor(img)

    objects_found = (list(outputs["instances"].pred_classes))

    save = False
    for i in range(len((objects_found))):
        if detr_class_names[objects_found[i].item()].upper() in wanted_images:
            save = True

    if save:
        filename = 'saved_images/detr2' + str(time.time()) + '.jpg'
        cv2.imwrite(filename, img)
        
    v = Visualizer(img[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = out.get_image()[:, :, ::-1]

    return img
################################
