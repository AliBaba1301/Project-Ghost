import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
import cv2
import time

image_h = 360
image_w = 480
with open('coco.names', 'rt') as f:
    class_names = f.read().split('\n')

######## YOLO SETUP ########
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(image_w, image_h)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def yolo_detection(img, wanted_images):
    # Only classifies objects once and if certainty is above 60%

    classIds, confs, bbox = net.detect(
        img, confThreshold=0.6, nmsThreshold=0.2)

    save = False
    for i in classIds:
        if class_names[i].upper() in wanted_images:
            save = True

    if save:
        filename = 'saved_images/yolov3' + str(time.time()) + '.jpg'
        cv2.imwrite(filename, img)

    for i in range(len(classIds)):
        box = bbox[i]
        classId = classIds[i] - 1
        conf = confs[i]

        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, h+y), (255, 115, 126), thickness=2)
        cv2.putText(img, f'{class_names[classId].upper()} ~ {round(conf * 100,2)}', (box[0]+10, box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (126, 115, 255), 1)

    return img


################################
######## DETECTRON SETUP ########
# # Setup detectron2 logger
setup_logger()

# import some common detectron2 utilities


def detr2_get_predictor():
    model_path = "detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_path)
    cfg.MODEL.DEVICE = "cpu"
    # set threshold for this model same as for yolov3 detection
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = 'detectron2/model_final_68b088.pkl'
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def detectron2_detection(img, wanted_images, predictor, cfg):

    current_time = time.gmtime()[5]
    old_time = 0

    outputs = predictor(img)

    objects_found = (list(outputs["instances"].pred_classes))

    save = False
    for i in range(len((objects_found))):
        if class_names[int(objects_found[i].item())].upper() in wanted_images:
            save = True

    if save and (current_time - old_time) >= 15:
        filename = 'saved_images/detr2' + str(time.time()) + '.jpg'
        cv2.imwrite(filename, img)
        current_time = time.gmtime()[5]

    v = Visualizer(img[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = out.get_image()[:, :, ::-1]

    return img
################################
