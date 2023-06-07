import numpy as np
import cv2
# import model wrapper class
from openvino.model_zoo.model_api.models import SSD, YOLO
# import inference adapter and helper for runtime setup
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core


# read input image using opencv
input_data = cv2.imread("6th-street.jpg")

# define the path to mobilenet-ssd model in IR format
model_path = "/home/jk/open_model_zoo/demos/object_detection_demo/python/public/yolo-v4-tf/FP16/yolo-v4-tf.xml"

# create adapter for OpenVINO™ runtime, pass the model path
model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")

config = "/home/jk/open_model_zoo/demos/object_detection_demo/python/public/yolo-v4-tf/keras-YOLOv3-model-set/cfg/yolov4.cfg"
# create model API wrapper for SSD architecture
# preload=True loads the model on CPU inside the adapter
ssd_model = YOLO(model_adapter, config, preload=True)

# apply input preprocessing, sync inference, model output postprocessing
results = ssd_model(input_data)


for det in results[0]:
    print(det.xmin)
    pts = []
    pts.append([det.xmin, det.ymin])
    pts.append([det.xmin, det.ymax])
    pts.append([det.xmax, det.ymax])
    pts.append([det.xmax, det.ymin])
    pts = np.array(pts, np.int32)
    is_closed = True
    color = (0,255,0)
    thickness = 2
    input_data = cv2.polylines(input_data, [pts], is_closed, color, thickness)
cv2.imshow("dets", input_data)
cv2.waitKey(0)