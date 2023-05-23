#!/usr/bin/env python3

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import rapidocr_openvinogpu as rog

VIS = True
TOPIC_NOVA_RAW_IMG = "/nova/rgb/image_raw"
TOPIC_VIS_IMG = "/nova/vis_image"

class NOVATextDetector(Node):

    def __init__(self):
        super().__init__('nova_text_detector')
        self.subscription = self.create_subscription(
            Image,
            TOPIC_NOVA_RAW_IMG,
            self.listener_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.vis_publisher_ = self.create_publisher(Image, TOPIC_VIS_IMG, 10)
        self.rapid_ocr = rog.RapidOCR()
        self.result = None
        self.elapse_list = None
        self.bridge = CvBridge()

    def draw_and_publish(self, img, boxes, txts, scores=None, text_score=0.5):
        
        img_resized = img#cv2.resize(img, (960, 544))
        if boxes is not None:
            for idx, (box, txt) in enumerate(zip(boxes, txts)):
                if scores is not None and float(scores[idx]) < text_score:
                    continue
                is_closed = True
                color = (0,255,0)
                thickness = 2
                pts = []
                for i in range(0, len(box)):
                    pts.append([box[i][0], box[i][1]])
                pts = np.array(pts, np.int32)
                img_resized = cv2.polylines(img_resized, [pts], is_closed, color, thickness)
                font_scale = 1.5
                text_thickness = 2
                text_org = (pts[0][0], pts[0][1])
                img_resized = cv2.putText(img_resized, txt, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)
        img_resized = np.array(img_resized, dtype="uint8")
        msg = self.bridge.cv2_to_imgmsg(img_resized, "bgr8")
        self.vis_publisher_.publish(msg)

    def listener_callback(self, msg):
        im_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        result, elapse_list = self.rapid_ocr(im_rgb)
        boxes = txts = scores = None
        if VIS: 
            if result is not None:
                print(result)
                print(elapse_list)
                boxes, txts, scores = list(zip(*result))
            np_img = np.array(im_rgb, dtype="uint8")
            self.draw_and_publish(np_img, boxes, txts, scores)

        
def main(args=None):
    rclpy.init(args=args)

    text_detector = NOVATextDetector()

    rclpy.spin(text_detector)
    text_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


