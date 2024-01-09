#!/usr/bin/env python3

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from phinix_perception_msgs.msg import BBoxMsg
from geometry_msgs.msg import Point
from std_msgs.msg import String

import cv2
import numpy as np
import rapidocr_openvinogpu as rog

VIS = True
TOPIC_PHINIX_RAW_IMG = "/phinix/rgb/image_raw"
TOPIC_VIS_IMG = "/phinix/vis_image"
TOPIC_TEXT_REC_BBOX = "/phinix/module/text_rec/bbox"

def make_point(x, y, z=0.0):
    pt = Point()
    pt.x = x
    pt.y = y
    pt.z = z
    return pt

def clock_angle(x_val):
    if x_val < 0.2:
        return 10
    if x_val < 0.4:
        return 11
    if x_val < 0.6:
        return 12
    if x_val < 0.8:
        return 1
    if x_val < 1.0:
        return 2
    else:
        return None

class PHINIXTextDetector(Node):

    def __init__(self):
        super().__init__('phinix_text_detector')
        self.subscription = self.create_subscription(
            Image,
            TOPIC_PHINIX_RAW_IMG,
            self.listener_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.vis_publisher_ = self.create_publisher(Image, TOPIC_VIS_IMG, 10)
        self.bbox_publisher_ = self.create_publisher(BBoxMsg, TOPIC_TEXT_REC_BBOX, 10)
        self.rapid_ocr = rog.RapidOCR()
        self.result = None
        self.elapse_list = None
        self.bridge = CvBridge()
        self.bbox_msg = BBoxMsg()

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
            self.update_bbox_msg(np_img, boxes, txts, scores)
        self.bbox_publisher_.publish(self.bbox_msg)
        self.bbox_msg = BBoxMsg()

    def update_bbox_msg(self, img, boxes, txts, scores=None, text_score=0.5):
        
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
                x_min = np.min(pts[:, 0])
                x_max = np.max(pts[:, 0])
                y_min = np.min(pts[:, 1])
                y_max = np.max(pts[:, 1])

                self.bbox_msg.top_left_x_ys.append(make_point(x_min*1.0, y_min*1.0))
                self.bbox_msg.bottom_right_x_ys.append(make_point(x_max*1.0, y_max*1.0))
                text_str = String()
                text_str.data = txt
                self.bbox_msg.texts.append(text_str)
                self.bbox_msg.confidences.append(float(scores[idx]))
                self.bbox_msg.module_name.data = "text_rec"
                xmin_norm = x_min/img.shape[1]
                xmax_norm = x_max/img.shape[1]
                self.bbox_msg.clock_angle.append(clock_angle((xmin_norm + xmax_norm)/ 2))

                img_resized = cv2.polylines(img_resized, [pts], is_closed, color, thickness)
                font_scale = 1.5
                text_thickness = 2
                text_org = (pts[0][0], pts[0][1])
                img_resized = cv2.putText(img_resized, txt, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)
        img_resized = np.array(img_resized, dtype="uint8")
        msg = self.bridge.cv2_to_imgmsg(img_resized, "bgr8")
        self.vis_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    text_detector = PHINIXTextDetector()

    rclpy.spin(text_detector)
    text_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


