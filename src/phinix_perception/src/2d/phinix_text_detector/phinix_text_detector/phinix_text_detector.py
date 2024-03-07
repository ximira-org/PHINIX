#!/usr/bin/env python3

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from phinix_perception_msgs.msg import BBoxMsg
from geometry_msgs.msg import Point
from std_msgs.msg import String
import message_filters

import cv2
import numpy as np
import rapidocr_openvinogpu as rog
from std_msgs.msg import Int32MultiArray




VIS = True
TOPIC_PHINIX_RAW_IMG = "/phinix/rgb/image_raw"
TOPIC_PHINIX_RAW_DEPTH = "/phinix/depth/image_raw"
TOPIC_VIS_IMG = "/phinix/vis_image"
TOPIC_TEXT_REC_BBOX = "/phinix/module/text_rec/bbox"
TOPIC_NODE_STATES = "/phinix/node_states"

node_state_index = 2
TEXT_DETECTOR_SKIP_EVERY = 4

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
        self.vis_publisher_ = self.create_publisher(Image, TOPIC_VIS_IMG, 10)
        self.bbox_publisher_ = self.create_publisher(BBoxMsg, TOPIC_TEXT_REC_BBOX, 10)
        self.rapid_ocr = rog.RapidOCR()
        self.result = None
        self.elapse_list = None
        self.bridge = CvBridge()
        self.bbox_msg = BBoxMsg()
        self.rgb_image_sub = message_filters.Subscriber(self, Image, TOPIC_PHINIX_RAW_IMG)
        self.depth_img_sub = message_filters.Subscriber(self, Image, TOPIC_PHINIX_RAW_DEPTH)
        self.node_state_subscriber = self.create_subscription(Int32MultiArray, TOPIC_NODE_STATES, self.node_state_callback, 10)

        # Am I active in the node manager
        self.node_active = False

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.rgb_image_sub, self.depth_img_sub), 5, 0.1)
        self._synchronizer.registerCallback(self.sync_callback)
        self.get_logger().info("Text Detector Node is ready")

        self.skip_every = TEXT_DETECTOR_SKIP_EVERY
        self.skip_counter = 0

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

    def sync_callback(self, rgb_msg, depth_msg):
        #self.get_logger().info("sync_callback")
        #early exit if this node is not enabled in node manager
        if self.node_active == False:
            return
        
        if self.skip_counter < self.skip_every:
            self.skip_counter += 1
            return
        self.skip_counter = 0
        
        self.get_logger().info(str(rgb_msg.height) + ", " + str(rgb_msg.width))
        self.get_logger().info("sync_callback")
        im_rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)
        #im_rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(544, 960, -1)
        im_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        result, elapse_list = self.rapid_ocr(im_rgb)
        boxes = txts = scores = None
        if VIS: 
            if result is not None:
                print(result)
                print(elapse_list)
                boxes, txts, scores = list(zip(*result))
            np_img = np.array(im_rgb, dtype="uint8")
            self.update_bbox_msg(np_img, boxes, txts, im_depth, scores)
        self.bbox_msg.header.stamp = rgb_msg.header.stamp
        self.bbox_publisher_.publish(self.bbox_msg)
        self.bbox_msg = BBoxMsg()
    
    #Set node_active to true if node manager so
    def node_state_callback(self, node_states: Int32MultiArray):
        self.node_active = node_states.data[node_state_index] == 1

    def update_bbox_msg(self, img, boxes, txts, depth_frame, 
                                    scores=None, text_score=0.5):
        
        depth_delta = 10
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
                clk_angle = clock_angle((xmin_norm + xmax_norm)/ 2)
                self.bbox_msg.clock_angle.append(clk_angle)
                img_resized = cv2.polylines(img_resized, [pts], is_closed, color, thickness)
                font_scale = 1.0
                text_thickness = 1
                text_org = (pts[0][0], pts[0][1])
                

                # depth (Z) calculation
                depth_centroid = [(x_min + x_max) // 2 , (y_min + y_max) // 2]
                x_min = max(depth_centroid[0] - int(depth_delta/2), 0) 
                y_min = max(depth_centroid[1] - int(depth_delta/2), 0) 
                x_max = min(depth_centroid[0] + int(depth_delta/2), depth_frame.shape[1]) 
                y_max = min(depth_centroid[1] + int(depth_delta/2), depth_frame.shape[0]) 
                depth_dist = np.mean(depth_frame[y_min:y_max, x_min:x_max]) / 1000 # mm to m
                # print(depth_dist)
                self.bbox_msg.depths.append(depth_dist)
                img_resized = cv2.putText(img_resized, 
                            txt + " @ " + str(clk_angle) + " @ " + str(depth_dist)[0:4], 
                            text_org, cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, color, text_thickness, cv2.LINE_AA)

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


