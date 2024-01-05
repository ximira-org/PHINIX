#!/usr/bin/env python3

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import time

import cv2
import numpy as np
import rapidocr_openvino as rog

VIS = True
TOPIC_PHINIX_RAW_IMG = "/phinix/rgb/image_raw"
TOPIC_VIS_IMG = "/phinix/vis_image"
TOPIC_WAKEWORD = "/phinix/wakeword"
TOPIC_DUMMY_TEXTS = "/phinix/tts_simulator/dummy_texts"
#Number of seconds to read to user when told to start reading
READING_TIME = 30

class PHINIXTextDetector(Node):

    def __init__(self):
        super().__init__('phinix_text_detector')
        self.subscription = self.create_subscription(
            Image,
            TOPIC_PHINIX_RAW_IMG,
            self.listener_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.vis_publisher_ = self.create_publisher(Image, TOPIC_VIS_IMG, 10)
        self.rapid_ocr = rog.RapidOCR()
        self.result = None
        self.elapse_list = None
        self.bridge = CvBridge()

        #Publish text I have read over tts
        self.tts_publisher = self.create_publisher(String, TOPIC_DUMMY_TEXTS, 10)
        #is the text detector actively reading text
        self.actively_reading_text = False
        #Text I have seen in this session
        self.text_read = []
        #listen for wakeword
        self.wakeword_sub = self.create_subscription(
            String, 
            TOPIC_WAKEWORD, 
            self.wakeword_callback, 
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    
    #When we get the word to identify people,id them for some amount of seconds
    def wakeword_callback(self, msg):
        if msg.data == "read_to_me":
            self.reading_timer = self.create_timer(READING_TIME, self.stop_reading)
            self.actively_reading_text = True
            self.text_read = []
            self.get_logger().info("Text Detector: Begin reading")
        elif msg.data == "stop_reading":
            self.stop_reading()
    
    def stop_reading(self):
        self.get_logger().info("Text Detector: Stop reading")
        if self.actively_reading_text:
            self.actively_reading_text = False
            self.reading_timer.destroy()

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
        if self.actively_reading_text == False:
            return
        
        im_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        result, elapse_list = self.rapid_ocr(im_rgb)
        boxes = txts = scores = None
        if VIS: 
            if result is not None:
                print(result)
                #self.get_logger().info(str(result))
                print(elapse_list)
                boxes, txts, scores = list(zip(*result))
                for txt in txts:
                    if txt not in self.text_read:
                        self.text_read.append(txt)
                        data = str(txt)
                        self.get_logger().info(data)
                        msg = String()
                        msg.data = data
                        self.tts_publisher.publish(msg)
                        break
            np_img = np.array(im_rgb, dtype="uint8")
            self.draw_and_publish(np_img, boxes, txts, scores)

        
def main(args=None):
    rclpy.init(args=args)

    text_detector = PHINIXTextDetector()

    rclpy.spin(text_detector)
    text_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


