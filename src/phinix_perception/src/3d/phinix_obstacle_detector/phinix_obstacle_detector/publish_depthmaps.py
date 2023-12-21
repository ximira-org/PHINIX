#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2

bridge = CvBridge()

TOPIC_NAME = "depthmap_topic"
WIDTH = 1920
HEIGHT = 1080
    
class DummyDepthmapNode(Node):
    def __init__(self):
        super().__init__("publish_depthmaps")
        self.publisher = self.create_publisher(Image, TOPIC_NAME, 10)        
        self.timer = self.create_timer(1.0, self.publish_dummy)
    
    def publish_dummy(self):
        # Generate random test depthmap
        depthmap_np = np.random.rand(WIDTH, HEIGHT).astype(np.float32)
        # Convert numpy array to ros Image msg
        # ros_image = bridge.cv2_to_imgmsg(depthmap_np, "32SC1")
        ros_image = bridge.cv2_to_imgmsg(depthmap_np, "32FC1")
        # Publish
        self.publisher.publish(ros_image)
        # Log
        print("Publishing Depthmap test ", self.get_clock().now())

def main():
    rclpy.init()
    node = DummyDepthmapNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()