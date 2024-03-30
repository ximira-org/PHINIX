import os
import sys
import cv2
import time
import random
import json
import pickle
import shutil
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
#import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import segmentation_models_pytorch as smp
from torchvision import datasets
import random
import torch
import torch.nn.functional as F
import sys
import torchvision
import time
from PIL import Image
#import albumentations as A
from torchvision.transforms import functional as FF
from torch import nn
import warnings  # to disable warnings on export to ONNX
from torch.jit import TracerWarning
#from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
import argparse
import imghdr
import mimetypes
from openvino.runtime import Core, serialize, Tensor

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

#TOPIC_IN_VIDEO = "/oak/rgb/image_raw"
TOPIC_IN_VIDEO = "/phinix/simulator/video"
TOPIC_VIS_IMG = "/phinix/vis/sidewalk"
TOPIC_NODE_STATES = "/phinix/node_states"


colors= {0:(255,0,0),1:(0,0,255),2:(0,255,0),3:(10,60,40),4:(110,80,90),5:(90,80,53),
            6:(200,36,1),7:(90,180,120),8:(150,150,180),9:(210,90,10)}
BACKGRND = (210,90,10)
colors= {0:BACKGRND,1:(0,0,255),2:BACKGRND,3:BACKGRND,4:(0,255,0),5:(145,20,255),
            6:BACKGRND,7:(0,255,255),8:BACKGRND,9:BACKGRND}
all_classes={"SideRoads":0,"Road":1,"Curb":2,"WaterDrain":3,"Sidewalk":4,"TactilePav":5,"Tactilepav":5,"Manhole":6,"Road-Mark":7,"Vehicle-Bar":8,"Background":9}
resize_height = 480
resize_width = int(480*1.78)

path_detection_node_state_index = 1

class PHINIXSidewalkDetector(Node):

    def __init__(self):
        super().__init__('phinix_sidewalk_detector')
        self.get_logger().info("PHINIXSidewalkDetector has been started")
        self.vis_publisher_ = self.create_publisher(Image, TOPIC_VIS_IMG, 10)
        self.bridge = CvBridge()

        self.subscriber_ = self.create_subscription(
            Image,
            TOPIC_IN_VIDEO,
            self.listener_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))

        self.node_state_subscriber = self.create_subscription(
            Int32MultiArray, 
            TOPIC_NODE_STATES, 
            self.node_state_callback, 
            10) 
        
        pkg_share_dir = get_package_share_directory("phinix_sidewalk_detector")
        model_dir = os.path.join(pkg_share_dir, "models")
        model_file_path = os.path.join(pkg_share_dir, "models/DeepLabV3Plus_timm-regnety_064_int8.xml")
        self.ie = Core()
        self.compiled_model = self.ie.compile_model(model_file_path, "GPU")
        self.infer_request = self.compiled_model.create_infer_request()

        # Define the directory for saving images
        self.output_image_directory = os.path.join(pkg_share_dir, "saved_images")
        os.makedirs(self.output_image_directory, exist_ok=True)
        self.image_sequence_number = 0
        self.capture_time = time.time()

        self.node_active = False

        self.get_logger().info("PHINIXSidewalkDetector Init complete")

    def listener_callback(self, rgb_msg):
        #if not self.node_active:
        #    return
        
        img = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)
        sample= cv2.resize(img,(448,448))
        #vis_img = sample.copy()
        sample = FF.to_tensor(sample)
        OPENVINO = True
        if OPENVINO:
            #t1 = time.time()
            input_tensor = Tensor(array=torch.unsqueeze(sample,0).numpy(), shared_memory=True)
            self.infer_request.set_input_tensor(input_tensor)
            self.infer_request.start_async()
            self.infer_request.wait()
            ov_out = self.infer_request.get_output_tensor()
            #t2 = time.time()
            #self.get_logger().info("openvino time = {}".format(t2 - t1))
            #t1 = time.time()
            ov_out_np=ov_out.data
            ov_out_np=np.argmax(ov_out_np, axis=1)
            ov_out_np=ov_out_np[0,:,:]
            pred = ov_out_np       
            mask_rgb = np.zeros((448, 448, 3), dtype=np.uint8)
            #t2 = time.time()
            #self.get_logger().info("data prep time = {}".format(t2 - t1))
            #t1 = time.time()
            for value, rgb in colors.items():
                mask_rgb[pred == value] = rgb
            
            vis_img = cv2.resize(img, (resize_width, resize_height))
            mask_rgb = cv2.resize (mask_rgb, (resize_width, resize_height))
            #t2 = time.time()
            #self.get_logger().info("data parse time = {}".format(t2 - t1))
            blended = cv2.addWeighted(vis_img, 0.7, mask_rgb, 0.3, 0)
            #cv2.imshow("pred", blended)
            #cv2.waitKey(1)

            #Save output images

            if time.time() - self.capture_time > 1:
                # Save the image
                rgb_image_name = f"rgb_output_image_{self.image_sequence_number}.png"
                rgb_image_path = os.path.join(self.output_image_directory, rgb_image_name)
                cv2.imwrite(rgb_image_path, vis_img)

                path_image_name = f"path_output_image_{self.image_sequence_number}.png"
                path_image_path = os.path.join(self.output_image_directory, path_image_name)
                cv2.imwrite(path_image_path, mask_rgb)

                self.image_sequence_number += 1
                self.capture_time = time.time()

            #self.get_logger().info("Publish sidewalk img")
            self.vis_publisher_.publish(self.bridge.cv2_to_imgmsg(blended, "bgr8"))
    
    #Set node_active to true if node manager so
    def node_state_callback(self, node_states: Int32MultiArray):
        self.node_active = node_states.data[path_detection_node_state_index] == 1
        

def main(args=None):
    rclpy.init(args=args)

    sidewalk_detector = PHINIXSidewalkDetector()

    rclpy.spin(sidewalk_detector)
    sidewalk_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()