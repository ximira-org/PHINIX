#!/usr/bin/env python3

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from phinix_perception_msgs.msg import BBoxMsg
from cv_bridge import CvBridge
import os 

import cv2
import numpy as np
import depthai as dai
import math 
import blobconverter
import time
from pathlib import Path
import json
from ament_index_python.packages import get_package_share_directory

import threading

VIS = True
TOPIC_PHINIX_RAW_IMG = "/oak/rgb/image_raw"
TOPIC_PHINIX_DEPTH_IMG = "/oak/depth/image_raw"
TOPIC_PHINIX_DISPARITY_IMG = "/oak/disparity/image_raw"
TOPIC_PHINIX_PREVIEW_IMG = "/phinix/vis/object_det"
TOPIC_OBJ_DET_BBOX = "/phinix/module/object_det/bbox"
TOPIC_NODE_STATES = "/phinix/node_states"

object_detection_node_state_index = 5

CAM_FPS = 30.0

RES_MAP = {
    '800': {'w': 1280, 'h': 800, 'res': dai.MonoCameraProperties.SensorResolution.THE_800_P },
    '720': {'w': 1280, 'h': 720, 'res': dai.MonoCameraProperties.SensorResolution.THE_720_P },
    '400': {'w': 640, 'h': 400, 'res': dai.MonoCameraProperties.SensorResolution.THE_400_P }
}

MEDIAN_MAP = {
    "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
    "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
    "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
    "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
}

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 1, self.line_type)
    def rectangle(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 1)

class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

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
        return -1

class OAKLaunch(Node):

    def __init__(self):
        super().__init__('oak_ros')

        camera_thread = threading.Thread(target=self.camera_thread_function)
        camera_thread.start()
        '''
        self.subscription = self.create_subscription(
            Image,
            TOPIC_PHINIX_RAW_IMG,
            self.listener_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        '''
        self.node_state_subscriber = self.create_subscription(
            Int32MultiArray, 
            TOPIC_NODE_STATES, 
            self.node_state_callback, 
            10)      
    

    def update_bbox_msg(self, frame, detections, depth_frame):
        # print("preview shape = ", frame.shape)
        depth_delta = 10
        for detection in detections:
            bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            self.bbox_msg.top_left_x_ys.append(make_point(bbox[0]*1.0, bbox[1]*1.0))
            self.bbox_msg.bottom_right_x_ys.append(make_point(bbox[2]*1.0, bbox[3]*1.0))
            label_str = String()
            label_str.data = self.labels[detection.label]
            self.bbox_msg.classes.append(label_str)
            self.bbox_msg.confidences.append(detection.confidence)
            self.bbox_msg.module_name.data = "object_det"
            self.bbox_msg.clock_angle.append(clock_angle((detection.xmin + detection.xmax)/ 2))

            # depth (Z) calculation
            depth_centroid = [(bbox[0] + bbox[2]) // 2 , (bbox[1] + bbox[3]) // 2]
            x_min = max(depth_centroid[0] - int(depth_delta/2), 0) 
            y_min = max(depth_centroid[1] - int(depth_delta/2), 0) 
            x_max = min(depth_centroid[0] + int(depth_delta/2), depth_frame.shape[1]) 
            y_max = min(depth_centroid[1] + int(depth_delta/2), depth_frame.shape[0]) 
            depth_dist = np.mean(depth_frame[y_min:y_max, x_min:x_max]) / 1000 # mm to m
            self.bbox_msg.depths.append(depth_dist)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(self, name, frame, detections):
        color = (255, 0, 0)
        for detection, depth in zip(detections, self.bbox_msg.depths):
            bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            clk_angle = str(clock_angle((detection.xmin + detection.xmax)/ 2))
            cv2.putText(frame, self.labels[detection.label] + " @ " + clk_angle + " @ " + str(depth)[0:4],
                                    (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        return frame
    '''
    def listener_callback(self, msg):
        self.get_logger().info('Received an image')
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
    '''

    #Set node_active to true if node manager so
    def node_state_callback(self, node_states: Int32MultiArray):
        self.object_recognition_active = node_states.data[object_detection_node_state_index] == 1
        #self.get_logger().info(f"object_recognition_active = {self.object_recognition_active}")
    
    # Define a function that will run in a separate thread
    def camera_thread_function(self):
        self.object_recognition_active = False

        self.rgb_publisher_ = self.create_publisher(Image, TOPIC_PHINIX_RAW_IMG, 10)
        self.depth_publisher_ = self.create_publisher(Image, TOPIC_PHINIX_DEPTH_IMG, 10)
        self.disparity_publisher_ = self.create_publisher(Image, TOPIC_PHINIX_DISPARITY_IMG, 10)
        self.preview_publisher_ = self.create_publisher(Image, TOPIC_PHINIX_PREVIEW_IMG, 10)
        self.bbox_publisher_ = self.create_publisher(BBoxMsg, TOPIC_OBJ_DET_BBOX, 10)
        self.bridge = CvBridge()
        self.lrcheck = True  # Better handling for occlusions
        self.extended = False  # Closer-in minimum depth, disparity range is doubled
        self.subpixel = False  # Better accuracy for longer distance, fractional disparity 32-levels
        self.W = 0
        self.H = 0
        self.pkg_share_dir = get_package_share_directory("oak_ros")
        self.model_dir = os.path.join(self.pkg_share_dir, "yolo_model")
        self.nnPath = os.path.join(self.model_dir, 'yolov8n_coco_640x352_openvino_2022.1_6shave.blob')
        self.config_dir = os.path.join(self.pkg_share_dir, "yolo_json")
        self.yolo_configPath = Path(self.config_dir+"/yolov8_640x352.json")
        with self.yolo_configPath.open() as f:
            self.yolo_config = json.load(f)
        self.nnConfig = self.yolo_config.get("nn_config", {})
        # parse input shape
        if "input_size" in self.nnConfig:
            self.W, self.H = tuple(map(int, self.nnConfig.get("input_size").split('x')))
        self.metadata = self.nnConfig.get("NN_specific_metadata", {})
        self.classes = self.metadata.get("classes", {})
        self.coordinates = self.metadata.get("coordinates", {})
        self.anchors = self.metadata.get("anchors", {})
        self.anchorMasks = self.metadata.get("anchor_masks", {})
        self.iouThreshold = self.metadata.get("iou_threshold", {})
        self.confidenceThreshold = self.metadata.get("confidence_threshold", {})
        self.nnMappings = self.yolo_config.get("mappings", {})
        self.labels = self.nnMappings.get("labels", {})
        self.resolution = RES_MAP['400']
        self.median = MEDIAN_MAP["7x7"]
        self.device = dai.Device()
        self.calibData = self.device.readCalibration()
        self.pipeline = dai.Pipeline()

        self.camLeft = self.pipeline.create(dai.node.MonoCamera)
        self.camRight = self.pipeline.create(dai.node.MonoCamera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.xoutLeft = self.pipeline.create(dai.node.XLinkOut)
        self.xoutRight = self.pipeline.create(dai.node.XLinkOut)
        self.xoutDisparity = self.pipeline.create(dai.node.XLinkOut)
        self.xoutStereoCfg = self.pipeline.create(dai.node.XLinkOut)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.camLeft.setCamera("left")
        self.camLeft.setResolution(self.resolution['res'])
        self.camLeft.setFps(CAM_FPS)
        self.camRight.setCamera("right")
        self.camRight.setResolution(self.resolution['res'])
        self.camRight.setFps(CAM_FPS)

        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        # self.stereo.setOutputSize(self.camLeft.getResolutionWidth(), self.camRight.getResolutionHeight())
        self.stereo.setOutputSize(640, 352)
        self.stereo.initialConfig.setMedianFilter(self.median)  # KERNEL_7x7 default
        self.stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
        self.stereo.setLeftRightCheck(self.lrcheck)
        self.stereo.setExtendedDisparity(self.extended)
        self.stereo.setSubpixel(self.subpixel)

        self.config = self.stereo.initialConfig.get()
        # self.config.postProcessing.speckleFilter.enable = False
        # self.config.postProcessing.speckleFilter.speckleRange = 50
        # self.config.postProcessing.temporalFilter.enable = True # slows down
        # self.config.postProcessing.spatialFilter.enable = True # slows down
        # self.config.postProcessing.spatialFilter.holeFillingRadius = 2 
        # self.config.postProcessing.spatialFilter.numIterations = 1
        # self.config.postProcessing.thresholdFilter.minRange = 400
        # self.config.postProcessing.thresholdFilter.maxRange = 15000
        self.config.postProcessing.decimationFilter.decimationFactor = 1
        self.stereo.initialConfig.set(self.config)

        self.xoutLeft.setStreamName("left")
        self.xoutRight.setStreamName("right")
        self.xoutDisparity.setStreamName("disparity")
        self.xoutStereoCfg.setStreamName("stereo_cfg")
        self.xoutRgb.setStreamName("rgb")

        self.camLeft.out.link(self.stereo.left)
        self.camRight.out.link(self.stereo.right)
        self.stereo.syncedLeft.link(self.xoutLeft.input)
        self.stereo.syncedRight.link(self.xoutRight.input)
        self.stereo.disparity.link(self.xoutDisparity.input)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        # self.stereo.setFps(15)
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutVideo = self.pipeline.create(dai.node.XLinkOut)

        self.xoutVideo.setStreamName("video")
        # self.Properties
        # self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        # self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        # self.camRgb.setVideoSize(3840,2160)
        # self.camRgb.setIspScale(1,1)
        self.camRgb.video.link(self.xoutVideo.input)
        self.camRgb.setPreviewSize(self.W, self.H)
        self.camRgb.setInterleaved(False)
        self.camRgb.setFps(CAM_FPS)
        # self.camRgb.setPreviewKeepAspectRatio(False)

        # self.nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
        self.detectionNetwork = self.pipeline.create(dai.node.YoloDetectionNetwork)

        # Network specific settings
        self.detectionNetwork.setConfidenceThreshold(self.confidenceThreshold)
        self.detectionNetwork.setNumClasses(self.classes)
        self.detectionNetwork.setCoordinateSize(self.coordinates)
        self.detectionNetwork.setAnchors(self.anchors)
        self.detectionNetwork.setAnchorMasks(self.anchorMasks)
        self.detectionNetwork.setIouThreshold(self.iouThreshold)
        self.detectionNetwork.setBlobPath(self.nnPath)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.input.setBlocking(False)

        self.nnOut = self.pipeline.create(dai.node.XLinkOut)
        self.nnOut.setStreamName('nn')

        # camera frames linked to NN input node
        self.camRgb.preview.link(self.detectionNetwork.input)
        self.detectionNetwork.passthrough.link(self.xoutRgb.input)
        self.detectionNetwork.out.link(self.nnOut.input)

        self.xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        self.xoutDepth.setStreamName("depth")
        self.stereo.depth.link(self.xoutDepth.input)
        self.stereo.outConfig.link(self.xoutStereoCfg.input)
        self.bbox_msg = BBoxMsg()

        with self.device:
            self.device.startPipeline(self.pipeline)
            self.device.setIrLaserDotProjectorBrightness(850) # in mA, 0..1200
            self.device.setIrFloodLightBrightness(0) # in mA, 0..1500
            inCfg = self.device.getOutputQueue("stereo_cfg", 8, blocking=False)
            # Create a receive queue for each stream
            # qList = [device.getOutputQueue(stream, 8, blocking=False) for stream in streams]
            config = inCfg.get()

            self.qFrames = self.device.getOutputQueue(name="video", maxSize=8, blocking=False)
            self.qDet = self.device.getOutputQueue(name="nn", maxSize=8, blocking=False)
            self.qDisp = self.device.getOutputQueue(name="disparity", maxSize=8, blocking=False)
            self.qDepth = self.device.getOutputQueue(name="depth", maxSize=8, blocking=False)
            self.qleft = self.device.getOutputQueue(name="left", maxSize=8, blocking=False)
            self.qright = self.device.getOutputQueue(name="right", maxSize=8, blocking=False)
            detections = []
            fps = FPSHandler()
            text = TextHelper()
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            # qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            startTime = time.monotonic()
            counter = 0
            color2 = (255, 255, 255)

            while True:
                inDepth = self.qDepth.get()
                inDisparity = self.qDisp.get()
                
                
                if inDisparity is not None:
                    dis_frame = inDisparity.getCvFrame()
                    maxDisp = self.stereo.initialConfig.getMaxDisparity()
                    # Normalization for better visualization
                    dis_frame = (dis_frame * (255 / maxDisp)).astype(np.uint8)

                    # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
                    dis_frame = cv2.applyColorMap(dis_frame, cv2.COLORMAP_JET)
                    ros_disparity = self.bridge.cv2_to_imgmsg(dis_frame, "bgr8")
                    ros_disparity.header.stamp = self.get_clock().now().to_msg()
                    self.disparity_publisher_.publish(ros_disparity)
                if inDepth is not None:
                    dep_frame = inDepth.getFrame()
                    dep_frame = dep_frame.astype(np.uint16)
                    ros_depth = self.bridge.cv2_to_imgmsg(dep_frame, "16UC1")
                    ros_depth.header.stamp = self.get_clock().now().to_msg()
                    self.depth_publisher_.publish(ros_depth)

                if not self.object_recognition_active:
                    continue
                
                inRgb = self.qRgb.get()
                inDet = self.qDet.get()
                inFrame = self.qFrames.get()
                
                if inDet is not None:
                    detections = inDet.detections
                    counter += 1

                if inRgb is not None:
                    det_frame = inRgb.getCvFrame()
                    cv2.putText(det_frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                                (2, det_frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
            
                if inFrame is not None:
                    full_frame = inFrame.getCvFrame()
                    full_frame = cv2.resize(full_frame, (640, 352))
                    ros_full_Frame = self.bridge.cv2_to_imgmsg(full_frame, "bgr8")
                    ros_full_Frame.header.stamp = self.get_clock().now().to_msg()
                    self.rgb_publisher_.publish(ros_full_Frame)

                self.update_bbox_msg(det_frame, detections, dep_frame)
                self.bbox_msg.header.stamp = self.get_clock().now().to_msg()
                self.bbox_publisher_.publish(self.bbox_msg)
                det_frame = self.displayFrame("rgb", det_frame, detections)
                # print("preview shape = ", frame.shape)
                ros_preview = self.bridge.cv2_to_imgmsg(det_frame, "bgr8")
                ros_preview.header.stamp = self.get_clock().now().to_msg()
                self.preview_publisher_.publish(ros_preview)
                self.bbox_msg = BBoxMsg()
        
def main(args=None):
    rclpy.init(args=args)

    oak_ros_node = OAKLaunch()

    rclpy.spin(oak_ros_node)
    oak_ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


