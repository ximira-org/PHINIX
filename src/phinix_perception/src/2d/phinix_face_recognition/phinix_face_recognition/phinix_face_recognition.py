#!/usr/bin/env python3

import time
import os
import argparse
import sys
from pathlib import Path
import yaml
from typing import Tuple, Dict

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from phinix_perception_msgs.msg import BBoxMsg
from geometry_msgs.msg import Point
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
import message_filters

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from PIL import Image

from openvino.runtime import Core, Model
from openvino.runtime import serialize, Tensor


from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.plotting import colors

VIS = True
TOPIC_PHINIX_RAW_IMG = "/phinix/rgb/image_raw"
TOPIC_VIS_IMG = "/phinix/vis/face_rec"
TOPIC_FACE_REC_BBOX = "/phinix/module/face_rec/bbox"
TOPIC_PHINIX_RAW_DEPTH = "/phinix/depth/image_raw"
TOPIC_NODE_STATES = "/phinix/node_states"

node_state_index = 1

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

def draw_anno(img,box,label,color,thickness, clk_angle, depth):

  txt_color=(0, 255, 0)

  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(img, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)

  tf = max(thickness - 1, 1)  # font thickness

  w, h = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]  # text width, height

  outside = p1[1] - (h + 4) >= 0  # label fits outside box
  p2 = p1[0] + w, p1[1] - (h+4) if outside else p1[1] + (h + 4)
  cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled

  cv2.putText(img, label + " @ " + clk_angle + " @ " + depth, (p1[0], p1[1] - 2\
          if outside else p1[1] + h), 0, thickness / 3, txt_color,thickness=tf, lineType=cv2.LINE_AA)
  return(img)


class identification_base(nn.Module): ###

    def __init__(self):
        super(identification_base, self).__init__()

        self.model = torchvision.models.regnet_y_800mf(pretrained=False)
        self.model.fc= nn.Sequential(nn.Dropout(0.5),nn.Linear(784, 512, bias=False),
                                            nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True))
    def forward(self, anc):
        f_anc=self.model(anc)
        return  F.normalize(f_anc)
        
class PHINIXFaceRecognizer(Node):

    def __init__(self):
        super().__init__('phinix_face_rec')
        self.declare_parameter('database_dir', rclpy.Parameter.Type.STRING)
        self.database_dir = self.get_parameter('database_dir').value
        self.vis_publisher_ = self.create_publisher(RosImage, TOPIC_VIS_IMG, 10)
        self.bbox_publisher_ = self.create_publisher(BBoxMsg, TOPIC_FACE_REC_BBOX, 10)
        self.bridge = CvBridge()
        self.quantized=False
        self.conf_thres=0.45
        self.iou_thres=0.45
        self.max_size=256

        if not self.quantized:
            self.reg_thresh=0.8
        else:
            self.reg_thresh=0.9

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        pkg_share_dir = get_package_share_directory("phinix_face_recognition")
        model_dir = os.path.join(pkg_share_dir, "models")
        self.det_model_path = os.path.join(model_dir, 'yolov8n-face.xml')
        self.core = Core()
        self.get_logger().info('det_model_path : "%s"' % self.det_model_path)
        self.det_ov_model = self.core.read_model(self.det_model_path)
        self.ov_device = "CPU"
        self.det_ov_model.reshape({0: [1, 3, 640, 640]})
        self.det_compiled_model = self.core.compile_model(self.det_ov_model, self.ov_device)
        model = YOLO(self.det_model_path.replace(".xml", ".pt"))
        self.label_map = model.model.names
        if self.quantized:
            #Loading the quantized model
            self.ie = Core()
            self.iden_model = ie.compile_model("/home/jk/face_rec/yolov5/idenr_int81/idenr_int8.xml", "GPU")
        else:
            self.iden_model=identification_base()
            self.iden_model.load_state_dict(torch.load(os.path.join(model_dir, "idenr.pt")))
            self.iden_model.eval()
        self.bbox_msg = BBoxMsg()
        self.rgb_image_sub = message_filters.Subscriber(self, RosImage, TOPIC_PHINIX_RAW_IMG)
        self.depth_img_sub = message_filters.Subscriber(self, RosImage, TOPIC_PHINIX_RAW_DEPTH)
        self.node_state_subscriber = self.create_subscription(Int32MultiArray, TOPIC_NODE_STATES, self.node_state_callback, 10)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.rgb_image_sub, self.depth_img_sub), 5, 0.01)
        self._synchronizer.registerCallback(self.sync_callback)

    @torch.no_grad()
    def get_features(self, faces,q=True): ###
        if not q:
            for index,img in enumerate(faces):
                face_img=torch.from_numpy(cv2.resize(img,(160,160)).transpose(2, 0, 1)).to(self.device)
                if index==0:
                    face_img_batch=torch.unsqueeze(face_img,0)
                else:
                    face_img_batch=torch.cat((face_img_batch,torch.unsqueeze(face_img,0)),0)

            face_img_batch=face_img_batch.float()/255.0
            img_embedding =self.iden_model(face_img_batch)
            return(img_embedding)
        else:
            img_embedding=[]
            for i in range(len(faces)):
                infer_request = self.iden_model.create_infer_request()
                #print(i,faces[i].shape)
                input_img=np.expand_dims(cv2.resize(faces[i],(160,160)).transpose(2, 0, 1),0)/255
                input_img=input_img.astype(np.float32)
                input_tensor = Tensor(np.ascontiguousarray(input_img), shared_memory=True)
                infer_request.set_input_tensor(input_tensor)
                infer_request.start_async()
                infer_request.wait()
                output = infer_request.get_output_tensor()
                pred=torch.tensor(output.data)
                img_embedding.append(pred)
            #print("q running")
            img_embedding=torch.stack(img_embedding).to('cpu')
            return(img_embedding)

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

    def find_match_normal(self, features,galley_data,galley_names):
        if len(galley_data)==0:
            cases_results=[["No match",(0,0,0)] for ii in range(len(features))]
            return(cases_results)

        cases_results=[]
        distances=torch.cdist(features,galley_data[:,:-3])

        for index in range(len(features)):
            # print(torch.min(distances[index,:]))
            if torch.min(distances[index,:])<=self.reg_thresh:
                min_ind=torch.argmin(distances[index,:])
                cases_results.append([galley_names[min_ind],galley_data[min_ind][-3:]])

            else:
                cases_results.append(["No match",(0,0,0)])
        return cases_results

    def plot_one_box(self, box:np.ndarray, img:np.ndarray, 
            color:Tuple[int, int, int] = None, 
            mask:np.ndarray = None, label:str = None, line_thickness:int = 5):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        if mask is not None:
            image_with_mask = img.copy()
            mask
            cv2.fillPoly(image_with_mask, pts=[mask.astype(int)], color=color)
            img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
        return img


    def draw_results(self, results:Dict, source_image:np.ndarray, label_map:Dict):
        boxes = results["det"]
        masks = results.get("segment")
        h, w = source_image.shape[:2]
        for idx, (*xyxy, conf, lbl) in enumerate(boxes):
            label = f'{label_map[int(lbl)]} {conf:.2f}'
            mask = masks[idx] if masks is not None else None
            source_image = plot_one_box(xyxy, source_image, mask=mask, label=label, color=colors(int(lbl)), line_thickness=1)
        return source_image
    
    def get_face_det_results(self, results:Dict, source_image:np.ndarray, label_map:Dict):
        boxes = results["det"]
        masks = results.get("segment")
        xyxys = []
        confs = []
        faces = []

        if len(boxes) == 0:
            faces = None
            return (faces, xyxys, confs)

        h, w = source_image.shape[:2]
        for idx, (*xyxy, conf, lbl) in enumerate(boxes):
            label = f'{label_map[int(lbl)]} {conf:.2f}'
            mask = masks[idx] if masks is not None else None
            # opencv notation
            xyxys.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
            confs.append(conf)
            faces.append(source_image[int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])])
        return (faces, xyxys, confs)

    def letterbox(self, img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), 
            color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, 
            scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


    def preprocess_image(self, img0: np.ndarray):
        
        # resize
        img = self.letterbox(img0)[0]
        # Convert HWC to CHW
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img


    def image_to_tensor(self, image:np.ndarray):

        input_tensor = image.astype(np.float32)  # uint8 to fp32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

        # add batch dimension
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor

    def postprocess(self, 
        pred_boxes:np.ndarray,
        input_hw:Tuple[int, int],
        orig_img:np.ndarray,
        min_conf_threshold:float = 0.25,
        nms_iou_threshold:float = 0.7,
        agnosting_nms:bool = False,
        max_detections:int = 300,
        pred_masks:np.ndarray = None,
        retina_mask:bool = False
    ):
        nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
        preds = ops.non_max_suppression(
            torch.from_numpy(pred_boxes),
            # TODO(Jagadish) : remove dead code and check below parameter values
            # conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, 
            # multi_label=False, labels=(), max_det=300, max_time_img=0.05, 
            # max_nms=30000, max_wh=7680
            # **nms_kwargs
        )
        results = []
        proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append({"det": [], "segment": []})
                continue
            if proto is None:
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                results.append({"det": pred})
                continue
            if retina_mask:
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
                segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
            results.append({"det": pred[:, :6].numpy(), "segment": segments})
        return results



    def detect(self, image:np.ndarray, model:Model):
        num_outputs = len(model.outputs)
        preprocessed_image = self.preprocess_image(image)
        input_tensor = self.image_to_tensor(preprocessed_image)
        result = model(input_tensor)
        boxes = result[model.output(0)]
        masks = None
        if num_outputs > 1:
            masks = result[model.output(1)]
        input_hw = input_tensor.shape[2:]
        detections = self.postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
        return detections

    def sync_callback(self, rgb_msg, depth_msg):
        #early exit if this node is not enabled in node manager
        if self.node_active == False:
            return
        
        img = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)
        im_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        galley_data=[]
        galley_names=[]
        for r,d,f in os.walk(self.database_dir):
            for file in f:
                if '.npy' in file:
                    galley_data.append(torch.from_numpy(np.load(os.path.join(r,file))))
                    galley_names.append(file[:-4])

        if len(galley_data)!=0:
            galley_data=torch.stack(galley_data).to(self.device)
        else:
            galley_data=torch.tensor([],dtype=torch.float32).to(self.device)

        width  = img.shape[1]   # float `width`
        height = img.shape[0]  # float `height`
        font_scale = (width + height) / 1500
        thickness = max(1, int(font_scale * 2))


        st_time = time.time()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.detect(img_rgb, self.det_compiled_model)[0]
        print(detections)
        faces, face_boxes, confs = self.get_face_det_results(detections, img, self.label_map)
        if faces is not None:
            features=self.get_features(faces,q=self.quantized)
            person_res = self.find_match_normal(features,galley_data,galley_names)
            img = self.update_bbox_msg(img, face_boxes, confs, person_res, thickness, im_depth)

        self.bbox_msg.header.stamp = rgb_msg.header.stamp
        self.bbox_publisher_.publish(self.bbox_msg)
        self.bbox_msg = BBoxMsg()
        end_time = time.time()
        print("time taken per frame = {}".format(end_time-st_time))
        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.vis_publisher_.publish(msg)

    def update_bbox_msg(self, frame, face_boxes, confs, person_res, thickness, depth_frame):
        depth_delta = 10
        for ipres,pres in enumerate(person_res):
            bbox=face_boxes[ipres]
            conf = confs[ipres]
            person_name,person_color=pres
            person_color=(int(person_color[0]),int(person_color[1]),int(person_color[2]))
            self.bbox_msg.top_left_x_ys.append(make_point(bbox[0]*1.0, bbox[1]*1.0))
            self.bbox_msg.bottom_right_x_ys.append(make_point(bbox[2]*1.0, bbox[3]*1.0))
            person_str = String()
            person_str.data = person_name
            self.bbox_msg.people_names.append(person_str)
            self.bbox_msg.confidences.append(conf)
            self.bbox_msg.module_name.data = "face_rec"
            xmin_norm = bbox[0]/frame.shape[1]
            xmax_norm = bbox[2]/frame.shape[1]
            clk_angle = clock_angle((xmin_norm + xmax_norm)/ 2)
            self.bbox_msg.clock_angle.append(clk_angle)

            # depth (Z) calculation
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            depth_centroid = [(x_min + x_max) // 2 , (y_min + y_max) // 2]
            x_min = max(depth_centroid[0] - int(depth_delta/2), 0) 
            y_min = max(depth_centroid[1] - int(depth_delta/2), 0) 
            x_max = min(depth_centroid[0] + int(depth_delta/2), depth_frame.shape[1]) 
            y_max = min(depth_centroid[1] + int(depth_delta/2), depth_frame.shape[0]) 
            depth_dist = np.mean(depth_frame[y_min:y_max, x_min:x_max]) / 1000 # mm to m
            self.bbox_msg.depths.append(depth_dist)
            draw_anno(frame,bbox,person_name,person_color,
                        thickness, str(clk_angle), str(depth_dist)[0:4])
        return frame

    #Set node_active to true if node manager so
    def node_state_callback(self, node_states: Int32MultiArray):
        self.node_active = node_states.data[node_state_index] == 1
        
def main(args=None):
    rclpy.init(args=args)

    face_recognizer = PHINIXFaceRecognizer()

    rclpy.spin(face_recognizer)
    text_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


