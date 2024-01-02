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


def draw_anno(img,box,label,color,thickness):

  txt_color=(255, 255, 255)

  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(img, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)

  tf = max(thickness - 1, 1)  # font thickness

  w, h = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]  # text width, height

  outside = p1[1] - (h + 4) >= 0  # label fits outside box
  p2 = p1[0] + w, p1[1] - (h+4) if outside else p1[1] + (h + 4)
  cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled

  cv2.putText(img, label, (p1[0], p1[1] - 2\
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
        super().__init__('phinix_face_recognition')
        self.subscription = self.create_subscription(
            RosImage,
            TOPIC_PHINIX_RAW_IMG,
            self.listener_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.vis_publisher_ = self.create_publisher(RosImage, TOPIC_VIS_IMG, 10)
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

        if self.quantized:
            #Loading the quantized model
            self.ie = Core()
            self.iden_model = self.ie.compile_model("/home/jk/face_rec/yolov5/idenr_int8/idenr_int8.xml", "CPU")
        else:
            self.iden_model=identification_base()
            self.iden_model.load_state_dict(torch.load("idenr.pt"))
            self.iden_model.eval()
          # Load model

        self.det_model_path = '/home/jk/ultra_face/yolov8n-face_openvino_model/yolov8n-face.xml'
        self.core = Core()
        self.det_ov_model = self.core.read_model(self.det_model_path)
        self.ov_device = "CPU"
        self.det_ov_model.reshape({0: [1, 3, 640, 640]})
        self.det_compiled_model = self.core.compile_model(self.det_ov_model, self.ov_device)
        model = YOLO('/home/jk/ultra_face/yolov8n-face.pt')
        self.label_map = model.model.names
    
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

    def listener_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        galley_data=[]
        galley_names=[]
        for r,d,f in os.walk('../database'):
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
        faces, face_boxes, _ = self.get_face_det_results(detections, img, self.label_map)
        # print(xyxys)
        if faces is not None:
            #start=time.time()
            features=self.get_features(faces,q=self.quantized)
            #print(time.time()-start)
            person_res = self.find_match_normal(features,galley_data,galley_names)
            for ipres,pres in enumerate(person_res):
                box=face_boxes[ipres]
                person_name,person_color=pres
                person_color=(int(person_color[0]),int(person_color[1]),int(person_color[2]))
                draw_anno(img,box,person_name,person_color,thickness)
        end_time = time.time()
        print("time taken per frame = {}".format(end_time-st_time))
        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.vis_publisher_.publish(msg)

        
def main(args=None):
    rclpy.init(args=args)

    face_recognizer = PHINIXFaceRecognizer()

    rclpy.spin(face_recognizer)
    text_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

