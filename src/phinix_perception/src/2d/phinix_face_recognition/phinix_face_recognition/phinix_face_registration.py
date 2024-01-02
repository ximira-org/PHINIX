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
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

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
# import rapidocr_openvinogpu as rog
import rapidocr_openvino as rog

VIS = True
TOPIC_PHINIX_RAW_IMG = "/phinix/rgb/image_raw"
TOPIC_VIS_IMG = "/phinix/vis/face_reg"
TOPIC_FACE_TEXTS = "/phinix/tts/face_reg"
NUM_PIXELS_BELOW_CHIN = 400 # pixels
NUM_PIXELS_LEFT_OF_FACE = 30 # pixels
NAME_READING_PERIOD = 10 # seconds
RECORDING_TIME = 30 # seconds


class identification_base(nn.Module): ###

    def __init__(self):
        super(identification_base, self).__init__()

        self.model = torchvision.models.regnet_y_800mf(pretrained=False)
        self.model.fc= nn.Sequential(nn.Dropout(0.5),nn.Linear(784, 512, bias=False),
                                            nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True))
    def forward(self, anc):
        f_anc=self.model(anc)
        return  F.normalize(f_anc)
        
class PHINIXFaceRegistrer(Node):

    def __init__(self):
        super().__init__('phinix_face_reg')
        self.declare_parameter('reg_dir', rclpy.Parameter.Type.STRING)
        self.declare_parameter('database_dir', rclpy.Parameter.Type.STRING)
        self.database_dir = self.get_parameter('database_dir').value
        self.reg_dir = self.get_parameter('reg_dir').value
        self.subscription = self.create_subscription(
            RosImage,
            TOPIC_PHINIX_RAW_IMG,
            self.listener_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.vis_publisher_ = self.create_publisher(RosImage, TOPIC_VIS_IMG, 10)
        self.text_publisher = self.create_publisher(String, TOPIC_FACE_TEXTS, 10)
        self.bridge = CvBridge()
        self.quantized=False
        self.conf_thres=0.45
        self.iou_thres=0.45
        self.max_size=256
        self.name_dict = {}

        if not self.quantized:
            self.reg_thresh=0.9
        else:
            self.reg_thresh=0.9

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.rapid_ocr = rog.RapidOCR()
        self.result = None
        self.elapse_list = None
        self.selected_face = None
        self.selected_face_bbox = None
        self.name_start_time = None
        self.save_start_time = None
        self.text_reading_complete = False
        self.reg_per_dir = None
        self.img_ctr = 0
        self.saving_complete = False
        self.reg_complete = False
        self.reg_person_name = None
        pkg_share_dir = get_package_share_directory("phinix_face_recognition")
        model_dir = os.path.join(pkg_share_dir, "models")
        self.det_model_path = os.path.join(model_dir, 'yolov8n-face.xml')
        self.core = Core()
        self.det_ov_model = self.core.read_model(self.det_model_path)
        self.ov_device = "GPU"
        if self.ov_device != "CPU":
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
                input_img=np.expand_dims(cv2.resize(faces[i],(160,160)).transpose(2, 0, 1),0)/255
                input_img=input_img.astype(np.float32)
                input_tensor = Tensor(np.ascontiguousarray(input_img), shared_memory=True)
                infer_request.set_input_tensor(input_tensor)
                infer_request.start_async()
                infer_request.wait()
                output = infer_request.get_output_tensor()
                pred=torch.tensor(output.data)
                img_embedding.append(pred)
            img_embedding=torch.stack(img_embedding).to(device)
            return(img_embedding)

    def draw_text(self, img, boxes, txts, scores=None, text_score=0.5):
        
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
                img = cv2.polylines(img, [pts], is_closed, color, thickness)
                font_scale = 1.5
                text_thickness = 2
                text_org = (pts[0][0], pts[0][1])
                img = cv2.putText(img, txt, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, color, text_thickness, cv2.LINE_AA)
        # img = np.array(img, dtype="uint8")
        return img
    
    def get_closest_text_to_face(self, boxes, txts, scores=None, text_score=0.5):
        closest_text = None
        if boxes is not None:
            for idx, (box, txt) in enumerate(zip(boxes, txts)):
                if scores is not None and float(scores[idx]) < text_score:
                    continue
                pts = []
                for i in range(0, len(box)):
                    pts.append([box[i][0], box[i][1]])
                pts = np.array(pts, np.int32)
                # opencv coordinates notation
                min_x, min_y = np.min(pts[:,0]), np.min(pts[:, 1])
                max_x, max_y = np.max(pts[:,0]), np.max(pts[:, 1])

                # skip if detected text is above face, read text only below the bottom of the face
                if min_y < self.selected_face_bbox[3]:
                    continue
                if (abs(min_y - self.selected_face_bbox[3]) < NUM_PIXELS_BELOW_CHIN) and \
                    (self.selected_face_bbox[0] - NUM_PIXELS_LEFT_OF_FACE <= min_x <= self.selected_face_bbox[2]):
                    if self.name_start_time is None:
                        self.name_start_time = time.time()
                    self.get_logger().info('Name found')
                    closest_text = txt
        return closest_text


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
            source_image = self.plot_one_box(xyxy, source_image, mask=mask, label=label, color=colors(int(lbl)), line_thickness=1)
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
        if self.reg_complete:
            print("skipping as registration is complete")
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        width  = img.shape[1]   # float `width`
        height = img.shape[0]  # float `height`
        font_scale = (width + height) / 1500
        thickness = max(1, int(font_scale * 2))


        st_time = time.time()
        # faces, face_boxes=self.get_multiple_face(img[...,::-1])
        detections = self.detect(img[...,::-1], self.det_compiled_model)[0]
        faces, face_boxes, _ = self.get_face_det_results(detections, img, self.label_map)
        largest_face_size = 0
        selected_faces = None
        selected_face_bbox = None
        person_name = None
        if faces is None:
            pub_string = "no face found"
            msg = String()
            msg.data = pub_string
            self.text_publisher.publish(msg)
        if faces is not None:
            #start=time.time()
            if len(faces) > 1:
                #TODO (Jagadish): below needs to be voiced to the user
                #TODO (Jagadish): use the closest face (depth) not just the face size
                #TODO (Jagadish): Also use person detection to make sure face belongs to same person holding the name board
                self.get_logger().info('more than one face detected, using the biggest face detected')
            for face, fbox in zip(faces, face_boxes):
                curr_face_size = face.shape[0] * face.shape[1]
                if curr_face_size > largest_face_size:
                    largest_face_size = curr_face_size
                    selected_face = face
                    selected_face_bbox = fbox
            
            self.selected_face = selected_face
            self.selected_face_bbox = selected_face_bbox

            if not self.text_reading_complete:
                result, elapse_list = self.rapid_ocr(img)
                boxes = txts = scores = None
                if result is not None:
                    boxes, txts, scores = list(zip(*result))
                    person_name = self.get_closest_text_to_face(boxes, txts, scores)
                    self.draw_text(img, boxes, txts)
                else:
                        pub_string = "no text for name found"
                        msg = String()
                        msg.data = pub_string
                        self.text_publisher.publish(msg)
                if person_name is not None:
                    person_name = str(person_name).replace(" ", "").lower()
                    if person_name not in self.name_dict:
                        self.name_dict[person_name] = 1
                    else:
                        self.name_dict[person_name] += 1
                if self.name_start_time is not None:
                    if time.time() - self.name_start_time > NAME_READING_PERIOD:
                        self.text_reading_complete = True
                        self.reg_person_name = max(self.name_dict, key=self.name_dict.get)
                        self.get_logger().info('Name registered as : "%s"' % self.reg_person_name)
                        pub_string = "name registered as {}".format(self.reg_person_name)
                        msg = String()
                        msg.data = pub_string
                        self.text_publisher.publish(msg)
                        self.reg_per_dir = os.path.join(self.reg_dir, self.reg_person_name)
                        if not os.path.exists(self.reg_per_dir):
                            os.makedirs(self.reg_per_dir)
            elif not self.saving_complete:
                if self.save_start_time is None:
                    self.get_logger().info('Recording features')
                    self.save_start_time = time.time()
                    pub_string = "recording face"
                    msg = String()
                    msg.data = pub_string
                    self.text_publisher.publish(msg)
                if selected_face is None:
                    self.get_logger().info('No face found')
                    return

                if time.time() - self.save_start_time < RECORDING_TIME:
                    img_path = os.path.join(self.reg_per_dir,str(self.img_ctr)+".jpg")
                    # image_rgb = cv2.cvtColor(self.selected_face, cv2.COLOR_BGR2RGB) 
                    cv2.imwrite(img_path, self.selected_face)
                    self.img_ctr += 1
                else:
                    self.saving_complete = True
                    self.get_logger().info('Recording complete')
                    pub_string = "recording of face complete"
                    msg = String()
                    msg.data = pub_string
                    self.text_publisher.publish(msg)
        if self.saving_complete and not self.reg_complete:
            self.register_img(img_dir=os.path.join(self.reg_dir,self.reg_person_name),person_name=self.reg_person_name)
            self.get_logger().info('Registration complete for : "%s"' % self.reg_person_name)
            pub_string = "registration complete for {}".format(self.reg_person_name)
            msg = String()
            msg.data = pub_string
            self.text_publisher.publish(msg)
            self.reg_complete = True
        vis_img = self.draw_results(detections, img.copy(), self.label_map)
        msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
        self.vis_publisher_.publish(msg)

    def get_face(self, img):

        im=self.check_img(img).transpose(2,0,1)
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im/255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            pred = self.model(im, augment=False, visualize=False)
        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=1000)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                    #imc = im0.copy() if save_crop else im0  # for save_crop
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        #det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                        det[:, 0]=det[:, 0]*(img.shape[1]/im.shape[3])
                        det[:, 1]=det[:, 1]*(img.shape[0]/im.shape[2])
                        det[:, 2]=det[:, 2]*(img.shape[1]/im.shape[3])
                        det[:, 3]=det[:, 3]*(img.shape[0]/im.shape[2])
                        # Write results
                        max_box_area=0
                        max_face=None
                        for *xyxy, conf, cls in reversed(det):
                            x1,y1,x2,y2 = [int(ii) for ii in xyxy]
                            if (y2-y1)*(x2-x1)>max_box_area:
                                max_box_area=(y2-y1)*(x2-x1)
                                max_face=img[y1:y2,x1:x2,:]
        return([max_face])

    def get_face(self, img):

        im=self.check_img(img).transpose(2,0,1)
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im/255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            pred = self.model(im, augment=False, visualize=False)
        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=1000)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                    #imc = im0.copy() if save_crop else im0  # for save_crop
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        #det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                        det[:, 0]=det[:, 0]*(img.shape[1]/im.shape[3])
                        det[:, 1]=det[:, 1]*(img.shape[0]/im.shape[2])
                        det[:, 2]=det[:, 2]*(img.shape[1]/im.shape[3])
                        det[:, 3]=det[:, 3]*(img.shape[0]/im.shape[2])
                        # Write results
                        max_box_area=0
                        max_face=None
                        for *xyxy, conf, cls in reversed(det):
                            x1,y1,x2,y2 = [int(ii) for ii in xyxy]
                            if (y2-y1)*(x2-x1)>max_box_area:
                                max_box_area=(y2-y1)*(x2-x1)
                                max_face=img[y1:y2,x1:x2,:]
        return([max_face])

    def register_img(self,img_dir,person_name):

        if not os.path.exists(self.database_dir):
            os.mkdir(self.database_dir)
        features=[]
        for r,d,f in os.walk(img_dir):
            for file in f:
                if ".jp" in file or ".png" in file:
                    img=cv2.imread(os.path.join(r,file))
                    #img=check_img(img)
                    #face=get_face(Image.fromarray(img[...,::-1]))
                    # face=self.get_face(img[...,::-1])
                    # face = [img[...,::-1]]
                    face = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
                    feature=self.get_features(face,q=self.quantized)
                    features.append(feature)
        features = torch.mean(torch.stack(features),dim=0)
        features_colors=np.append(features.cpu().detach().numpy(),
                        (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))
        features_colors=features_colors.astype(np.float32)
        np.save(self.database_dir + '/{}'.format(person_name),features_colors)
        person_img=face[0]
        person_img=person_img[...,::-1]
        person_img=person_img.astype(np.uint8)
        Image.fromarray(
            person_img[...,::-1]).save(os.path.join(self.database_dir, person_name + ".jpg"))

def main(args=None):
    rclpy.init(args=args)

    face_registrer = PHINIXFaceRegistrer()

    rclpy.spin(face_registrer)
    text_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


