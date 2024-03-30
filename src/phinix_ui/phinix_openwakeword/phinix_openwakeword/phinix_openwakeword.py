#!/usr/bin/env python3
import os 
from glob import glob
import time 

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from ament_index_python.packages import get_package_share_directory

import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
from playsound import playsound

AWAKE_PERIOD = 7 # seconds
NODE_NAME="phinix_openwakeword"

TOPIC_WAKEWORD = "/phinix/wakeword"
TOPIC_NODE_STATES = "/phinix/node_states"

OBSTACLE_DETECTION_NODE_INDEX = 0
PATH_DETECTION_NODE_INDEX = 1
TEXT_DETECTION_NODE_INDEX = 2
FACE_DETECTION_NODE_INDEX = 3
FACE_REGESTRATION_NODE_INDEX = 4
OBJECT_DETECTION_NODE_INDEX = 5

WAKEWORD_CONFIDENT_THRESHOLD = 0.999

class PHINIXWW(Node):

    def __init__(self):
        super().__init__(NODE_NAME)
        self.timer_period = 0.02  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.wakeword_publisher = self.create_publisher(String, TOPIC_WAKEWORD, 10)
        self.node_state_subscriber = self.create_subscription(
            Int32MultiArray, 
            TOPIC_NODE_STATES, 
            self.node_state_callback, 
            10) 
        self.object_detection_active = False
        self.text_detection_active = False
        self.face_recognition_active = False
        self.path_detection_active = False
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1280
        self.audio = pyaudio.PyAudio()
        self.mic_stream = self.audio.open(format=self.format, 
                            channels=self.channels, 
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=self.chunk)
        self.phinix_awake = False
        self.awake_start_time = None
        self.wakeword_count = 0
        self.still_wait = True
        self.command_list = [
            "ok_phinix",
            "ok_mira",
            "describe_surroundings",
            "read_to_me",
            #"identify_people",
            #"register_face",
            "stop_describing",
            "stop_reading",
            #"stop_identifying",
            #"enable_path_detection",
            #"disable_path_detection",
            #"set_ground_level"
        ]
        self.commands = {}
        self.wakeword_spotted_dict = {}
        self.scores_window = {}
        self.scores_last = {}
        self.modules_dict = {}
        self.pkg_share_dir = get_package_share_directory(NODE_NAME)
        for command in self.command_list:
            self.wakeword_spotted_dict[command] = False
            self.scores_window[command] = []
            self.scores_last[command] = 0
            #does this line work?
            self.commands[command] = os.path.join(self.pkg_share_dir, "model", command+".onnx")

        # Load pre-trained openwakeword models
        self.oww_activation_model = Model(
            wakeword_model_paths= [
                self.commands["ok_mira"], 
                self.commands["ok_phinix"]
            ],
            # wakeword_model_paths= [ok_phinix],
            enable_speex_noise_suppression=True,
            inference_framework="onnx",
            vad_threshold=0.5)
        
        # Load pre-trained openwakeword models
        self.oww_module_models = Model(
            wakeword_model_paths= [
                self.commands["describe_surroundings"],
                self.commands["read_to_me"],
                #self.commands["identify_people"],
                #self.commands["register_face"],
                self.commands["stop_describing"],
                #self.commands["stop_identifying"],
                self.commands["stop_reading"],
                #self.commands["enable_path_detection"],
                #self.commands["disable_path_detection"],
                #self.commands["set_ground_level"]
            ],
            enable_speex_noise_suppression=True,
            inference_framework="onnx",
            vad_threshold=0.5)

        self.ignore_until_time = time.time()
    
    def reset_scores_window(self):
        for command in self.command_list:
            self.scores_window[command] = []

    def reset_scores_last(self):
        for command in self.command_list:
            self.scores_last[command] = 0

    def clear_model_buffer(self):
        for mdl in self.oww_activation_model.prediction_buffer.keys():
            self.oww_activation_model.prediction_buffer[mdl].clear()
        for mdl in self.oww_module_models.prediction_buffer.keys():
            self.oww_module_models.prediction_buffer[mdl].clear()
            
    def timer_callback(self):
        audio_clip = np.frombuffer(self.mic_stream.read(self.chunk), dtype=np.int16)

        # Feed to openWakeWord model
        n_spaces = 22
        # Column titles
        
        if time.time() < self.ignore_until_time:
            return
        
        self.oww_module_models.predict(audio_clip)
        self.oww_activation_model.predict(audio_clip)

        if not self.phinix_awake:
            
            for mdl in self.oww_activation_model.prediction_buffer.keys():
                # Add scores in formatted table
                scores = list(self.oww_activation_model.prediction_buffer[mdl])
                if scores[-1] >= WAKEWORD_CONFIDENT_THRESHOLD and mdl == "ok_phinix":
                    self.get_logger().info("wake word : " + mdl)
                    self.phinix_awake = True
                    self.awake_start_time = time.time()
                    self.reset_scores_window()
                    self.reset_scores_last()
                    self.clear_model_buffer()
                    playsound(os.path.join(self.pkg_share_dir, "audio_clips", "start_listening.wav"), block=False)
        else:
            if (time.time() - self.awake_start_time) > AWAKE_PERIOD:
                self.awake_start_time = None
                self.phinix_awake = False
                self.reset_scores_window()
                self.reset_scores_last()
                self.clear_model_buffer()
                return
            
            higest_score = 0
            target_mdl = None
            for mdl in self.oww_module_models.prediction_buffer.keys():
                # Add scores in formatted table
                scores = list(self.oww_module_models.prediction_buffer[mdl])
                self.scores_last[mdl] = scores[-1]
                if scores[-1] > higest_score and scores[-1] >= WAKEWORD_CONFIDENT_THRESHOLD:
                    higest_score = scores[-1]
                    target_mdl = mdl
            
            if target_mdl is not None:
                
                self.get_logger().info("wake word : " + target_mdl)
                if not self.wakeword_is_valid(target_mdl):
                    return
                playsound(os.path.join(self.pkg_share_dir, 
                        "audio_clips", target_mdl+".mp3"), block=False)
                self.phinix_awake = False
                msg = String()
                msg.data = str(target_mdl)
                self.wakeword_publisher.publish(msg)
                self.reset_scores_window()
                self.reset_scores_last()
                self.clear_model_buffer()
                self.ignore_until_time = time.time() + 1
    
    def wakeword_is_valid(self, wword): 
        if wword == "register_face" and self.face_recognition_active:
            return False
        if wword == "read_to_me" and self.text_detection_active:
            return False
        if wword == "stop_reading" and not self.text_detection_active:
            return False
        if wword == "identify_people" and self.face_recognition_active:
            return False
        if wword == "stop_identifying" and not self.face_recognition_active:
            return False
        if wword == "describe_surroundings" and self.object_detection_active:
            return False
        if wword == "stop_describing" and not self.object_detection_active:
            return False
        if wword == "enable_path_detection" and self.path_detection_active:
            return False
        if wword == "disable_path_detection" and not self.path_detection_active:
            return False
        return True
    
    #Set node_active to true if node manager so
    def node_state_callback(self, node_states: Int32MultiArray):
        self.path_detection_active = node_states.data[PATH_DETECTION_NODE_INDEX] == 1
        self.text_detection_active = node_states.data[TEXT_DETECTION_NODE_INDEX] == 1
        self.face_recognition_active = node_states.data[FACE_REGESTRATION_NODE_INDEX] == 1
        self.object_detection_active = node_states.data[OBJECT_DETECTION_NODE_INDEX] == 1
        
                
def main(args=None):
    rclpy.init(args=args)

    phinix_ww = PHINIXWW()

    rclpy.spin(phinix_ww)
    tts_simulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()