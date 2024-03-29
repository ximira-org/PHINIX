#!/usr/bin/env python3
import os 
from time import time
import subprocess
import numpy as np

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool
import wave
from balacoon_tts import TTS
import sounddevice as sd
from ament_index_python.packages import get_package_share_directory
from message_interface.msg import JugMsg
from pydub import AudioSegment

#TOPIC_DUMMY_TEXTS = "/phinix/tts_simulator/dummy_texts"
TOPIC_VOICE = "/phinix_ui_message_juggler/voice_events"
SAMPLE_RATE_16K = 16000
SAMPLE_RATE_24K = 24000
LIGHT92_URL = "https://huggingface.co/balacoon/tts/resolve/main/en_us_hifi92_light_cpu.addon"
LIGHT_HIFI_URL = "https://huggingface.co/balacoon/tts/resolve/main/en_us_hifi_jets_cpu.addon"
LIGHT_CMUARTIC_URL = "https://huggingface.co/balacoon/tts/resolve/main/en_us_cmuartic_jets_cpu.addon"
TOPIC_TTS_AVAILABLE = "/phinix/tts_simulator/tts_available"

class PHINIXTTSBalacoon(Node):

    def __init__(self):
        super().__init__('phinix_tts_balacoon')
        self.declare_parameter('model', rclpy.Parameter.Type.STRING)
        self.model = self.get_parameter('model').value
        self.get_logger().info('Model : "%s"' % self.model)
        self.subscriber_ = self.create_subscription(
            JugMsg,
            TOPIC_VOICE,
            self.listener_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.available = True
        self.availability_publisher = self.create_publisher(Bool, TOPIC_TTS_AVAILABLE, 10)

        pkg_share_dir = get_package_share_directory("phinix_tts_balacoon")
        model_dir = os.path.join(pkg_share_dir, "models")
        self.get_logger().info('Model dir : "%s"' % model_dir)
        # create models dir if it doesn't exist
        if not os.path.exists(os.path.join(model_dir)):
            os.mkdir(model_dir)
        if self.model == "super_light92":
            model_file_path = os.path.join(pkg_share_dir, "models/en_us_hifi92_light_cpu.addon")
            if not os.path.exists(model_file_path):
                self.get_logger().info("Super Light 92 model not found. So dowloading...")
                subprocess.run(['wget', '-O', model_file_path, LIGHT92_URL], check=True)
            self.tts = TTS(model_file_path)
            self.sample_rate = SAMPLE_RATE_16K
        elif self.model == "light_hifi":
            model_file_path = os.path.join(pkg_share_dir, "models/en_us_hifi_jets_cpu.addon")
            if not os.path.exists(model_file_path):
                self.get_logger().info("Light Hifi model not found. So dowloading...")
                subprocess.run(['wget', '-O', model_file_path, LIGHT_HIFI_URL], check=True)
            self.tts =TTS(model_file_path)
            self.sample_rate = SAMPLE_RATE_24K
        else:
            model_file_path = os.path.join(pkg_share_dir, "models/en_us_cmuartic_jets_cpu.addon")
            if not os.path.exists(model_file_path):
                self.get_logger().info("Light cmuartic model not found. So dowloading...")
                subprocess.run(['wget', '-O', model_file_path, LIGHT_CMUARTIC_URL], check=True)
            self.tts =TTS(model_file_path)
            self.sample_rate = SAMPLE_RATE_24K

    def listener_callback(self, msg):
        if self.available == False:
            self.get_logger().info("WARNING: Ignoring:{} because sound is playing".format(msg.string.data))
            return
        
        supported_speakers = self.tts.get_speakers()
        speaker = supported_speakers[-1]
        # finally run synthesis
        start_time = time()
        message_string = msg.string.data
        samples = self.tts.synthesize(message_string, speaker)
        end_time = time()
        print("[speaker : {}] audio generation time : {}".format(speaker, (end_time - start_time)))

        # Convert samples to 32-bit floats for better precision
        samples_float = np.array(samples, dtype='f')

        # Scale down the volume by a factor of 0.8 using 32-bit floats
        pan = msg.panning.data
        #lerp the left volume where -1 = 1 and 1 = 0
        pan_left_float = (1.0 - pan) / 2.0
        pan_right_float = (1.0 + pan) / 2.0


        samples_left_float = samples_float * pan_left_float
        samples_right_float = samples_float * pan_right_float

        # Convert the scaled samples back to 16-bit integers
        samples_left_int = samples_left_float.astype('h')
        samples_right_int = samples_right_float.astype('h')

        # Combine the left and right channels into a stereo signal
        panned_samples = np.column_stack((samples_left_int, samples_right_int))

        # Calculate duration
        duration = len(samples) / self.sample_rate

        self.set_available(False)
        self.timer = self.create_timer(duration, self.playback_complete_callback) 

        sd.play(panned_samples, samplerate=self.sample_rate, mapping=[1, 2], blocking=False)

        
    def playback_complete_callback(self):
        self.set_available(True)
        self.timer.cancel()
    
    def set_available(self, available):
        self.available = available
        msg = Bool()
        msg.data = available
        self.availability_publisher.publish(msg)


        
def main(args=None):
    rclpy.init(args=args)
    tts_balacoon = PHINIXTTSBalacoon()
    rclpy.spin(tts_balacoon)
    tts_balacoon.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

