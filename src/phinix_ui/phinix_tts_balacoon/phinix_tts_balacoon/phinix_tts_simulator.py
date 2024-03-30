#!/usr/bin/env python3
import os 

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

TOPIC_DUMMY_TEXTS = "/phinix/tts_simulator/dummy_texts"
class PHINIXTTSSimulator(Node):

    def __init__(self):
        super().__init__('phinix_tts_balacoon')
        self.text_publisher_ = self.create_publisher(String, TOPIC_DUMMY_TEXTS, 10)

        self.timer_period = 0.5  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        pkg_share_dir = get_package_share_directory("phinix_tts_balacoon")
        text_sample_file_path = os.path.join(pkg_share_dir, 'text_samples/tts_text_samples.txt')
        self.fp = open(text_sample_file_path, 'r')
        self.idx = 0

    def timer_callback(self):
        msg = String()
        msg.data = self.fp.readline()

        if msg.data == "": #end of file
            self.fp.seek(0)
            msg.data = self.fp.readline()
            
        self.text_publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        
def main(args=None):
    rclpy.init(args=args)

    tts_simulator = PHINIXTTSSimulator()

    rclpy.spin(tts_simulator)
    tts_simulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


