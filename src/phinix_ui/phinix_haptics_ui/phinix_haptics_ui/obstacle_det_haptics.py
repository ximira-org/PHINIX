import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from bluepy.btle import Peripheral, DefaultDelegate, ADDR_TYPE_RANDOM
import time
import numpy as np
from std_msgs.msg import Int32MultiArray
from phinix_haptics_ui.phinix_haptics_obstacle_detection import obstacle_detection
from phinix_haptics_ui.phinix_haptics_path_detection import path_detection
from phinix_haptics_ui.phinix_haptics_cardinal_direction import cardinal_direction
from phinix_haptics_ui.phinix_haptics_battery_level import battery_level
from phinix_haptics_ui.phinix_haptics_confirmation import confirmation_buzz
from phinix_haptics_ui.phinix_haptics_face_recognition import face_recognition

CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Bluetooth UART service UUID
DEVICE_NAME = "Ximira_Phinix_Bracelet"  # Replace with your device's name
DEVICE_ADDRESS = "D7:FD:3F:8F:E2:72"  # Replace with your device's name
TOPIC_OBSTACLE_DETS = "/phinix/obstacle_detector/detections"

class HapticsObstacleDet(Node):

    def __init__(self):
        super().__init__('haptics_obstacle_det')
        self.haptics_publisher = self.create_publisher(String, '/phinix/haptics', 10)
        self.device_address = read_mac_address_from_file(self)
        self.obs_det_sub = self.create_subscription(Int32MultiArray, TOPIC_OBSTACLE_DETS, self.obs_det_callback, 10)
        self.commands = [
            "!o111000000",
            "!o000010000",
            "!o000000011",
            "!p10",
            "!p01",
            "!p22",
            "!p33",
            "!cNN",
            "!cSW",
            "!cEE",
            "!b3",
            "!b1",
            "!y",
            "!f0",
            "!f2",
            "!f4",
        ]
        self.command_index = 0
        self.haptic_band = Peripheral()
        self.haptic_band.connect(DEVICE_ADDRESS, ADDR_TYPE_RANDOM)
        self.characteristic = self.haptic_band.getCharacteristics(uuid=CHARACTERISTIC_UUID)[0]

        self.get_logger().info(f"Connected to device: {self.device_address}")

    def obs_det_callback(self, obs_det_msg: Int32MultiArray):
        obs_det = np.array(obs_det_msg.data)
        if np.sum(obs_det) >= 1:
            self.get_logger().info(f"ostacle det : {obs_det}")
            self.get_logger().info(f"ostacle det0 : {obs_det[0]}")
            self.get_logger().info(f"ostacle det1 : {obs_det[1]}")
            self.get_logger().info(f"ostacle det1 : {np.array(obs_det)}")

            current_command = self.commands[0]
            command_list = []
            if current_command[1] == 'o':
                self.get_logger().info(f"command invokes obstacle detection")
                command_list = obstacle_detection(current_command, obs_det)
            for command in command_list:
                # print(command)
                self.characteristic.write(bytearray(command, "utf-8"))
                self.get_logger().info(f"Sent command: {command} to device")
            time.sleep(3)

def read_mac_address_from_file(self):
        # Change this path to the file containing the MAC address
        file_path = "settings/mac_address_file.txt"

        with open(file_path, 'r') as file:
            mac_address = file.readline().strip()  # Assuming MAC address is on the first line
            return mac_address

def main(args=None):
    rclpy.init(args=args)
    haptics_node = HapticsObstacleDet()
    rclpy.spin(haptics_node)
    haptics_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 