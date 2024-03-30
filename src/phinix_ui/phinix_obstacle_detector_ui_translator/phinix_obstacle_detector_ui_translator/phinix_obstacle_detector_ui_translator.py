import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from bluepy.btle import Peripheral, DefaultDelegate, ADDR_TYPE_RANDOM
import time
from std_msgs.msg import Int32MultiArray
from enum import Enum
from message_interface.msg import JugMsg

TOPIC_OBSTACLE_DET_UI_EVENTS = "/phinix_ui_message_juggler/module_messages"
TOPIC_OBSTACLE_DETS = "/phinix/obstacle_detector/detections"
NUM_OBS_DET_CHANNELS = 6
# the amount of time a cell needs to be unocupied to be considered off.
# prevents sounds playing repeatedly if something is flickering a bit.
OBS_DET_OFF_SECONDS = .5
OBS_DET_ON_SECONDS = .25

MSGS = ["High Left", "Low Left", 
        "High Center", "Low Center",
        "High Right", "Low Right"]

PANNING_VALUES = [-1.0, -1.0, 
                  0.0, 0.0, 
                  1.0, 1.0]

class OBS_DET_CHANNEL_STATE(Enum):
    ON = 1, 
    OFF = 2, 
    TURNING_ON = 3,
    TURNING_OFF = 4

class PhinixObstacleDetectorUiTranslator(Node):

    def __init__(self):
        super().__init__('phinix_obstacle_detector_ui_translator')

        print("Init PhinixObstacleDetectorUiTranslator")

        #Subscribe to obs det topic
        self.obs_det_sub = self.create_subscription(Int32MultiArray, TOPIC_OBSTACLE_DETS, self.obs_det_callback, 10)

        self.current_obs_det_state = [0] * NUM_OBS_DET_CHANNELS
        self.prev_obs_det_ui_state = [0] * NUM_OBS_DET_CHANNELS

        self.obs_det_channel_states = [OBS_DET_CHANNEL_STATE.OFF] * NUM_OBS_DET_CHANNELS
        self.obs_det_off_times = [0] * NUM_OBS_DET_CHANNELS
        self.obs_det_on_times = [0] * NUM_OBS_DET_CHANNELS

        # Creates ROS2 publisher. Args = message data type, topic to publish to, size of message queue
        self.publisher = self.create_publisher(JugMsg, TOPIC_OBSTACLE_DET_UI_EVENTS, 10)
    
    #when we get new obs det data
    def obs_det_callback(self, obs_det_msg: Int32MultiArray):
        #self.get_logger().info(obs_det_msg.data)
        i = 0
        while i < NUM_OBS_DET_CHANNELS:
            sound_on = obs_det_msg.data[i] == 1

            if self.obs_det_channel_states[i] == OBS_DET_CHANNEL_STATE.ON:
                if sound_on == False:
                    self.obs_det_channel_states[i] = OBS_DET_CHANNEL_STATE.TURNING_OFF
                    self.obs_det_off_times[i] = time.time() + OBS_DET_OFF_SECONDS
            elif self.obs_det_channel_states[i] == OBS_DET_CHANNEL_STATE.TURNING_OFF:
                if sound_on == True:
                    self.obs_det_channel_states[i] = OBS_DET_CHANNEL_STATE.ON
                elif time.time() >= self.obs_det_off_times[i]:
                    self.obs_det_channel_states[i] = OBS_DET_CHANNEL_STATE.OFF
            elif self.obs_det_channel_states[i] == OBS_DET_CHANNEL_STATE.OFF:
                if sound_on == True:
                    self.obs_det_channel_states[i] = OBS_DET_CHANNEL_STATE.TURNING_ON
                    self.obs_det_on_times[i] = time.time() + OBS_DET_ON_SECONDS
            elif self.obs_det_channel_states[i] == OBS_DET_CHANNEL_STATE.TURNING_ON:
                if sound_on == False:
                    self.obs_det_channel_states[i] = OBS_DET_CHANNEL_STATE.OFF
                elif time.time() >= self.obs_det_on_times[i]:
                    self.obs_det_channel_states[i] = OBS_DET_CHANNEL_STATE.ON
                    self.send_message_to_ui_juggler(MSGS[i], PANNING_VALUES[i])

            i += 1
    
    def send_message_to_ui_juggler(self, message_string, pan_value):
        print(message_string)

        # Declare custom message type JugMsg
        msg = JugMsg()
        
        msg.string.data = str(message_string)

        # Store Module ID
        msg.int32.data = int(2)

        msg.Bool.data = True

        msg.panning.data = pan_value

        # Get current time
        current_time = self.get_clock().now()
        
        # Store current time in message (seconds and nanoseconds)
        msg.header.stamp.sec = current_time.to_msg().sec
        msg.header.stamp.nanosec = current_time.to_msg().nanosec

        formatted_time = f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
        
        # Publishes the message
        self.publisher.publish(msg)

        self.get_logger().info('Publishing : %s at ROS time: %s' % (msg.string.data, formatted_time))

def main(args=None):
    rclpy.init(args=args)
    node = PhinixObstacleDetectorUiTranslator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
