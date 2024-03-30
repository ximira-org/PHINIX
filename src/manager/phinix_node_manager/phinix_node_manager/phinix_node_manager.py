import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
import time

TOPIC_WAKE_WORD = "/phinix/wakeword"
TOPIC_NODE_STATES = "/phinix/node_states"

NUM_OF_NODES = 6

LOOP_TIME = 1
READ_TO_ME_TIME = 30
IDENTIFY_PEOPLE_TIME = 10
DESCRIBE_SURROUNDINGS_TIME = 300

OBSTACLE_DETECTION_NODE_INDEX = 0
PATH_DETECTION_NODE_INDEX = 1
TEXT_DETECTION_NODE_INDEX = 2
FACE_DETECTION_NODE_INDEX = 3
FACE_REGESTRATION_NODE_INDEX = 4
OBJECT_DETECTION_NODE_INDEX = 5

class PHINIXNodeManager(Node):

    def __init__(self):
        super().__init__('phinix_node_manager')
        self.get_logger().info(f"init node manager")

        #self.declare_parameter('topic_wakeword', rclpy.Parameter.Type.STRING)
        #self.topic_wakeword = self.get_parameter('topic_wakeword').value

        #self.get_logger().info(f"topic wakeword: {self.topic_wakeword}")

        self.wake_word_sub = self.create_subscription(String, TOPIC_WAKE_WORD, self.wakeword_callback, 10)

        self.state = Int32MultiArray()
        self.state.data = [0] * NUM_OF_NODES
        self.state_publisher = self.create_publisher(Int32MultiArray, TOPIC_NODE_STATES, 10)

        # when to turn off nodes I control
        # 0 = Obstacle detection
        # 1 = Pathway detection
        # 2 = Text recognition
        # 3 = Face recognition
        # 4 = Face regestration
        # 5 = Describe surroundings

        self.off_times = [0] * NUM_OF_NODES

        #Set obstacle and path detection to always be on
        self.off_times[0] = -1

        #timer for turning off nodes after time has past and broadcasting state
        self.timer = self.create_timer(LOOP_TIME, self.timer_callback)

        #Only one of these nodes can be on at a time
        self.tts_description_nodes = [TEXT_DETECTION_NODE_INDEX, FACE_DETECTION_NODE_INDEX, OBJECT_DETECTION_NODE_INDEX]

    
    def wakeword_callback(self, wakeword: String):
        word = wakeword.data
        self.get_logger().info(word)

        if word == "read_to_me":
            self.disable_all_description_nodes()
            self.off_times[TEXT_DETECTION_NODE_INDEX] = time.time() + READ_TO_ME_TIME
        elif word == "stop_reading":
            self.off_times[TEXT_DETECTION_NODE_INDEX] = time.time()
        elif word == "identify_people":
            self.disable_all_description_nodes()
            self.off_times[FACE_DETECTION_NODE_INDEX] = time.time() + IDENTIFY_PEOPLE_TIME
        elif word == "stop_identifying":
            self.off_times[FACE_DETECTION_NODE_INDEX] = time.time()
        elif word == "describe_surroundings":
            self.disable_all_description_nodes()
            self.off_times[OBJECT_DETECTION_NODE_INDEX] = time.time() + DESCRIBE_SURROUNDINGS_TIME
        elif word == "stop_describing":
            self.off_times[OBJECT_DETECTION_NODE_INDEX] = time.time()
        elif word == "enable_path_detection":
            self.off_times[PATH_DETECTION_NODE_INDEX] = -1
        elif word == "disable_path_detection":
            self.off_times[PATH_DETECTION_NODE_INDEX] = time.time()
        elif word == "register_face":
            self.off_times[FACE_REGESTRATION_NODE_INDEX] = -1
    
    def disable_all_description_nodes(self):
        for node in self.tts_description_nodes:
            self.off_times[node] = time.time()


    def timer_callback(self):
        current_time = time.time()
        i = 0
        while i < NUM_OF_NODES:
            #Set the node value to true if the time is less than the time it should turn off
            self.state.data[i] = current_time <= self.off_times[i]
            #if the off time is -1, that means the node should always be on
            if self.off_times[i] == -1:
                self.state.data[i] = 1
            i += 1
        #self.get_logger().info(str(self.state.data))
        self.state_publisher.publish(self.state)
        

def main(args=None):
    rclpy.init(args=args)
    node = PHINIXNodeManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
