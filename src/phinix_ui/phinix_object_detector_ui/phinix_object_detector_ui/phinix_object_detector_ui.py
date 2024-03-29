import os
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from message_interface.msg import JugMsg

from phinix_perception_msgs.msg import BBoxMsg

TOPIC_OBJ_DET_BBOX = "/phinix/module/object_det/bbox"
TOPIC_NODE_STATES = "/phinix/node_states"
JUG_MESSAGES = "/phinix_ui_message_juggler/module_messages"

object_detection_node_state_index = 5

frame_width = 640

#differnece from the started detected distance before we consider it a new object
DISTANCE_DIFFERENCE_THRESHOLD = 0.5
#difference in position before we consider it a new object
POSITION_DIFFERENCE_THRESHOLD = 100
#number of times I need to read the same object before I publish a message
READ_BEFORE_PUBLISH_COUNT = 3

def is_new_position(position, positions):
    for pos in positions:
        if abs(pos - position) < POSITION_DIFFERENCE_THRESHOLD:
            return False
    return True

def is_new_distance(distance, distances):
    for dist in distances:
        if abs(dist - distance) < DISTANCE_DIFFERENCE_THRESHOLD:
            return False
    return True

class PHINIXObjectDetectorUI(Node):

    def __init__(self):
        super().__init__('phinix_object_detector_ui')
        self.get_logger().info("Begin Object Detector UI")

        #subscribe to object detection bounding box
        self.obj_det_bbox_sub = self.create_subscription(BBoxMsg, TOPIC_OBJ_DET_BBOX, self.obj_det_bbox_callback, 10)

        self.node_state_subscriber = self.create_subscription(
            Int32MultiArray, 
            TOPIC_NODE_STATES, 
            self.node_state_callback, 
            10) 
    
        self.publisher = self.create_publisher(JugMsg, JUG_MESSAGES, 10)

        #am i currently detecting objects?
        self.currently_detecting_objects = False
        #objects detected duing the current detection cycle
        self.detected_object_classes = []
        #distances at which those obejects were detected
        self.detected_object_distances = []
        #positions of the objects in the detected_objects array
        self.detected_object_positions = []
        #number of times the object at that distance has been detected
        self.detected_object_counts = []


    def obj_det_bbox_callback(self, msg):
        #loop through the detected objects
        for i, class_name in enumerate(msg.classes):
            #have I seen this object before?
            already_detected = False
            #enumerate through the detected objects
            for j, detected_object in enumerate(self.detected_object_classes):
                #if the class name is the same as the detected object
                if class_name.data == detected_object:
                    #if the position and distance are the same
                    if abs(msg.top_left_x_ys[i].x - self.detected_object_positions[j]) < POSITION_DIFFERENCE_THRESHOLD and abs(msg.depths[i] - self.detected_object_distances[j]) < DISTANCE_DIFFERENCE_THRESHOLD:
                        #increment the count
                        self.detected_object_counts[j] += 1
                        already_detected = True
                        #if the object has been detected enough times to send a message
                        if self.detected_object_counts[j] == READ_BEFORE_PUBLISH_COUNT:
                            #create a new jug message
                            output = class_name.data
                            depth = msg.depths[i]
                            #calculate the panning value
                            pan_value = 0.0
                            left_x = msg.top_left_x_ys[i].x
                            right_x = msg.bottom_right_x_ys[i].x
                            center_x = (left_x + right_x) / 2
                            pan_value = (center_x - (frame_width / 2)) / (frame_width / 2)
                            #format depth to 2 decimal places
                            output += " " + "{:.2f}".format(depth) + " meters"
                            self.send_message_to_ui_juggler(output, pan_value)
                        break
            #if the object has not been detected before
            if not already_detected:
                #add the object to the detected objects array
                self.detected_object_classes.append(class_name.data)
                self.detected_object_positions.append(msg.top_left_x_ys[i].x)
                self.detected_object_distances.append(msg.depths[i])
                self.detected_object_counts.append(1)
    
    #Set node_active to true if node manager so
    def node_state_callback(self, node_states: Int32MultiArray):
        self.object_recognition_active = node_states.data[object_detection_node_state_index] == 1
        #self.get_logger().info(f"object_recognition_active = {self.object_recognition_active}")
        if not self.currently_detecting_objects and self.object_recognition_active:
            self.currently_detecting_objects = True
            #reset the detected objects array
            self.detected_objects = []
        elif self.currently_detecting_objects and not self.object_recognition_active:
            self.currently_detecting_objects = False
    
    def send_message_to_ui_juggler(self, message_string, pan_value):
        print(message_string)

        # Declare custom message type JugMsg
        msg = JugMsg()
        
        msg.string.data = str(message_string)

        # Store Module ID
        msg.int32.data = int(2)

        #secondary message priority
        msg.Bool.data = False

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
    node = PHINIXObjectDetectorUI()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
