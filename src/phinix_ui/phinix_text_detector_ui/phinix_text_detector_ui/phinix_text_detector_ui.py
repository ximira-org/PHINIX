import os
import rclpy
from rclpy.node import Node

from message_interface.msg import JugMsg
from phinix_perception_msgs.msg import BBoxMsg

TOPIC_TEXT_REC_BBOX = "/phinix/module/text_rec/bbox"
JUG_MESSAGES = "/phinix_ui_message_juggler/module_messages"

#Number of times I need to read the same text befor eI publish a message
READ_BEFORE_PUBLISH_COUNT = 3

frame_width = 640

class PHINIXTextDetectorUI(Node):

    def __init__(self):
        super().__init__('phinix_text_detector_ui')
        self.get_logger().info("Begin Text Detector UI")

        self.bbox_subscriber = self.create_subscription(BBoxMsg, TOPIC_TEXT_REC_BBOX, self.bbox_callback, 10)

        self.tts_publisher = self.create_publisher(JugMsg, JUG_MESSAGES, 10)

        #Text I have already read this session
        self.text_that_has_been_read = []
        self.text_counts = []
    
    def bbox_callback(self, msg):
        for i in range(len(msg.texts)):
            #if I have not yet seen this text 
            if not msg.texts[i].data in self.text_that_has_been_read:
                self.text_that_has_been_read.append(msg.texts[i].data)
                self.text_counts.append(1)
                continue
            else:
                self.text_counts[self.text_that_has_been_read.index(msg.texts[i].data)] += 1
            
            if not self.text_counts[self.text_that_has_been_read.index(msg.texts[i].data)] == READ_BEFORE_PUBLISH_COUNT:
                continue
            output = msg.texts[i].data
            depth = msg.depths[i]

            #calculate the panning value
            pan_value = 0.0
            left_x = msg.top_left_x_ys[0].x
            right_x = msg.bottom_right_x_ys[0].x
            center_x = (left_x + right_x) / 2
            pan_value = (center_x - (frame_width / 2)) / (frame_width / 2)
            
            #add the depth to the output if it is far away
            if depth > 1.0:
                output += " " + "{:.2f}".format(depth) + " meters"
            
            self.send_message_to_ui_juggler(output, pan_value)

            

    def send_message_to_ui_juggler(self, message_string, pan_value):
        print(message_string)

        # Declare custom message type JugMsg
        msg = JugMsg()
        
        msg.string.data = str(message_string)

        # Store Module ID
        msg.int32.data = int(2)

        #Secondary message priority
        msg.Bool.data = False

        msg.panning.data = pan_value

        # Get current time
        current_time = self.get_clock().now()
        
        # Store current time in message (seconds and nanoseconds)
        msg.header.stamp.sec = current_time.to_msg().sec
        msg.header.stamp.nanosec = current_time.to_msg().nanosec

        formatted_time = f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
        
        # Publishes the message
        self.tts_publisher.publish(msg)

        self.get_logger().info('Publishing : %s at ROS time: %s' % (msg.string.data, formatted_time))


def main(args=None):
    rclpy.init(args=args)
    node = PHINIXTextDetectorUI()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
