#!/usr/bin/env python3
import os
import rclpy
import queue
from rclpy.clock import Clock
from phinix_ui_message_juggler.ros2_topics import *
from datetime import datetime
from message_interface.msg import JugMsg
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory


class PHINIXJugglerSimulator(Node):

    def __init__(self):

        # Initializes node with name 'phinix_ui_message_juggler'
        super().__init__('phinix_ui_message_juggler')

        # Create the clock
        self.clock = Clock()

        # Creates ROS2 publisher. Args = message data type, topic to publish to, size of message queue
        self.general_publisher = self.create_publisher(JugMsg, TOPIC_MODULE_MESSAGES, 10)

        # Set timer to publish every X seconds
        self.timer_period = 0.3  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # Get shared package directory filepath
        pkg_share_dir = get_package_share_directory("phinix_ui_message_juggler")
        
        # Use shared package directory filepath to find text sample file
        text_sample_file_path = os.path.join(pkg_share_dir, 'text_samples/phinix_ui_delayed_text_sample.txt')
        
        # Open the text sample file for reading
        self.fp = open(text_sample_file_path, 'r')


    # FINISH: sorting messages based on their priority, check messages by timestamp, check output periphery by message type
    def timer_callback(self):
        
        # Flag used to set whether timestamps are on-the-fly or hardcoded
        hardcodedTimestamps = True

        # Flag used to determine whether to print in UNIX timestamp or human-readable timestamp
        timestamp_UNIX_output = False

        # Declare custom message type JugMsg
        msg = JugMsg()

        # Get current time
        current_time = self.get_clock().now()
        
        # Read from file line by line, store then in msg
        msg.string.data = self.fp.readline()
        
        # If reaching the end of file, loop from beginning of file
        if msg.string.data == "": # end of file
            self.fp.seek(0)
            msg.string.data = self.fp.readline()

        if (hardcodedTimestamps == False):
            
            # Separate message content, module ID and priority level
            message, module, priority = msg.string.data.split(',')
            
            # Store current time in message (seconds and nanoseconds)
            msg.header.stamp.sec = current_time.to_msg().sec
            msg.header.stamp.nanosec = current_time.to_msg().nanosec
        
        elif (hardcodedTimestamps == True):
            
            # Separate message content, module ID, priority level and timestamp
            message, module, priority, hardcodedTimestamp = msg.string.data.split(',')

            hardcodedSecs = int(hardcodedTimestamp)
            #hardcodedNanoSecs = int((int(hardcodedTimestamp) - int(hardcodedSecs)) * 1e9))

            # Store current time in message (seconds and nanoseconds)
            msg.header.stamp.sec = hardcodedSecs
            #msg.header.stamp.nanosec = hardcodedNanoSecs.to_msg().nanosec

        # Store message content
        msg.string.data = message

        # Store Module ID
        msg.int32.data = int(module)

        # Set priority of message from priority according to priority parameter
        if int(priority) == 1:
            msg.Bool.data = True
        elif int(priority) == 0:
            msg.Bool.data = False


        # Print to console as a regular timestamp
        if (timestamp_UNIX_output == True):

            # Combine seconds and nanoseconds
            combined_time = (current_time.to_msg().sec * 1e9) + current_time.to_msg().nanosec

            # Convert the timestamp to a Python datetime object
            current_datetime = datetime.fromtimestamp(combined_time / 1e9)

            # Format the datetime object as a string with fractional seconds
            formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Print to console as UNIX timestamp
        else:
            
            formatted_time = f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'

        # Print the current time
        #self.get_logger().info('Current time: %s' % formatted_time)
            
        # Set timestamp for messages
        #msg.header.stamp = self.get_clock().now().to_msg()
        
        # Publishes the message
        self.general_publisher.publish(msg)

        # Writes message to command line
        #self.get_logger().info('Publishing: "%s"' % msg.data)

        self.get_logger().info('Publishing : %s at ROS time: %s' % (msg.string.data, formatted_time))



def main(args=None):
    
    # Initialize ROS2 client library
    rclpy.init(args=args)

    # Create ROS2 node of TTS simulator
    juggler_simulator = PHINIXJugglerSimulator()

    # Stop executing main, loop for ROS2 events
    rclpy.spin(juggler_simulator)
    
    # Destory node then shutdown
    juggler_simulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




