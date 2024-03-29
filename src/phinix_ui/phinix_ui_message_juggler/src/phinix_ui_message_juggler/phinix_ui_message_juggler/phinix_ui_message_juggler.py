#!/usr/bin/env python3
import os
import rclpy
import threading
import time
from rclpy.clock import Clock
from phinix_ui_message_juggler.ros2_topics import *
from message_interface.msg import JugMsg
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool
from ament_index_python.packages import get_package_share_directory

TOPIC_TTS_AVAILABLE = "/phinix/tts_simulator/tts_available"
TOPIC_WAKE_WORD = "/phinix/wakeword"

user_peripheral_output_preferences = {
        
    # Possible Combinations: 
    # H = Haptics, V = Voice, S = Sound
    # HVS, HV, HS, SV, H, V, S

    # 1: Pathway Recognition
    1: "HVS",

    # 2: Obstacle Avoidance
    2: "HVS",

    # 3: Battery Level
    3: "VS",

    # 4: Recognized Individual
    4: "V",

    # 5: User Input
    5: "HVS",

    }

left_tts_msgs = ["High Left", "Low Left"]
center_tts_msgs = ["High Center", "Low Center"]
right_tts_msgs = ["High Right", "Low Right"]
tts_mgs_sets = [left_tts_msgs, center_tts_msgs, right_tts_msgs]

class PHINIXJuggler(Node):

    # Todo features: Message, peripheral output type, priority status, timestamp

    # Custom message type for each periphery
    # Haptics - See Ben's API
    # Voice - Simply reads from a string (with interruptions for priority messages)
    # Sound - Integer representing the sound to be played


    def __init__(self):

        # Initializes node with name 'phinix_ui_message_juggler'
        super().__init__('phinix_ui_message_juggler')
        
        # Create the clock
        self.clock = Clock()

        # Queues to store priority and secondary messages
        self.tts_priority_messages_list = []
        self.tts_secondary_messages_list = []

        # Create a subscriber for the Juggler to receive messages from AI modules
        self.module_message_listener = self.create_subscription(JugMsg, TOPIC_MODULE_MESSAGES, self.listener_callback, 10)

        #Listen for if TTS is available
        self.tts_available_listener = self.create_subscription(Bool, TOPIC_TTS_AVAILABLE, self.tts_available_callback, 10)

        #Listen for wakewords. stop wakeowrds can clear the message queue
        self.wake_word_sub = self.create_subscription(String, TOPIC_WAKE_WORD, self.wakeword_callback, 10)

        # Creating ROS2 publishers. Args = message data type, topic to publish to, size of message queue
        
        # Create ROS2 publisher for publishing messages to all peripherals simultaneougly
        self.general_publisher = self.create_publisher(JugMsg, TOPIC_JUGGLER_MESSAGES, 10)

        # Create ROS2 publisher for each peripheral output
        self.haptic_publisher = self.create_publisher(JugMsg, TOPIC_HAPTICS, 10)
        self.voice_publisher = self.create_publisher(JugMsg, TOPIC_VOICE, 10)
        self.sound_publisher = self.create_publisher(JugMsg, TOPIC_SOUND, 10)

        # Set timer to check for messages to publish every 0.1 seconds
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # is tts currently available?
        self.tts_available = True
    
    #Listen for tts saying if it is available or not
    def tts_available_callback(self, msg):
        self.tts_available = msg.data


    # FINISH: Check messages by timestamp, check output periphery by message type
    # Note: Timestamps should be sent by the AI modules themselves with the message and module ID. 
    # We want the timestamp of the *original notification*, i.e. when it was created, not the timestamp at which the juggler *recieves* it.
    def listener_callback(self, msg):
        
        #time.sleep(6)

        # Store message string
        msg_string = msg.string.data

        # Get message priority level
        msg_priority = msg.Bool.data

        # Store message timestamp
        msg_timestamp = msg.header.stamp.sec

        # Store current timestamp
        current_time = self.get_clock().now().to_msg().sec

        # Store the time difference between current time and the message timestamp
        time_delay = current_time - msg_timestamp

        # What UIs does this message play on?
        outputPeripherals = user_peripheral_output_preferences.get(msg.int32.data)

        # if the message has tts
        if "V" in outputPeripherals:
            # check its priority and add it to the priorty queue
            if msg_priority:
                self.get_logger().info("Add voice priority message")
                self.tts_priority_messages_list.append(msg)
            else:
                self.tts_secondary_messages_list.append(msg)


    # Publish messages in both queues. If no messages remain in priority queue, then publish from secondary queue
    # Uses a separate thread from the listener.
    def timer_callback(self):

        #if tts is available, publish the next message
        if self.tts_available:
            #if the priority queue is not empty,
            if len(self.tts_priority_messages_list) > 0:
                self.collapse_tts_messages("Left", self.tts_priority_messages_list, left_tts_msgs)
                self.collapse_tts_messages("Center", self.tts_priority_messages_list, center_tts_msgs)
                self.collapse_tts_messages("Right", self.tts_priority_messages_list, right_tts_msgs)
                
                self.voice_publisher.publish(self.tts_priority_messages_list[0])
                self.tts_priority_messages_list.pop(0)
                self.tts_available = False
            # if the priorty message queue is empty, publish the next message in the secondary queue
            elif len(self.tts_secondary_messages_list) > 0:
                self.voice_publisher.publish(self.tts_secondary_messages_list[0])
                self.tts_secondary_messages_list.pop(0)
                self.tts_available = False
    
    # Collapse redundant tts messages
    def collapse_tts_messages(self, collapsed_msg, msgs_to_collapse, msg_set):
        # if the first message in the queue is part of the message set
        if msgs_to_collapse[0].string.data in msg_set == False:
            return
        #loop though the queue and collapse all the messages from that message set
        i = len(msgs_to_collapse) - 1
        multiple_msgs_from_set_found = False
        while i >= 0:
            if msgs_to_collapse[i].string.data in msg_set:
                if i > 0:
                    msgs_to_collapse.pop(i)
                    multiple_msgs_from_set_found = True
                elif multiple_msgs_from_set_found:
                    msgs_to_collapse[i].string.data = collapsed_msg
            i -= 1
    
    #Listen for wakewords. Clear the secondary message queue if the wakeword says to stop reading
    def wakeword_callback(self, wakeword: String):
        word = wakeword.data

        if word == "stop_reading" or word == "stop_identifying" or word == "stop_describing":
            self.tts_secondary_messages_list = []



def main(args=None):
    
    # Initialize ROS2 client library
    rclpy.init(args=args)

    # Create ROS2 node of TTS simulator
    juggler = PHINIXJuggler()

    # Stop executing main, loop for ROS2 events
    rclpy.spin(juggler)
    
    # Destory node then shutdown
    juggler.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




