#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, NavSatFix
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy


topic_mappings_type = {
"/oak/stereo/camera_info" : ["/phinix/stereo/camera_info", CameraInfo],
"/oak/rgb/image_raw" : ["/phinix/rgb/image_raw", Image] ,
}

class NoveSensorAbstractorNode(Node):
    def __init__(self):
        super().__init__('phinix_sensor_abstractor_node')

        self._output_dict = {}
        self._input_dict = {}
        topic_in = None
        topic_out = None

        for key, value in topic_mappings_type.items():
            topic_in = key
            topic_out = value[0]

            self._output_dict[topic_out] = self.create_publisher(
                    value[1],
                    topic_out,
                    10
                )
            self._input_dict[topic_in] = self.create_subscription(
                value[1],
                topic_in,
                lambda msg, topic=topic_out: self.cb_publisher(msg, topic),
                10
            )

            self.get_logger().info("Mapping {} to {}".format(topic_in, topic_out))

            topic_in = None
            topic_out = None

    def cb_publisher(self, msg, topic_out):
        pub = self._output_dict[topic_out]
        pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    phinix_sensor_abstractor_node = NoveSensorAbstractorNode()
    rclpy.spin(phinix_sensor_abstractor_node)
    phinix_sensor_abstractor_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()