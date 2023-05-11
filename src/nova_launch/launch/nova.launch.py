from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()


    nova_sensor_abstractor_node = Node(
        package="nova_sensor_abstractor",
        executable="nova_sensor_abstractor_py_exe"
    )

    nova_text_detector_node = Node(
        package="nova_text_detector",
        executable="nova_text_detector_py_exe"
    )

    ld.add_action(nova_sensor_abstractor_node)
    ld.add_action(nova_text_detector_node)
    return ld