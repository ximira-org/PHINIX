from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()


    phinix_sensor_abstractor_node = Node(
        package="phinix_sensor_abstractor",
        executable="phinix_sensor_abstractor_py_exe"
    )

    phinix_text_detector_node = Node(
        package="phinix_text_detector",
        executable="phinix_text_detector_py_exe"
    )

    ld.add_action(phinix_sensor_abstractor_node)
    ld.add_action(phinix_text_detector_node)
    return ld