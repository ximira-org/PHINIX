import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import  Node


def generate_launch_description():

    tts_balacoon_pkg_prefix = get_package_share_directory('phinix_tts_balacoon')
    tts_balacoon_node_param_file = os.path.join(
        tts_balacoon_pkg_prefix, 'param/phinix_tts_balacoon.param.yaml')
    tts_balacoon_param = DeclareLaunchArgument(
        'tts_balacoon_node_param_file',
        default_value=tts_balacoon_node_param_file,
        description='tts settings'
    )

    phinix_obstacle_detector_node = Node(
        package="phinix_obstacle_detector",
        executable="phinix_obstacle_detector_py_exe"
    )

    phinix_sensor_abstractor_node = Node(
        package="phinix_sensor_abstractor",
        executable="phinix_sensor_abstractor_py_exe"
    )

    phinix_text_detector_node = Node(
        package="phinix_text_detector",
        executable="phinix_text_detector_py_exe"
    )

    phinix_tts_simulator_node = Node(
        package="phinix_tts_balacoon",
        executable="phinix_tts_simulator_py_exe"
    )
    
    phinix_tts_balacoon_node = Node(
        package="phinix_tts_balacoon",
        executable="phinix_tts_balacoon_py_exe",
        parameters=[LaunchConfiguration('tts_balacoon_node_param_file')],
    )

    ld = [
        phinix_obstacle_detector_node,
        tts_balacoon_param,
        phinix_text_detector_node,
        phinix_tts_simulator_node,
        phinix_tts_balacoon_node,
        phinix_sensor_abstractor_node
    ]

    return LaunchDescription(ld)


