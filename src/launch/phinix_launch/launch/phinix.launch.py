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
    
    face_rec_pkg_prefix = get_package_share_directory('phinix_face_recognition')
    face_rec_node_param_file = os.path.join(
        face_rec_pkg_prefix, 'param/phinix_face_recognition.param.yaml')
    face_rec__param = DeclareLaunchArgument(
        'face_rec_node_param_file',
        default_value=face_rec_node_param_file,
        description='face recognition settings'
    )

    oak_ros_node = Node(
        package="oak_ros",
        executable="oak_ros_py_exe"
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

    phinix_face_rec_node = Node(
        package="phinix_face_recognition",
        executable="phinix_face_rec_py_exe",
        parameters=[LaunchConfiguration('face_rec_node_param_file')],
    )

    phinix_wakeword_node = Node(
        package="phinix_openwakeword",
        executable="phinix_openwakeword_py_exe"
    )


    phinix_face_reg_node = Node(
        package="phinix_face_recognition",
        executable="phinix_face_reg_py_exe",
        parameters=[LaunchConfiguration('face_rec_node_param_file')],
    )

    phinix_sound_effects_ui_node = Node(
        package="phinix_sound_effects_ui",
        executable="phinix_sound_effects_ui_py_exe"
    )

    phinix_haptics_ui_node = Node(
        package="phinix_haptics_ui",
        executable="phinix_haptics_ui_py_exe"
    )

    phinix_node_manager_node = Node(
        package="phinix_node_manager",
        executable="phinix_node_manager_py_exe"
    )

    phinix_ui_message_juggler_node = Node(
        package="phinix_ui_message_juggler",
        executable="phinix_ui_message_juggler_py_exe"
    )

    phinix_obstacle_detector_ui_translator_node = Node(
        package="phinix_obstacle_detector_ui_translator",
        executable="phinix_obstacle_detector_ui_translator_py_exe"
    )

    phinix_object_detector_ui = Node(
        package="phinix_object_detector_ui",
        executable="phinix_object_detector_ui_py_exe"
    )

    phinix_sidewalk_detector = Node(
        package="phinix_sidewalk_detector",
        executable="phinix_sidewalk_detector_py_exe"
    )


    ld = [
        oak_ros_node,
        #phinix_obstacle_detector_node,
        tts_balacoon_param,
        #face_rec__param,
        #phinix_text_detector_node,
        #phinix_tts_simulator_node,
        phinix_tts_balacoon_node,
        phinix_sensor_abstractor_node,
        #phinix_face_rec_node,
        #phinix_face_reg_node,
        phinix_wakeword_node,
        phinix_sound_effects_ui_node,
        #phinix_haptics_ui_node,
        phinix_node_manager_node,
        phinix_ui_message_juggler_node,
        #phinix_obstacle_detector_ui_translator_node,
        phinix_object_detector_ui,
        phinix_sidewalk_detector
    ]

    return LaunchDescription(ld)