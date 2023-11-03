import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    log_level = 'info'
    if(context.environment.get('DEPTHAI_DEBUG')=='1'):
        log_level='debug'

    urdf_launch_dir = os.path.join(get_package_share_directory('depthai_descriptions'), 'launch')
    
    params_file = LaunchConfiguration("params_file")
    camera_model = LaunchConfiguration('camera_model',  default = 'OAK-D')

    name = LaunchConfiguration('name').perform(context)

    parent_frame = LaunchConfiguration('parent_frame',  default = 'oak-d-base-frame')
    cam_pos_x    = LaunchConfiguration('cam_pos_x',     default = '0.0')
    cam_pos_y    = LaunchConfiguration('cam_pos_y',     default = '0.0')
    cam_pos_z    = LaunchConfiguration('cam_pos_z',     default = '0.0')
    cam_roll     = LaunchConfiguration('cam_roll',      default = '0.0')
    cam_pitch    = LaunchConfiguration('cam_pitch',     default = '0.0')
    cam_yaw      = LaunchConfiguration('cam_yaw',       default = '0.0')
    
    return [
            Node(
                condition=IfCondition(LaunchConfiguration("use_rviz").perform(context)),
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="log",
                arguments=["-d", LaunchConfiguration("rviz_config")],
            ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(urdf_launch_dir, 'urdf_launch.py')),
            launch_arguments={'tf_prefix': name,
                              'camera_model': camera_model,
                              'base_frame': name,
                              'parent_frame': parent_frame,
                              'cam_pos_x': cam_pos_x,
                              'cam_pos_y': cam_pos_y,
                              'cam_pos_z': cam_pos_z,
                              'cam_roll': cam_roll,
                              'cam_pitch': cam_pitch,
                              'cam_yaw': cam_yaw}.items()),

        ComposableNodeContainer(
            name=name+"_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                    ComposableNode(
                        package="depthai_ros_driver",
                        plugin="depthai_ros_driver::Camera",
                        name=name,
                        parameters=[params_file],
                    )
            ],
            arguments=['--ros-args', '--log-level', log_level],
            output="both",
        ),

    ]


def generate_launch_description():
    depthai_prefix = get_package_share_directory("depthai_ros_driver")

    tts_balacoon_pkg_prefix = get_package_share_directory('phinix_tts_balacoon')
    tts_balacoon_node_param_file = os.path.join(
        tts_balacoon_pkg_prefix, 'param/phinix_tts_balacoon.param.yaml')
    tts_balacoon_param = DeclareLaunchArgument(
        'tts_balacoon_node_param_file',
        default_value=tts_balacoon_node_param_file,
        description='tts settings'
    )

    declared_arguments = [
        DeclareLaunchArgument("name", default_value="oak"),
        DeclareLaunchArgument("parent_frame", default_value="oak-d-base-frame"),
        DeclareLaunchArgument("cam_pos_x", default_value="0.0"),
        DeclareLaunchArgument("cam_pos_y", default_value="0.0"),
        DeclareLaunchArgument("cam_pos_z", default_value="0.0"),
        DeclareLaunchArgument("cam_roll", default_value="0.0"),
        DeclareLaunchArgument("cam_pitch", default_value="0.0"),
        DeclareLaunchArgument("cam_yaw", default_value="0.0"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(depthai_prefix, 'config', 'camera.yaml')),
        DeclareLaunchArgument("use_rviz", default_value='false'),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(depthai_prefix, "config", "rviz", "rgbd.rviz"))
    ]

    ld = LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])


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

    ld.add_action(tts_balacoon_param)
    ld.add_action(phinix_sensor_abstractor_node)
    ld.add_action(phinix_text_detector_node)
    ld.add_action(phinix_tts_simulator_node)
    ld.add_action(phinix_tts_balacoon_node)

    return ld


