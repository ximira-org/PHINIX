from setuptools import setup
import os
from glob import glob
 
package_name = 'oak_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'yolo_json'), glob('yolo_json/*.*')),
        (os.path.join('share', package_name, 'yolo_model'), glob('yolov8_model/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jk',
    maintainer_email='jagadishkmahendran@gmail.com',
    description='ROS wrapper for OAK camera',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "oak_ros_py_exe = oak_ros.oak_ros:main",
            "oak_ros_waist_py_exe = oak_ros.oak_ros_waist:main"
        ],
    },
)
