# phinix_sidewalk_detector
This is a standalone ROS node to detect sidewalk

This is a ROS2 package with two nodes:

`phinix_sidewalk_detector` - node that runs sidewalk model and produces mask image

`video_simulator` - node that reads video and publishes


### Requirements:

Install requirements:

`python -m pip install segmentation-models-pytorch onnx nncf[torch] openvino openvino-dev`

`python -m pip install torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117`

### To run standalone:

`cd ~`

`mkdir sidewalk_src`

`cd sidewalk_src`

`git@github.com:ximira-org/phinix_sidewalk_detector.git`

`colcon build`

`cd phinix_sidewalk_detector/phinix_sidewalk_detector && python video_simulator.py <file_path>/PXL_20230904_201217246.TS.mp4` # run simulator node

#### open other terminal

`ros2 run phinix_sidewalk_detector phinix_sidewalk_detector_py_exe` # run sidewalk detector node