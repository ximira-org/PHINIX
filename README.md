# PHINIX
The main repository for PHINIX. The current setup is developed and tested on Ubuntu 20.04 with ROS2 Foxy.

## Local machine setup

* Follow [this doc](docs/README_setup_local.md) to setup fully then proceed to next steps.

## Setup code and running:

### Clone the repo
* `git clone git@github.com:ximira-org/PHINIX.git` 

* `cd ~/PHINIX`

### Clone sub repositories
* `vcs import < phinix.repos` # import depency repos

### build docker image

* `docker build -f utilities/docker/Dockerfile --build-arg USE_RVIZ=1 -t phinix_openvino_ros2 .`

### run docker 

In the below command replace <username> with your system user name

* `docker run --name phinix_container -it -v /dev/:/dev/ --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix phinix_openvino_ros2`

### build all the ROS modules

* `cd /home/PHINIX`

* `source /opt/ros/foxy/setup.bash`

* `colcon build`

### Launch camera node

* `cd /home/PHINIX`

* `source install/setup.bash`

* `ros2 launch depthai_ros_driver camera.launch.py camera_model:=OAK-D-PRO-W`

### Launch sensor abstractor
Open new terminal:

* `docker exec -it phinix_container bash`

* `cd /home/PHINIX`

* `colcon build --packages-select phinix_sensor_abstractor` # optional, colcon build builds automatically

* `source install/setup.bash`

* `ros2 run phinix_sensor_abstractor phinix_sensor_abstractor_py_exe`

### Launch text detector (Note: works only with Intel GPU)
Open new terminal:

* `docker exec -it phinix_container bash`

* `cd /home/PHINIX`

* `colcon build --packages-select phinix_text_detector`# optional

* `source install/setup.bash`

* `ros2 run phinix_text_detector phinix_text_detector_py_exe`

### Launch TTS
Open new terminal (for TTS simulator):

* `docker exec -it phinix_container bash`

* `cd /home/PHINIX`

* `source install/setup.bash`

* `ros2 run phinix_tts_balacoon phinix_tts_simulator_py_exe`  # if simulator is needed

* `ros2 run phinix_tts_balacoon phinix_tts_balacoon_py_exe --ros-args --params-file src/phinix_ui/phinix_tts_balacoon/param/phinix_tts_balacoon.param.yaml`

Open new terminal (for TTS node):

* `docker exec -it phinix_container bash`

* `cd /home/PHINIX`

* `source install/setup.bash`

* `ros2 run phinix_tts_balacoon phinix_tts_balacoon_py_exe --ros-args --params-file src/phinix_ui/phinix_tts_balacoon/param/phinix_tts_balacoon.param.yaml`
