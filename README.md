# PHINIX
The main repository for PHINIX. The current setup is developed and tested on Ubuntu 20.04 with ROS2 Foxy.

## Setup code and running:

### Install Ubuntu
* Download Ubuntu 20.04 from [here](https://releases.ubuntu.com/focal/)

* Installation: follow the instructions from [here](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview).

### Install ROS2 Foxy
* Install ROS2 Foxy using the binary version. Follow [this](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html).

* Include `source /opt/ros/foxy/setup.bash` in bashrc.

* Make sure ROS2 installed correctly by running the talker listener examples explained in the above link.

* Install colcon for ROS2, follow [here](https://colcon.readthedocs.io/en/released/user/installation.html).

### Install vim and git
* `sudo apt-get install vim git -y`

### Setup git
* Follow [this](https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git) and [this](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address) to setup git configs. 

* Set up SSH key using [this](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

### Clone the repo

* `cd ~`

* `git clone git@github.com:ximira-org/PHINIX.git` 

* `cd ~/PHINIX`

### Clone sub repositories
Install vcs tool: `sudo pip install vcstool`

* `vcs import < phinix.repos` # import depency repos

### build docker image

* `docker build -f utilities/docker/Dockerfile --build-arg USE_RVIZ=1 -t phinix_openvino_ros2 .`

### run docker 

In the below command replace <username> with your system user name

* `docker run --name phinix_container -it -v /dev/:/dev/ -v /home/<username>/PHINIX:/home/PHINIX -v /tmp/.X11-unix:/tmp/.X11-unix --privileged -e DISPLAY phinix_openvino_ros2`

If PHINIX repo is downloaded to different location, replace `/home/<username>/PHINIX` with the correct path in the above command

### build all the ROS modules

* `cd /home/PHINIX`

* In file `~/PHINIX/src/external/depthai-ros/depthai_filters/CMakeLists.txt` edit line 7 from `set(opencv_version 4)` to `set(opencv_version 4.7.0)`

* `source /opt/ros/foxy/setup.bash`

* `./src/external/depthai-ros/build.sh -s $BUILD_SEQUENTIAL -r 1 -m 1`

* `colcon build`

* `source install/setup.bash`

### Launch all nodes at once via launch file

* `ros2 launch phinix_launch phinix.launch.py camera_model:=OAK-D-PRO-W`

### Launch RViZ for visualization

Open new terminal (for rviz2)

* `xhost +local:docker`

* `docker exec -it phinix_container bash`

* `source install/setup.bash`

* `rviz2` # add the necessary topics for visualization


## Below steps are for launching nodes invidually. The below steps are not needed if `phinix.launch.py` (above) is launched.

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

Open new terminal (for TTS node):

* `docker exec -it phinix_container bash`

* `cd /home/PHINIX`

* `source install/setup.bash`

* `ros2 run phinix_tts_balacoon phinix_tts_balacoon_py_exe --ros-args --params-file src/phinix_ui/phinix_tts_balacoon/param/phinix_tts_balacoon.param.yaml`

### Launch RViZ for visualization

Open new terminal (for rviz2)

* `xhost +local:docker`

* `docker exec -it phinix_container bash`

* `source install/setup.bash`

* `rviz2` # add the necessary topics for visualization

