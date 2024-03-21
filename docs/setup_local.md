## This document describes how to setup NUC to run PHINIX locally. 

### Install Ubuntu
* Follow instructions to install Ubuntu 22.04-iot and GPU drivers [here](https://github.com/intel/edge-insights-vision) 

* Installation: follow the instructions from [here](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview).

### Install vim and git
* `sudo apt-get install vim git -y`

### Setup git
* Follow [this](https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git) and [this](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address) to setup git configs. 

* Set up SSH key using [this](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

### Install ROS2 Humble
* Install ROS2 Humble using the binary version. Follow [this](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html).

* Include `source /opt/ros/humble/setup.bash` in bashrc.

* Make sure ROS2 installed correctly by running the talker listener examples explained in the above link.

* Install colcon for ROS2, follow [here](https://colcon.readthedocs.io/en/released/user/installation.html).

### Install Dependancies
* `sudo sh build_stack_humble.sh`

### Test With DepthAi Demo
* To make sure the camera is functioning properly, you can run the DepthAi demo `https://github.com/luxonis/depthai#depthai-api-demo-program`

### Pull External Repos
* `sh setup_code.sh`

### Build PHINIX
* Source ROS2 `source /opt/ros/humble/setup.bash`
* Build PHINX `colcon build`
* Source your install folder `source install/setup.bash`

### Launch all nodes at once with launch.py
* `ros2 launch phinix_launch phinix.launch.py camera_model:=OAK-D-PRO-W`