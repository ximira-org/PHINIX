## This document describes how to setup NUC to run NOVA locally. 

### Install Ubuntu
* Download Ubuntu 20.04 from [here](https://releases.ubuntu.com/focal/)

* Installation: follow the instructions from [here](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview).

### Install vim and git
* `sudo apt-get install vim git -y`

### Setup git
* Follow [this](https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git) and [this](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address) to setup git configs. 

* Set up SSH key using [this](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

### Install ROS2 Foxy
* Install ROS2 Foxy using the binary version. Follow [this](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html).

* Include `source /opt/ros/foxy/setup.bash` in bashrc.

* Make sure ROS2 installed correctly by running the talker listener examples explained in the above link.

* Install colcon for ROS2, follow [here](https://colcon.readthedocs.io/en/released/user/installation.html).

### Install VCS tool 
* `sudo pip install vcstool`

### Install OpenVINO
* Install OpenVino following the steps [here](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_from_archive_linux.html).

* Install development tools by following C++ instructions [here](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html#install-dev-tools).

* Install requirements for at least tensorflow2, pytorch, onnx. Eg: `pip install -r tools/requirements_pytorch.txt`

* [This](https://docs.openvino.ai/2022.3/notebooks/002-openvino-api-with-output.html#loading-openvino-runtime-and-showing-info) example should be working.

* Install Opencl Openvino runtime : https://github.com/intel/compute-runtime/

* Install GPU requirements: https://github.com/intel/compute-runtime/releases/tag/22.35.24055

### Install depthai-ros

* Follow steps [here](https://github.com/luxonis/depthai-ros).

* Probably building from source is a better idea as it provides more control. Check [here](https://github.com/luxonis/depthai-ros#install-from-source)

* `ros2 launch depthai_ros_driver camera.launch.py` should produce the necessary topics.

* Note: The above command doesn’t work for OAK-D Lite.
