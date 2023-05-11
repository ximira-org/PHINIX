# NOVA
The main repository for NOVA. The current setup is developed and tested on Ubuntu 20.04 with ROS2 Foxy.

## Local machine setup

* Follow [this doc](docs/README_setup_local.md) to setup fully then proceed to next steps.

## Setup code and running:

### Clone the repo
* `git clone git@github.com:ximira-org/NOVA.git` 

* `cd ~/NOVA`

### Clone sub repositories
* `vcs import < nova.repos` # import depency repos

### Build & install
* `colcon build`

* `source install/setup.bash`

### Launch camera node
* `ros2 launch depthai_ros_driver camera.launch.py camera_model:=OAK-D-PRO-W`

### Launch sensor abstractor
Open new terminal:

* `cd ~/NOVA`

* `colcon build --packages-select nova_sensor_abstractor`

* `source install/setup.bash`

* `ros2 run nova_sensor_abstractor nova_sensor_abstractor_py_exe`

### Launch text detector (Note: works only with Intel GPU)
Open new terminal:

* `cd ~/NOVA`

* `colcon build --packages-select nova_text_detector`

* `source install/setup.bash`

* `ros2 run nova_text_detector nova_text_detector_py_exe`

### Launch Rviz2 for visualization
Open new terminal:

* `Rviz2` # view topics by adding desired topic in the GUI

### Status:

- [x] Initial architecture design and documentation
- [x] depthai-ros integration
- [x] initial version of sensor abstraction
- [x] text-detection (openvino gpu)
- [x] launch config
- [x] testing depthai object detection models
- [x] running depthai image overlays
- [ ] text detection CPU
- [ ] custom image overlays
- [ ] advanced sensor abstraction
- [ ] NCS2 integration
- [ ] wrist band integration
- [ ] earphone integration
- [ ] speech recognition module
- [ ] text-to-speech module
- [ ] obstacle detection (3D)
- [ ] object detection node
- [ ] palm recognition
- [ ] facial recognition
- [ ] sidewalk scene understanding
- [ ] floor / ground segmentation [3D]
- [ ] elevation detection
- [ ] portal detection
- [ ] currency detection?
