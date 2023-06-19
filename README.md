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

### Build & install
* `colcon build`

* `source install/setup.bash`

### Launch camera node
* `ros2 launch depthai_ros_driver camera.launch.py camera_model:=OAK-D-PRO-W`

### Launch sensor abstractor
Open new terminal:

* `cd ~/PHINIX`

* `colcon build --packages-select phinix_sensor_abstractor`

* `source install/setup.bash`

* `ros2 run phinix_sensor_abstractor phinix_sensor_abstractor_py_exe`

### Launch text detector (Note: works only with Intel GPU)
Open new terminal:

* `cd ~/PHINIX`

* `colcon build --packages-select phinix_text_detector`

* `source install/setup.bash`

* `ros2 run phinix_text_detector phinix_text_detector_py_exe`

### Launch Rviz2 for visualization
Open new terminal:

* `Rviz2` # view topics by adding desired topic in the GUI

### Launch wakeword detector
Open new terminal:

* `cd ~/PHINIX`
* `colcon build --packages-select phinix_openwakeword`
* `source install/setup.bash`
* `ros2 run phinix_openwakeword phinix_openwakeword_py_exe --ros-args --params-file src/phinix_ui/phinix_openwakeword/param/phinix_openwakeword.param.yaml`

### Launch TTS
Open new terminal:

* `cd ~/PHINIX`
* `colcon build --packages-select phinix_tts_balacoon`
* `source install/setup.bash`
* `ros2 run phinix_tts_balacoon phinix_tts_simulator_py_exe`  # if simulator is needed
* `ros2 run phinix_tts_balacoon phinix_tts_balacoon_py_exe --ros-args --params-file src/phinix_ui/phinix_tts_balacoon/param/phinix_tts_balacoon.param.yaml`

### Status:

- [x] Initial architecture design and documentation
- [x] depthai-ros integration
- [x] initial version of sensor abstraction
- [x] text-detection (openvino gpu)
- [x] launch config
- [x] testing depthai object detection models
- [x] running depthai image overlays
- [ ] primary param file - very important. passing separate param file every time is not convenient
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
- [ ] elevation detection [3D]
- [ ] portal detection
- [ ] currency detection?
