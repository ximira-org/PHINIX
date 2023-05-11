# NOVA
The main repository for NOVA

## Setup and Running:

### Clone the repo
`git clone git@github.com:ximira-org/NOVA.git` 

`cd ~/NOVA`

### Clone sub repositories
`vcs import < nova.repos` # import depency repos

### Build & install
`colcon build`

`source install/setup.bash`

### Launch camera node
`ros2 launch depthai_ros_driver camera.launch.py camera_model:=OAK-D-PRO-W`

### Launch sensor abstractor
Open new terminal:

`cd ~/NOVA`

`colcon build --packages-select nova_sensor_abstractor`

`source install/setup.bash`

`ros2 run nova_sensor_abstractor nova_sensor_abstractor_py_exe`

### Launch text detector
Open new terminal:

`cd ~/NOVA`

`colcon build --packages-select nova_text_detector`

`source install/setup.bash`

`ros2 run nova_text_detector nova_text_detector_py_exe`

### Launch Rviz2 for visualization

`Rviz2` # view topics by adding desired topic in the GUI




