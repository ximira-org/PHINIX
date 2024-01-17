# Launching PHINIX's nodes individually

For all the commands listed below, please make sure you are in the PHINIX
repository's top-level directory.

### Launch camera node

To launch the camera node, run:

``` bash
source install/setup.bash
ros2 launch depthai_ros_driver camera.launch.py camera_model:=OAK-D-PRO-W
```

### Launch sensor abstractor

In a new terminal window, run:

``` bash
docker exec -it phinix_container bash
colcon build --packages-select phinix_sensor_abstractor
source install/setup.bash
ros2 run phinix_sensor_abstractor phinix_sensor_abstractor_py_exe
```

### Launch text detector (Note: works only with Intel GPU)
In a new terminal window, run:

``` bash
docker exec -it phinix_container bash
colcon build --packages-select phinix_text_detector
source install/setup.bash
ros2 run phinix_text_detector phinix_text_detector_py_exe
```

### Launch TTS

Open a new terminal window for the TTS simulator and run:

``` bash
docker exec -it phinix_container bash
source install/setup.bash
ros2 run phinix_tts_balacoon phinix_tts_simulator_py_exe
```

Open a new terminal for the TTS node and run:

``` bash
docker exec -it phinix_container bash
source install/setup.bash
ros2 run phinix_tts_balacoon phinix_tts_balacoon_py_exe --ros-args --params-file src/phinix_ui/phinix_tts_balacoon/param/phinix_tts_balacoon.param.yaml
```

### Launch RViZ for visualization

Open new terminal window for `rviz2` and run:

``` bash
xhost +local:docker
docker exec -it phinix_container bash
source install/setup.bash
rviz2
```
