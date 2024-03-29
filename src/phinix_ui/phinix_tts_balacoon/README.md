# phinix_tts_balacoon
This is a standalone and integrate-able Balacoon based TTS solution for PHINIX visual assistance system

This is a ROS2 package with two nodes:

`phinix_tts_simulator` publishes sample text strings

`phinix_tts_balacoon` subsrcibes to string topic and converts it to speech using Balacoon TTS engine

Thanks to Balacoon TTS for their fantastic solution. More information can be found [here](https://balacoon.com/use/tts/package)

### Requirements:
Install Pip:

`sudo apt install python3-pip`

Install Colcon following these instructions:

https://colcon.readthedocs.io/en/released/user/installation.html

Install Ros2

https://docs.ros.org/en/foxy/Installation.html

Install Balacoon TTS:

`pip install -i https://pypi.fury.io/balacoon/ balacoon-tts`

Install SoundDevice

`pip install sounddevice --user`

Install Port Audio

`sudo apt-get install libportaudio2`

### To run standalone:

`git clone git@github.com:ximira-org/phinix_tts_balacoon.git`

`cd phinix_tts_balacoon`

`colcon build`

 `source install/setup.bash` #might vary for MacOS and windows. may be for Mac: `. install/setup.bash`
 for Ubuntu standalone if you followed the ros2 installation page
`source /opt/ros/foxy/setup.bash`

`ros2 run phinix_tts_balacoon phinix_tts_simulator_py_exe` # run simulator node

Open another terminal and run:

`ros2 run phinix_tts_balacoon phinix_tts_balacoon_py_exe --ros-args --params-file param/phinix_tts_balacoon.param.yaml` # run TTS node

### To run with complete PHINIX setup:

Add this repo to [phinix.repos](https://github.com/ximira-org/PHINIX/blob/main/phinix.repos) on PHINIX repo 

Import the repo to PHINIX by `vcs import < phinix.repos # import depency repos`

`cd ~/PHINIX`

`colcon build --packages-select phinix_tts_balacoon`

`ros2 run phinix_tts_balacoon phinix_tts_simulator_py_exe` # run simulator node if needed

`ros2 run phinix_tts_balacoon phinix_tts_balacoon_py_exe --ros-args --params-file param/phinix_tts_balacoon.param.yaml` # run TTS node

### Status:

- [x] ROS2 integration
- [x] TTS simulator integration
- [x] Switching between models
- [x] Auto download models if not present
- [ ] Move models outsid of share folder
- [ ] Launch file
- [ ] Choose speaking style

