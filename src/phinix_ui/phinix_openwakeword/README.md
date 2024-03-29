# phinix_openwakeword

This is a standalone and integrate-able OpenWakeWord based wakeword detector for PHINIX visual assistance system

This is a ROS2 package with the node:

`phinix_openwakeword` subsrcibes to string topic and converts it to speech using Balacoon TTS engine

Thanks to Openwakeword for their fantastic solution. More information can be found [here](https://github.com/dscripka/openWakeWord)

### Requirements:

Install Openwakeword 

`pip install openwakeword`

Install [speex](https://www.speex.org/) noise suppression:

`sudo apt-get install libspeexdsp-dev`

`pip install https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp38-cp38-linux_x86_64.whl`

Thanks to [TeaPoly](https://github.com/TeaPoly/speexdsp-ns-python) for speex python wrapper.

### To run standalone:

`git clone git@github.com:ximira-org/phinix_openwakeword.git`

`cd phinix_openwakeword`

`colcon build`

`ros2 run phinix_openwakeword phinix_openwakeword_py_exe --ros-args --params-file param/phinix_openwakeword.param.yaml` # run openwakeword node

### To run with complete PHINIX setup:

Add this repo to [phinix.repos](https://github.com/ximira-org/PHINIX/blob/main/phinix.repos) on PHINIX repo 

Import the repo to PHINIX by `vcs import < phinix.repos # import depency repos`

`cd ~/PHINIX`

`colcon build --packages-select phinix_openwakeword`

`ros2 run phinix_openwakeword phinix_openwakeword_py_exe --ros-args --params-file param/phinix_openwakeword.param.yaml` # run openwakeword node

### Status:

- [x] ROS2 integration
- [x] Switching between models
- [ ] Launch file
- [ ] Improve model accuracy
- [ ] Model optimization
- [ ] Potentially extend this as the primary ASR system?