# PHINIX_UI_Message_Juggler
This repository contains the code for the PHINIX UI Message Juggler. The juggler recieves messages from the various PHINIX modules and sends them to the appropriate peripheral outputs. Each message consists of a message string and a module ID. The juggler places each message into one of two First-in-First-out queues, named the priority and secondary queues. Which queue each message gets placed in is determined by the priority level of it's origin module, which the juggler determines by the message's module ID. Messages originating from safety-related modules, such as Pathway Recognition and Obstacle Detection, are sorted into the priority queue. All messages from non-safety related modules, such as Facial Recognition, Battery Level and User Input are sorted into the secondary queue. The juggler then sends the messages in each queue to the appropriate peripheral outputs (voice, haptics, sound effects) depending on the message type and the user's settings.

See the images at the bottom of this readme for a visualization.

The current setup is developed and tested on Ubuntu 20.04 with ROS2 Foxy.

## Local machine setup

* Follow [this doc](docs/README_setup_local.md) (everything before OpenVINO) to setup fully then proceed to next steps.

## Setup code and running:

### Clone the repo

* `cd ~`

* `git clone git@github.com:ximira-org/phinix_ui_message_juggler.git` 

* `cd ~/phinix_ui_message_juggler`

### Clone sub repositories
Install vcs tool: `sudo pip install vcstool`

* `vcs import < phinix.repos` # import depency repos

### Build the ROS module
Open new terminal (for the juggler)

* `cd ~/phinix_ui_message_juggler`

* `colcon build`

* `source install/setup.bash`

* `source /opt/ros/foxy/setup.bash`

* `ros2 run phinix_ui_message_juggler phinix_ui_message_juggler_py_exe`

Open new terminal (for the simulator - use to simulating module notifications)

* `cd ~/phinix_ui_message_juggler`

* `colcon build`

* `source install/setup.bash`

* `source /opt/ros/foxy/setup.bash`

* `ros2 run phinix_ui_message_juggler phinix_ui_juggler_simulator_py_exe`

### Status:

- [x] New simulator text file
- [x] Multithreaded publisher and subscriber (to avoid timer bottleneck)
- [ ] Make API style documentation, what is the input and output for message juggler (the real thing, i.e. not using the simulator)
- [x] Make a separate .py file with all message topics, and import them into the project
- [ ] Make a separate .py file with basic user default settings for preferred peripheral outputs (based on module ID)
- [x] Make git.ignore for build, log and install
- [x] Check message timestamps and discard out-of-date messages (Juggler should recieve timestamps from modules, make .txt timestamps for testing, also make live timestamp reading from file)
- [x] Check output periphery by module ID of each message and user preferences and publish accordingly
- [x] Interrupt secondary notification if priority notification is recieved.
- [ ] Build voice module
- [ ] Build sound effect module


![Priority Queues 1](docs/Priority_Queues_1.jpg)

![Priority Queues 2](docs/Priority_Queues_2.jpg)
