# phinix_haptics_ui

# Run the Ros2 environment
Setup for the Ros2 is similar to the standalone TTS node.

### Requirements:
Install Pip:

`sudo apt install python3-pip`

Install bluepy:

`pip install bluepy`

Install Colcon following these instructions:

https://colcon.readthedocs.io/en/released/user/installation.html

Install Ros2

https://docs.ros.org/en/foxy/Installation.html

### To run standalone:
Turn on the haptic band and it will automatically start advertising a BLE connection.

`cd phinix_haptics_ble`

`colcon build`

 `source install/setup.bash` #might vary for MacOS and windows. may be for Mac: `. install/setup.bash`
 for Ubuntu standalone if you followed the ros2 installation page
`source /opt/ros/foxy/setup.bash`

`ros2 run phinix_haptics_ble phinix_haptics_ble_py_exe` # run simulator node

To run along with ostacle detection 

`ros2 run phinix_haptics_ble phinix_haptics_ble_py_exe`

# Haptic Band API:
## What the node expects:
The Phinix_Haptics_BLE node expects to receive messages from the phinix system that tell it about ways it needs to tell the bracelet to vibrate. Currently supported message types are:

### Obstacle detection:
State of the world as a series of boolean values represented as 0 or 1

`!o[upper left][middle left][low left][upper center][mid center][low center][upper right][mid right][low right]`

ex: `!o000111000` would be if there is a tall narrow object directly in front of you.

### Path detection:
How close are you to the edge of the path on the left or right as a pair or urgency values between 0 (not close) and 3 (very close)

`!p[left urgency][right urgency]`

ex: `!p03` would be if you were extremely close to the edge of the path on the right

### Cardinal Direction:
Which way are you facing as an abbreviated direction.

`!c[direction letter 1][direction letter 2]`

ex: `!cSW` would be Southwest or `!cNN` would be North. (I wanted all cardinal directions to be the same number of characters.)

### Battery Level:
How many bars are left as a value between 0 and 3

`!b[battery left]`

ex: `!b2` would be having 2 bars left

### Face Recognition:
Like speed dial when the system recognizes a persons face. There are 8 options, and it buzzes that many of the buzzer channels for a quarter second in order

`!f[person index]`

examples:

`!f0` would be the first person, and buzzer 0 would buzz one time for a quarter second

`!f3` would be person three, so buzzers 0 thorough 4 would buzz for a quater second one at a time.
