# Setup PHINIX with Docker

This guide presents step-by-step instructions on how to set up PHINIX with
Docker. The current setup is developed and tested on Ubuntu 20.04 with ROS2
Foxy.

## Step 1: Install Ubuntu

- You can download the required version of Ubuntu from the
  [official Ubuntu 20.04.6 LTS (Focal Fossa) release page](https://releases.ubuntu.com/focal/).

- To install Ubuntu, follow the instructions in the
  [Install Ubuntu desktop tutorial](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview).

## Step 2: Install ROS2 Foxy

The Robot Operating System (ROS) is a set of software libraries and tools
for building robot applications. Follow these steps to install and set up ROS2:

- Install ROS2 Foxy by following
  [ROS2's official installation docs for Ubuntu](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html).

- Add the following line to the end of your `.bashrc` file:
  ``` bash
  source /opt/ros/foxy/setup.bash
  ```

- To make sure that ROS2 is installed correctly, you can run 
  [the talker and listener examples in their](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html#try-some-examples).

- Install `colcon`` for ROS2 by following
  [colcon's official installation instructions](https://colcon.readthedocs.io/en/released/user/installation.html).

## Step 3: Forking the PHINIX repository

If you haven't already, we recommend going through all the sections in GitHub's
official guide to [Fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo?tool=webui&platform=linux).
You should apply the steps outlined therein to fork and setup the
[ximira-org/PHINIX repository](https://github.com/ximira-org/PHINIX). Once you
have cloned and set up the repository locally, you may proceed to **Step 4**.

## Step 4: Running `setup_code.sh`

The `setup_code.sh` script will clone all the repositories required for
PHINIX. Please follow these steps:

- First, install `pip` for Python 3 by running
  `sudo apt-get install python3-pip`.

- Now, install `vcstool` using `pip` by running `sudo pip install vcstool`.

- Finally, run `sh setup_code.sh` to clone all the requisite sub-repositories.

## Step 5: Build Docker image

To build the Docker image:

- Make sure you are in the top-level `PHINIX/` directory.

- Build the `Dockerfile` by running the following command:
  ``` bash
  docker build -f utilities/docker/Dockerfile --build-arg USE_RVIZ=1 -t phinix_openvino_ros2 .
  ```

## Step 6: Run Docker

Run Docker using the following command:

``` bash
docker run --name phinix_container -it -v /dev/:/dev/ -v /home/<username>/PHINIX:/home/PHINIX -v /tmp/.X11-unix:/tmp/.X11-unix --privileged -e DISPLAY phinix_openvino_ros2
```

You should replace `<username>` in the above command with your username on your
Ubuntu installation. You should also replace `/home/<username>/PHINIX` with the
correct path to the PHINIX repository.

## Step 7: Build all the ROS modules

Follow these steps to build all the ROS modules:

- In the Terminal, make sure you are in the top-level directory of the `PHINIX`
  repository.

- In file `~/PHINIX/src/external/depthai-ros/depthai_filters/CMakeLists.txt`
  change line 7 from `set(opencv_version 4)` to `set(opencv_version 4.7.0)`.

- Run the following commands one after the other:
  ``` bash
  source /opt/ros/foxy/setup.bash
  ./src/external/depthai-ros/build.sh -s $BUILD_SEQUENTIAL -r 1 -m 1
  colcon build
  source install/setup.bash
  ```

## Step 8: Launch all ROS2 nodes

To launch all nodes together, run the following command:

``` bash
ros2 launch phinix_launch phinix.launch.py camera_model:=OAK-D-PRO-W
```

Alternatively, you may choose to launch each node individually. To do so, you
can follow the steps outlined in [this guide](launch_nodes_individually.md).

## Step 9: Launch RViZ for visualization

To launch `rviz2` for visualization, open a new terminal window and run the
following commands:

``` bash
xhost +local:docker
docker exec -it phinix_container bash
source install/setup.bash
rviz2
```
