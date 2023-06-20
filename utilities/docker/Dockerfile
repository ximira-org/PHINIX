ARG ROS_DISTRO=foxy
FROM ros:${ROS_DISTRO}-ros-base
ARG USE_RVIZ
ARG BUILD_SEQUENTIAL=0
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
   && apt-get -y install --no-install-recommends software-properties-common git libusb-1.0-0-dev wget zsh python3-colcon-common-extensions

RUN apt-get update && \
      apt-get -y install sudo

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
RUN echo 'ALL ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/
ENV DEBIAN_FRONTEND=dialog

ENV WS=/ws
RUN mkdir -p $WS/src

RUN if [ "$USE_RVIZ" = "1" ] ; then echo "RVIZ ENABLED" && sudo apt install -y ros-${ROS_DISTRO}-rviz2 ros-${ROS_DISTRO}-rviz-imu-plugin ; else echo "RVIZ NOT ENABLED"; fi