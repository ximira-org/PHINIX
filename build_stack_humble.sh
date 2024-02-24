#ARG ROS_DISTRO=humble
#FROM openvino/ubuntu20_dev:2023.2.0
#ARG USE_RVIZ
#ARG BUILD_SEQUENTIAL=0
#ARG OPENCV_VERSION=4.7.0
#ENV DEBIAN_FRONTEND=noninteractive
#USER root
#RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
sudo echo 'ALL ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
sudo apt-get update -q && \
    apt-get upgrade -yq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends keyboard-configuration language-pack-en && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends software-properties-common libusb-1.0-0-dev wget curl git build-essential ca-certificates tzdata tmux gnupg2 \
        vim sudo lsb-release locales bash-completion zsh iproute2 iputils-ping net-tools dnsutils && \
    rm -rf /var/lib/apt/lists/*

# setup timezone
sudo echo 'Etc/UTC' > /etc/timezone && \
   #  ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
sudo apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
#sudo echo "deb http://packages.ros.org/ros2/ubuntu focal main" > /etc/apt/sources.list.d/ros2-latest.list

# setup keys
#sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
#ENV LANG C.UTF-8
#ENV LC_ALL C.UTF-8

#ENV ROS_DISTRO humble

# install ros2 packages
sudo apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-core=0.9.2-1* \
    && rm -rf /var/lib/apt/lists/*

# install bootstrap tools
sudo apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    software-properties-common \
    libusb-1.0-0-dev \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# setup colcon mixin and metadata
colcon mixin add default \
      https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
      https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update

# install ros2 packages
sudo apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base=0.9.2-1* \
    && rm -rf /var/lib/apt/lists/*

sudo apt-get update && apt-get install -y --no-install-recommends \
    libqt5svg5 ros-humble-rviz2 ros-humble-rviz2 ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-camera-info-manager* ros-humble-image-transport*\
    && rm -rf /var/lib/apt/lists/*

python3 -m pip install --upgrade pip

# install dependencies for obstacle detector
python3 -m pip install pyopencl

# install dependencies for TTS
sudo apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2
python3 -m pip install sounddevice 
python3 -m pip install -i https://pypi.fury.io/balacoon/ balacoon-tts
python3 -m pip install pydub

# install dependencies for Text detection 
python3 -m pip install pip install rapidocr-openvinogpu

cd /opt/
# Install OpenCV from Source
git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git && \
    cd opencv && \
    mkdir build && \
    cd build && \
    cmake \
	-D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/ \
	-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
	-D WITH_V4L=ON \
	-D WITH_QT=OFF \
	-D WITH_OPENGL=ON \
	-D WITH_GSTREAMER=ON \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	-D BUILD_EXAMPLES=OFF .. && \
   make -j"$(nproc)" && \
   make install

# install dependencies for openwakeword
sudo apt-get update && apt-get install -y --no-install-recommends \
    libspeexdsp-dev portaudio19-dev python3-pyaudio pulseaudio
python3 -m pip install openwakeword \
      https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp310-cp310-linux_x86_64.whl

# install dependancies for bt connection 
pyhton3 -m pip install bluepy

# install dependancies for SFX UI
python3 -m pip install -U pygame --user

# install dependancies for wakeword
python3 -m pip install playsound

cd /home/
mkdir -p /home/PHINIX
#COPY ./ /home/PHINIX
# RUN wget -qO- https://raw.githubusercontent.com/luxonis/depthai-ros/main/install_dependencies.sh | sudo bash
sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
cd /home/PHINIX/src/external && rosdep install --from-paths . --ignore-src  -y
#ENV DEBIAN_FRONTEND=dialog
cd /home/PHINIX/src/external && cmake -S depthai-python/depthai-core -B /build -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=ON -D CMAKE_INSTALL_PREFIX=/usr/local
cmake --build /build --parallel 4 --config Relase --target install
cd /home/PHINIX/src/external/depthai-python && python3 -m pip install .
python3 -m pip install blobconverter
sudo apt install pocl-opencl-icd -y
python3 -m pip install rapidocr_openvino
python3 -m pip install ultralytics==8.0.231
sudo apt-get install libspeexdsp-dev
python3 -m pip install openwakeword https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp310-cp310-linux_x86_64.whl
cd /home/PHINIX
