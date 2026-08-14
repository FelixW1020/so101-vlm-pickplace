# ROS 2 Jazzy for the SO-101 stack, arm64-native.
#
# Base image choice matters more than it looks: osrf/ros:jazzy-desktop is
# linux/amd64 ONLY, so on an M-series Mac it silently runs under emulation and
# anything graphical crawls. The official Docker Library `ros` images are
# multi-arch, so we use those and apt-install what we need.
#
# Verify you actually got arm64:
#   docker inspect --format '{{.Architecture}}' so101:jazzy
#
# Jazzy over Kilted (EOL Dec 2026) and Lyrical (too new -- the SO-ARM ecosystem
# has not caught up). Jazzy is LTS to May 2029.
FROM --platform=linux/arm64 ros:jazzy-ros-base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-moveit \
        ros-jazzy-xacro \
        ros-jazzy-joint-state-publisher-gui \
        ros-jazzy-robot-state-publisher \
        ros-jazzy-cv-bridge \
        ros-jazzy-foxglove-bridge \
        ros-jazzy-image-transport \
        ros-jazzy-usb-cam \
        python3-pip \
        python3-colcon-common-extensions \
        git \
    && rm -rf /var/lib/apt/lists/*

# PEP 668 is enforced on Noble. These are small pure-Python packages that do not
# disturb the apt-installed numpy/opencv that cv_bridge links against -- do NOT
# pip install torch or lerobot here, that breaks cv_bridge's ABI.
RUN pip3 install --no-cache-dir --break-system-packages \
        pyzmq \
        msgpack

WORKDIR /ws

# Source install of the host-side package so the nodes can import
# so101_pickplace.* (geometry + protocol only; the heavy extras stay out).
COPY pyproject.toml /ws/
COPY so101_pickplace /ws/so101_pickplace
RUN pip3 install --no-cache-dir --break-system-packages --no-deps -e /ws

COPY ros2_ws/src /ws/ros2_ws/src

# Build inside the image, not on a bind mount -- VirtioFS makes colcon roughly
# 4x slower across the host filesystem boundary.
RUN source /opt/ros/jazzy/setup.bash \
    && cd /ws/ros2_ws \
    && colcon build --symlink-install --event-handlers console_direct+

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
    && echo "source /ws/ros2_ws/install/setup.bash" >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
