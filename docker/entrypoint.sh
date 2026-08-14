#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
if [ -f /ws/ros2_ws/install/setup.bash ]; then
    source /ws/ros2_ws/install/setup.bash
fi

exec "$@"
