#!/usr/bin/env bash
# Fetch the upstream SO-ARM100 hardware repo (meshes, STEP, URDF, MJCF).
#
# Not committed: it is ~428 MB of CAD, and it is upstream's to version, not ours.
# We use it as a mesh + kinematics donor only.
#
# Note the URDF in there is NOT ROS-ready -- relative mesh paths, ROS 1
# <transmission> tags, no <ros2_control> block. See README for what that means.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO_ROOT}/SO-ARM100"

if [ -d "${TARGET}" ]; then
    echo "already present: ${TARGET}"
    exit 0
fi

echo "cloning SO-ARM100 (~428 MB)..."
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git "${TARGET}"

echo
echo "done. Relevant files:"
echo "  ${TARGET}/Simulation/SO101/so101_new_calib.urdf   <- use this calibration"
echo "  ${TARGET}/Simulation/SO101/so101_new_calib.xml    <- MuJoCo, has real actuator limits"
echo "  ${TARGET}/Simulation/SO101/assets/                <- meshes"
echo
echo "WARNING: so101_old_calib.* uses a different joint zero (fully extended vs"
echo "mid-range). Mixing conventions silently offsets everything by ~90 degrees."
