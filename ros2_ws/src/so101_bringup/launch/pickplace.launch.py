"""Bring up the full two-stage stack.

    ros2 launch so101_bringup pickplace.launch.py
    ros2 launch so101_bringup pickplace.launch.py policy_enabled:=true \\
        checkpoint:=FelixW1020/act_so101_lang_pickplace

Before this does anything useful, the native macOS serial bridge must be running
OUTSIDE the container:

    python -m so101_pickplace.control.so101_bridge --port /dev/tty.usbmodem... --id ...

foxglove_bridge is launched instead of RViz. RViz2 over XQuartz on Apple Silicon
falls back to llvmpipe software rendering (ros2/rviz#929) and crawls on the
SO-101's multi-megabyte collision meshes. Connect Foxglove Studio (native arm64)
to ws://localhost:8765 instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    host = LaunchConfiguration("host")
    checkpoint = LaunchConfiguration("checkpoint")
    policy_enabled = LaunchConfiguration("policy_enabled")
    backend = LaunchConfiguration("backend")
    fallback_backend = LaunchConfiguration("fallback_backend")
    calibration_path = LaunchConfiguration("calibration_path")
    use_foxglove = LaunchConfiguration("use_foxglove")

    return LaunchDescription([
        DeclareLaunchArgument(
            "host", default_value="host.docker.internal",
            description="where the native serial bridge is listening",
        ),
        DeclareLaunchArgument("checkpoint", default_value=""),
        DeclareLaunchArgument("policy_enabled", default_value="false"),
        DeclareLaunchArgument("backend", default_value="gdino"),
        DeclareLaunchArgument(
            "fallback_backend", default_value="",
            description="set to 'claude' to escalate on low-confidence grounding",
        ),
        DeclareLaunchArgument(
            "calibration_path", default_value="calibration/table_homography.npz"
        ),
        DeclareLaunchArgument("use_foxglove", default_value="true"),

        Node(
            package="so101_ros", executable="bridge_node", name="so101_bridge",
            output="screen", parameters=[{"host": host, "rate_hz": 30.0}],
        ),
        Node(
            package="so101_ros", executable="grounding_node", name="so101_grounding",
            output="screen",
            parameters=[{
                "backend": backend,
                "fallback_backend": fallback_backend,
                "calibration_path": calibration_path,
            }],
        ),
        Node(
            package="so101_ros", executable="policy_node", name="so101_policy",
            output="screen",
            parameters=[{"checkpoint": checkpoint, "rate_hz": 30.0, "device": "cpu"}],
        ),
        Node(
            package="so101_ros", executable="orchestrator_node", name="so101_orchestrator",
            output="screen", parameters=[{"policy_enabled": policy_enabled}],
        ),
        Node(
            package="foxglove_bridge", executable="foxglove_bridge",
            name="foxglove_bridge", output="screen",
            condition=IfCondition(use_foxglove),
            parameters=[{"port": 8765}],
        ),
    ])
