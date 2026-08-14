"""Wire protocol between the native macOS serial bridge and the ROS 2 container.

Why this exists at all: Docker Desktop on macOS runs the daemon inside a Linux
VM and cannot pass through USB devices. `--device /dev/tty.usbmodem*` fails
because that path does not exist inside the VM -- the VM never sees the adapter.
This was closed as won't-fix upstream (docker/for-mac#5263, #900), and the
USB/IP escape hatch needs a macOS usbipd server that is still blocked on Apple
driver approval.

So: a native macOS process owns the serial bus, and ROS 2 reaches it over TCP.

Topology:
    macOS host                          Docker container (ros:jazzy, arm64)
    so101_bridge.py                     so101_bridge_ros node
      owns /dev/tty.usbmodem*    <--->    controller commands <-> ZMQ
      PUB state  :5555  ---------------->  SUB
      SUB goals  :5556  <----------------  PUB

msgpack over ZMQ rather than JSON: joint vectors are small but go at 30 Hz, and
msgpack avoids float repr churn. PUB/SUB (not REQ/REP) because a dropped state
frame should never stall the control loop -- last value wins.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import msgpack

STATE_PORT = 5555
GOAL_PORT = 5556

TOPIC_STATE = b"state"
TOPIC_GOAL = b"goal"

# LeRobot's SO-101 joint order. Keep this exact and in this order everywhere --
# the policy emits a flat vector and a silent reordering is nearly undebuggable.
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# ROS 2 joint names carry a _joint suffix (ros2_so_arm convention). Mapping
# lives here so the translation happens in exactly one place.
ROS_JOINT_NAMES = tuple(f"{n}_joint" for n in JOINT_NAMES)


@dataclass
class JointState:
    """Follower arm state. Positions are LeRobot-normalized, NOT radians.

    Body joints are in degrees (use_degrees=True) and the gripper is 0-100.
    Converting to radians for ROS/URDF happens in the ROS node, not here.
    """

    positions: list[float]
    timestamp: float = field(default_factory=time.time)
    velocities: list[float] | None = None

    def pack(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def unpack(cls, raw: bytes) -> JointState:
        return cls(**msgpack.unpackb(raw, raw=False))


@dataclass
class GoalPositions:
    """Commanded joint targets, same units and order as JointState."""

    positions: list[float]
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    """Who commanded this -- "policy" | "scripted" | "teleop". Logged so a rollout
    recording can be split into scripted vs learned segments after the fact."""

    def pack(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def unpack(cls, raw: bytes) -> GoalPositions:
        return cls(**msgpack.unpackb(raw, raw=False))


def to_lerobot_action(positions: list[float]) -> dict[str, float]:
    """Flat vector -> LeRobot's `{joint}.pos` action dict."""
    return {f"{name}.pos": float(p) for name, p in zip(JOINT_NAMES, positions, strict=True)}


def from_lerobot_observation(obs: dict) -> list[float]:
    """LeRobot observation dict -> flat vector in JOINT_NAMES order."""
    return [float(obs[f"{name}.pos"]) for name in JOINT_NAMES]
