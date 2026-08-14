"""Scripted motion primitives -- the scaffolding around the learned policy.

Division of labour, and the reason this file exists:

    grounding  -> table XY          (perception)
    SCRIPTED   -> move above target (here)   <- large, coarse, easy to verify
    POLICY     -> descend and grasp (learned) <- small, precise, hard to script
    SCRIPTED   -> retreat and drop  (here)

The payoff is in demonstration count. If every demo starts from the same
scripted pre-grasp pose above the object, the policy only ever sees an
object-centric neighbourhood and never has to learn "where on the table is it" --
that is what the homography is for. Far fewer demos, and no grounding annotation
needed inside the dataset at all.

It also makes failures legible, which matters more than elegance here: wrong
place = grounding/calibration bug; right place but no grasp = policy bug.

One rule that is easy to get wrong: RECORD YOUR DEMONSTRATIONS STARTING FROM THE
SAME PRE-GRASP POSE this file produces. If the scripted approach hands over at a
pose the policy never saw in training, you get an out-of-distribution jump at the
handoff and the grasp fails for reasons that look like a bad policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .protocol import JOINT_NAMES  # noqa: F401  (documents the vector order)

# Joint-space waypoints, degrees, in JOINT_NAMES order. Placeholders -- jog the
# arm with `lerobot-teleoperate` and read off real numbers before trusting these.
# TODO(felix): replace with measured poses.
HOME_POSE = [0.0, -90.0, 90.0, 0.0, 0.0, 0.0]
DROP_POSE = [-60.0, -45.0, 60.0, 0.0, 0.0, 0.0]

GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 10.0
"""Not 0 -- close onto the object, not onto itself. The STS3215 will happily
stall and cook itself trying to reach a target it cannot make."""

APPROACH_HEIGHT_MM = 80.0
"""Hover height above the table for the pre-grasp pose."""


@dataclass
class Waypoint:
    positions: list[float]
    duration_s: float = 1.5
    label: str = ""


def interpolate(start: list[float], end: list[float], steps: int) -> list[list[float]]:
    """Linear joint-space interpolation.

    Deliberately not a spline. At these speeds on a 6-DoF desktop arm the
    difference is invisible, and linear interpolation has no overshoot -- which
    matters when the last waypoint is 8 cm above a table.
    """
    s, e = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return [(s + (e - s) * t).tolist() for t in np.linspace(0.0, 1.0, max(steps, 2))]


def build_approach(
    pre_grasp_joints: list[float],
    from_joints: list[float] | None = None,
) -> list[Waypoint]:
    """home -> open gripper -> hover above the grounded target."""
    start = from_joints if from_joints is not None else HOME_POSE
    open_at_home = list(start[:5]) + [GRIPPER_OPEN]
    return [
        Waypoint(open_at_home, 1.0, "open gripper"),
        Waypoint(list(pre_grasp_joints[:5]) + [GRIPPER_OPEN], 2.0, "hover above target"),
    ]


def build_retreat_and_drop(from_joints: list[float]) -> list[Waypoint]:
    """lift (still gripping) -> move to container -> release -> home."""
    lifted = list(from_joints)
    lifted[1] -= 25.0  # shoulder_lift up; crude but adequate for a vertical lift
    return [
        Waypoint(lifted, 1.0, "lift"),
        Waypoint(list(DROP_POSE[:5]) + [lifted[5]], 2.5, "move over container"),
        Waypoint(list(DROP_POSE[:5]) + [GRIPPER_OPEN], 0.8, "release"),
        Waypoint(HOME_POSE, 2.0, "home"),
    ]


def execute(
    waypoints: list[Waypoint],
    send_goal,
    get_state=None,
    hz: float = 30.0,
) -> list[float]:
    """Stream interpolated waypoints through `send_goal`.

    Args:
        send_goal: callable taking a list[float] of joint positions.
        get_state: optional callable returning current positions, used to start
            the first segment from where the arm actually is rather than from an
            assumed pose.
        hz: command rate. Match the bridge and the policy.

    Returns the final commanded position.
    """
    current = get_state() if get_state is not None else list(HOME_POSE)
    period = 1.0 / hz

    for wp in waypoints:
        steps = max(int(wp.duration_s * hz), 2)
        for target in interpolate(current, wp.positions, steps):
            send_goal(target)
            time.sleep(period)
        current = wp.positions

    return current
