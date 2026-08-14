"""Inverse kinematics for the pre-grasp pose.

Only used for one thing: turn a table (x, y) from the homography into joint
angles that hover above it. The grasp itself is learned, so this does not need
to be precise -- it needs to be repeatable, because the policy will be trained
from wherever this puts the arm.

Link lengths below are placeholders. Extract the real values from the URDF joint
origins:

    SO-ARM100/Simulation/SO101/so101_new_calib.urdf

Note that URDF has all joint axes as `0 0 1` with the rotation baked into each
joint origin's rpy, so read the xyz offsets between consecutive joint origins
rather than assuming a textbook DH table.

Also note there are TWO calibration conventions in that repo and they differ by
roughly 90 degrees:
  - so101_new_calib.urdf  : mid-range zero  (matches current LeRobot)
  - so101_old_calib.urdf  : fully-extended zero
Mixing a policy trained under one with a URDF of the other silently offsets
everything. This module assumes new_calib.

TODO(felix): replace the closed-form placeholder with either
  (a) measured link lengths from the URDF, or
  (b) a numerical IK call through pinocchio/KDL inside the ROS container,
      which also gets you joint-limit handling for free.
"""

from __future__ import annotations

import math

import numpy as np

# Link lengths in mm, read from so101_new_calib.urdf joint origins.
#
#   shoulder_pan  origin z  = 0.0624          -> base height
#   elbow_flex    origin    = (-0.11257, -0.028, 0)   |.| = 0.1160  -> upper arm
#   wrist_flex    origin    = (-0.1349, 0.0052, 0)    |.| = 0.1350  -> lower arm
#
# The wrist->TCP distance is the one number NOT directly readable: the chain runs
# wrist_flex -> wrist_roll (offset 0.0637) -> gripper_frame (offset 0.0984), and
# the intervening rpy rotations mean those two do not simply add. 0.162 is the
# collinear upper bound; the true value is shorter. 120 mm is an estimate.
L_BASE_HEIGHT = 62.4
L_UPPER_ARM = 116.0
L_LOWER_ARM = 135.0
L_WRIST_TO_TIP = 120.0  # ESTIMATE -- see WRIST_OFFSET_DEG note below

WRIST_OFFSET_DEG = 0.0
"""Calibration constant for the wrist_flex zero.

The URDF's wrist_flex zero is not aligned with "gripper horizontal", and the
exact offset depends on the accumulated rpy rotations through the chain. Measure
it once: jog the arm until the gripper points straight down, read wrist_flex,
and set this to the value that makes pre_grasp_ik() return that number.

Until this is measured, top-down targets will often raise IKError for exceeding
the +/-95 deg wrist_flex limit. That failure is intentional and honest -- it
means the geometry does not close, not that the arm cannot do it.
"""

JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-97.0, 97.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.0, 163.0),
    "gripper": (-10.0, 100.0),
}
"""Derived from so101_new_calib.urdf radian limits, converted to degrees.
The URDF's effort=10 N.m / velocity=10 rad/s are onshape-to-robot placeholders,
not real STS3215 numbers -- the MJCF has the honest values (forcerange 3.35 N.m).
"""


class IKError(ValueError):
    """Target is unreachable or violates a joint limit."""


def clamp_to_limits(
    joints_deg: list[float], joint_names: list[str], tolerance_deg: float = 1.0
) -> list[float]:
    """Clamp to joint limits, loudly.

    A silent clamp is dangerous here: it turns "this target is unreachable" into
    "the arm confidently moved somewhere else", which reads downstream as a
    policy failure rather than a geometry bug. Anything clamped by more than
    `tolerance_deg` raises instead.
    """
    out: list[float] = []
    violations: list[str] = []

    for value, name in zip(joints_deg, joint_names, strict=True):
        lo, hi = JOINT_LIMITS_DEG[name]
        clamped = float(np.clip(value, lo, hi))
        if abs(clamped - value) > tolerance_deg:
            violations.append(f"{name}={value:.1f} deg outside [{lo:.0f}, {hi:.0f}]")
        out.append(clamped)

    if violations:
        raise IKError("joint limits violated: " + "; ".join(violations))
    return out


def pre_grasp_ik(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    wrist_pitch_deg: float = -90.0,
) -> list[float]:
    """Planar 2-link IK for a top-down pre-grasp pose.

    Args:
        x_mm, y_mm: table position from TableHomography.pixel_to_table().
        z_mm: height above the table to hover at.
        wrist_pitch_deg: -90 keeps the gripper pointing straight down, which is
            what a top-down grasp on a flat table wants.

    Returns joint positions in degrees, in JOINT_NAMES order, gripper open.
    """
    # Base rotation handles the azimuth; the rest is planar in the reach plane.
    pan = math.degrees(math.atan2(y_mm, x_mm))

    # Back off along the gripper axis so the 2-link solve targets the WRIST JOINT
    # rather than the tip. With wrist_pitch = -90 (straight down) the offset is
    # purely vertical: the wrist joint sits L_WRIST_TO_TIP above the contact point.
    pitch = math.radians(wrist_pitch_deg)
    reach = math.hypot(x_mm, y_mm)
    planar_r = reach - L_WRIST_TO_TIP * math.cos(pitch)
    planar_z = (z_mm - L_BASE_HEIGHT) - L_WRIST_TO_TIP * math.sin(pitch)

    dist = math.hypot(planar_r, planar_z)
    max_reach = L_UPPER_ARM + L_LOWER_ARM
    if dist > max_reach:
        raise IKError(f"target at {dist:.0f} mm exceeds reach {max_reach:.0f} mm")
    if dist < abs(L_UPPER_ARM - L_LOWER_ARM):
        raise IKError(f"target at {dist:.0f} mm is inside the minimum reach")

    # Law of cosines, elbow-up.
    cos_elbow = (dist**2 - L_UPPER_ARM**2 - L_LOWER_ARM**2) / (2 * L_UPPER_ARM * L_LOWER_ARM)
    elbow = math.acos(float(np.clip(cos_elbow, -1.0, 1.0)))

    cos_correction = (dist**2 + L_UPPER_ARM**2 - L_LOWER_ARM**2) / (2 * dist * L_UPPER_ARM)
    shoulder = math.atan2(planar_z, planar_r) + math.acos(float(np.clip(cos_correction, -1.0, 1.0)))

    shoulder_deg = math.degrees(shoulder)
    # Elbow-up: the second link bends back relative to the first.
    elbow_deg = -math.degrees(elbow)

    # The wrist compensates so the gripper holds `wrist_pitch_deg` in the world
    # frame regardless of arm pose. Absolute link angles accumulate down the
    # chain, so the wrist must cancel the sum of the two before it.
    wrist_deg = wrist_pitch_deg - (shoulder_deg + elbow_deg) + WRIST_OFFSET_DEG

    joints = [pan, -shoulder_deg, elbow_deg, wrist_deg, 0.0, 100.0]
    return clamp_to_limits(joints, list(JOINT_LIMITS_DEG.keys()))
