"""Wire-protocol and joint-ordering tests.

Joint order bugs are the nastiest class of failure in this stack: nothing throws,
the arm just moves to a plausible wrong pose. Pin the order down in a test.
"""

from __future__ import annotations

import pytest

from so101_pickplace.control.protocol import (
    JOINT_NAMES,
    ROS_JOINT_NAMES,
    GoalPositions,
    JointState,
    from_lerobot_observation,
    to_lerobot_action,
)


def test_joint_order_is_lerobot_canonical():
    assert JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )


def test_ros_names_add_joint_suffix():
    assert ROS_JOINT_NAMES == tuple(f"{n}_joint" for n in JOINT_NAMES)


def test_joint_state_round_trip():
    state = JointState(positions=[1.0, -2.0, 3.5, 0.0, 12.25, 87.5])
    restored = JointState.unpack(state.pack())
    assert restored.positions == state.positions
    assert restored.timestamp == pytest.approx(state.timestamp)


def test_goal_round_trip_preserves_source():
    goal = GoalPositions(positions=[0.0] * 6, source="policy")
    assert GoalPositions.unpack(goal.pack()).source == "policy"


def test_lerobot_action_conversion_round_trips():
    positions = [10.0, -20.0, 30.0, -40.0, 50.0, 75.0]
    action = to_lerobot_action(positions)

    assert action["shoulder_pan.pos"] == 10.0
    assert action["gripper.pos"] == 75.0

    # An observation uses the same `.pos` keys, so the inverse must recover order.
    assert from_lerobot_observation(action) == positions


def test_action_conversion_rejects_wrong_length():
    with pytest.raises(ValueError):
        to_lerobot_action([0.0, 1.0, 2.0])
