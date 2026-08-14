"""IK tests.

These pin down structure and failure behaviour, not absolute accuracy -- the
wrist_flex zero offset is still unmeasured (see WRIST_OFFSET_DEG). What they do
guarantee is that unreachable targets and limit violations raise loudly instead
of silently returning a clamped pose the arm would happily move to.
"""

from __future__ import annotations

import math

import pytest

from so101_pickplace.control import kinematics as k


@pytest.fixture
def calibrated(monkeypatch):
    """Approximate the measured wrist offset so the workspace closes.

    15 deg is illustrative, not measured. Replace it once the real value is
    read off the arm; these tests should keep passing.
    """
    monkeypatch.setattr(k, "WRIST_OFFSET_DEG", 15.0)


def test_pan_follows_target_azimuth(calibrated):
    for x, y in [(200.0, 0.0), (180.0, 40.0), (150.0, -60.0)]:
        joints = k.pre_grasp_ik(x, y, 80.0)
        assert joints[0] == pytest.approx(math.degrees(math.atan2(y, x)), abs=0.1)


def test_solution_respects_all_joint_limits(calibrated):
    joints = k.pre_grasp_ik(180.0, 40.0, 80.0)
    for value, (lo, hi) in zip(joints, k.JOINT_LIMITS_DEG.values(), strict=True):
        assert lo <= value <= hi


def test_gripper_starts_open(calibrated):
    assert k.pre_grasp_ik(180.0, 40.0, 80.0)[5] == 100.0


def test_beyond_reach_raises(calibrated):
    with pytest.raises(k.IKError, match="exceeds reach"):
        k.pre_grasp_ik(400.0, 0.0, 80.0)


def test_limit_violation_raises_rather_than_clamping():
    """Without the wrist offset applied, top-down targets exceed wrist_flex.

    The point of the test is the *raise*: a silently clamped pose reads
    downstream as a policy failure rather than a geometry bug.
    """
    with pytest.raises(k.IKError, match="joint limits violated"):
        k.pre_grasp_ik(200.0, 0.0, 80.0)


def test_clamp_reports_every_violated_joint():
    with pytest.raises(k.IKError) as excinfo:
        k.clamp_to_limits([999.0, 999.0, 0.0, 0.0, 0.0, 50.0], list(k.JOINT_LIMITS_DEG))
    message = str(excinfo.value)
    assert "shoulder_pan" in message
    assert "shoulder_lift" in message


def test_clamp_tolerates_rounding():
    """A hair outside the limit is rounding, not a geometry failure."""
    limits = list(k.JOINT_LIMITS_DEG)
    lo, _ = k.JOINT_LIMITS_DEG["shoulder_pan"]
    out = k.clamp_to_limits([lo - 0.4, 0.0, 0.0, 0.0, 0.0, 50.0], limits)
    assert out[0] == pytest.approx(lo)


def test_link_lengths_match_urdf():
    """Guards against someone 'fixing' these back to round numbers."""
    assert k.L_UPPER_ARM == pytest.approx(116.0, abs=0.5)
    assert k.L_LOWER_ARM == pytest.approx(135.0, abs=0.5)
    assert k.L_BASE_HEIGHT == pytest.approx(62.4, abs=0.5)
