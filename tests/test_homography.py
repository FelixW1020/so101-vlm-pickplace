"""Geometry tests. These are the pieces that can be verified with no hardware."""

from __future__ import annotations

import numpy as np
import pytest

from so101_pickplace.perception.calibration.homography import TableHomography, fit_homography
from so101_pickplace.perception.grounding.base import Grounding, select_by_spatial_qualifier


def _synthetic_correspondences(n_side: int = 4):
    """A known perspective warp: table grid -> image pixels."""
    xs = np.linspace(-100.0, 100.0, n_side)
    ys = np.linspace(50.0, 250.0, n_side)
    table = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)

    true_H = np.array(
        [
            [2.5, 0.1, 320.0],
            [0.05, -2.2, 400.0],
            [0.0001, 0.0002, 1.0],
        ],
        dtype=np.float64,
    )
    import cv2

    pixels = cv2.perspectiveTransform(table.reshape(-1, 1, 2), true_H).reshape(-1, 2)
    return pixels.astype(np.float32), table


def test_homography_round_trip():
    pixels, table = _synthetic_correspondences()
    calib = fit_homography(pixels, table, z_table=0.0)

    assert calib.rms_error_mm < 0.5, "exact synthetic data should fit near-perfectly"

    for uv, expected in zip(pixels, table, strict=True):
        got = calib.pixel_to_table(tuple(uv))
        assert got == pytest.approx(tuple(expected), abs=0.5)


def test_homography_inverse():
    pixels, table = _synthetic_correspondences()
    calib = fit_homography(pixels, table, z_table=0.0)

    for uv in pixels:
        xy = calib.pixel_to_table(tuple(uv))
        back = calib.table_to_pixel(xy)
        assert back == pytest.approx(tuple(uv), abs=0.5)


def test_homography_save_load(tmp_path):
    pixels, table = _synthetic_correspondences()
    calib = fit_homography(pixels, table, z_table=12.5)

    path = tmp_path / "calib.npz"
    calib.save(path)
    loaded = TableHomography.load(path)

    assert loaded.z_table == pytest.approx(12.5)
    assert loaded.pixel_to_table((320.0, 400.0)) == pytest.approx(
        calib.pixel_to_table((320.0, 400.0)), abs=1e-6
    )


def test_too_few_points_rejected():
    with pytest.raises(ValueError, match="at least 4"):
        fit_homography(np.zeros((3, 2)), np.zeros((3, 2)), z_table=0.0)


def test_mismatched_point_counts_rejected():
    with pytest.raises(ValueError, match="mismatch"):
        fit_homography(np.zeros((5, 2)), np.zeros((4, 2)), z_table=0.0)


def _g(x: float, y: float, conf: float = 0.9) -> Grounding:
    return Grounding(uv=(x, y), bbox=(x - 10, y - 10, x + 10, y + 10), conf=conf, source="test")


def test_contact_point_uses_bottom_edge():
    """A tall object's box centre projects behind its footprint on the table."""
    g = Grounding(uv=(100.0, 100.0), bbox=(80.0, 60.0, 120.0, 140.0), conf=0.9, source="test")
    assert g.contact_point() == (100.0, 140.0)


def test_contact_point_without_bbox_falls_back_to_uv():
    g = Grounding(uv=(50.0, 60.0), bbox=None, conf=0.9, source="test")
    assert g.contact_point() == (50.0, 60.0)


def test_spatial_qualifier_leftmost():
    left, right = _g(10.0, 100.0), _g(200.0, 100.0)
    assert select_by_spatial_qualifier([right, left], "the leftmost red cube") is left


def test_spatial_qualifier_nearest_is_lower_in_frame():
    far, near = _g(100.0, 20.0), _g(100.0, 300.0)
    assert select_by_spatial_qualifier([far, near], "the nearest block") is near


def test_falls_back_to_highest_confidence():
    low, high = _g(10.0, 10.0, conf=0.4), _g(200.0, 200.0, conf=0.95)
    assert select_by_spatial_qualifier([low, high], "the red block") is high


def test_empty_candidates_returns_none():
    assert select_by_spatial_qualifier([], "anything") is None
