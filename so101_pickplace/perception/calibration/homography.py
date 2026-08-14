"""Pixel -> table-plane mapping.

No depth camera needed. Every target lies on one flat table, and that plane
constraint removes the missing degree of freedom exactly: there is an exact 3x3
homography from image pixels to table coordinates in the robot's base XY frame.

Two ways to get the correspondences:

  (a) Print a ChArUco board, lay it at a known origin, detect corners. Fast, but
      only calibrates the camera -- arm kinematic error stays uncorrected.

  (b) TEACH BY TOUCH (recommended, see collect.py): jog the arm to ~12-20 points,
      touch the tip down, record forward-kinematics XY alongside the tip pixel.
      This folds camera error AND arm error into one map, which is what you
      actually want on a hobby-servo arm.

Accuracy to expect: ~2-5 mm from vision, another few mm from STS3215 backlash
and repeatability. Budget 5-10 mm total and design for it -- compliant/wide-jaw
grasp, slow vertical descent, objects >= 25 mm wide. This is not an industrial
arm and pretending otherwise is how you end up with a policy that never closes
on anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TableHomography:
    """Maps image pixels to table coordinates (mm) in the robot base XY plane."""

    H: np.ndarray
    """3x3 pixel -> table-mm."""

    z_table: float
    """Table height in the robot base frame, mm. The Z the gripper descends to."""

    rms_error_mm: float = float("nan")
    """Fit residual. If this is worse than ~5 mm, recalibrate before going further."""

    def pixel_to_table(self, uv: tuple[float, float]) -> tuple[float, float]:
        """Project one pixel onto the table plane. Returns (x_mm, y_mm)."""
        pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float32)
        xy = cv2.perspectiveTransform(pt, self.H)[0, 0]
        return float(xy[0]), float(xy[1])

    def table_to_pixel(self, xy: tuple[float, float]) -> tuple[float, float]:
        """Inverse map -- useful for drawing debug overlays."""
        pt = np.array([[[float(xy[0]), float(xy[1])]]], dtype=np.float32)
        uv = cv2.perspectiveTransform(pt, np.linalg.inv(self.H))[0, 0]
        return float(uv[0]), float(uv[1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, H=self.H, z_table=self.z_table, rms_error_mm=self.rms_error_mm)

    @classmethod
    def load(cls, path: str | Path) -> TableHomography:
        d = np.load(Path(path))
        return cls(
            H=d["H"],
            z_table=float(d["z_table"]),
            rms_error_mm=float(d.get("rms_error_mm", np.nan)),
        )


def fit_homography(
    image_points: np.ndarray,
    table_points: np.ndarray,
    z_table: float,
    ransac_reproj_threshold: float = 3.0,
) -> TableHomography:
    """Fit pixel -> table-mm from N >= 4 correspondences (8-20 recommended).

    Args:
        image_points: (N, 2) pixel coordinates.
        table_points: (N, 2) table coordinates in mm, robot base XY frame.
        z_table: table height in mm in the base frame.
    """
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    table_points = np.asarray(table_points, dtype=np.float32).reshape(-1, 2)

    if len(image_points) != len(table_points):
        raise ValueError(
            f"point count mismatch: {len(image_points)} pixels vs {len(table_points)} table points"
        )
    if len(image_points) < 4:
        raise ValueError(f"need at least 4 correspondences, got {len(image_points)}")

    H, inliers = cv2.findHomography(
        image_points, table_points, cv2.RANSAC, ransac_reproj_threshold
    )
    if H is None:
        raise RuntimeError(
            "homography fit failed -- points are probably collinear or badly mismatched"
        )

    predicted = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    residuals = np.linalg.norm(predicted - table_points, axis=1)
    rms = float(np.sqrt(np.mean(residuals**2)))

    n_in = int(inliers.sum()) if inliers is not None else len(image_points)
    if n_in < len(image_points):
        # Worth surfacing: a rejected point usually means a mis-clicked pixel.
        print(f"[calib] {len(image_points) - n_in} outlier(s) rejected by RANSAC")

    return TableHomography(H=H, z_table=z_table, rms_error_mm=rms)
