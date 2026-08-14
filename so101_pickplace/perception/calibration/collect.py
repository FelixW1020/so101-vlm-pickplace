"""Teach-by-touch calibration collector.

Procedure, repeated ~12-20 times over the reachable workspace (include the
corners -- a homography fit only on the centre extrapolates badly):

  1. Jog the follower arm until the gripper tip touches the table.
  2. Read the tip XY from forward kinematics (or a ruler, for a first pass).
  3. Click that same tip in the camera image.

This folds camera error and arm kinematic error into a single map, which is the
right call on an arm with this much backlash.

Usage:
    python -m so101_pickplace.perception.calibration.collect \\
        --camera 0 --out calibration/table_homography.npz --z-table 0

TODO(felix): read XY from the live bridge instead of typing it. Needs FK, which
needs the URDF loaded -- see control/kinematics.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .homography import fit_homography

WINDOW = "calibration -- click gripper tip, 'u' undo, 'q' finish"


def collect_interactive(camera_index: int) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera_index}")

    image_points: list[tuple[float, float]] = []
    table_points: list[tuple[float, float]] = []
    pending: list[tuple[float, float]] = []

    def on_click(event, x, y, flags, param):  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((float(x), float(y)))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)

    print(__doc__)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")

            for i, (u, v) in enumerate(image_points):
                cv2.circle(frame, (int(u), int(v)), 5, (0, 255, 0), -1)
                cv2.putText(
                    frame, str(i), (int(u) + 8, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
            cv2.putText(
                frame, f"{len(image_points)} points collected",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            cv2.imshow(WINDOW, frame)

            # A click is only committed once the table XY has been entered, so a
            # stray click costs nothing.
            if pending:
                u, v = pending.pop()
                cv2.destroyWindow(WINDOW)
                try:
                    raw = input(f"clicked ({u:.0f}, {v:.0f}) -- table x,y in mm (blank to skip): ")
                    if raw.strip():
                        x_str, y_str = raw.split(",")
                        image_points.append((u, v))
                        table_points.append((float(x_str), float(y_str)))
                        print(f"  -> point {len(image_points) - 1} recorded")
                except (ValueError, IndexError):
                    print("  -> could not parse, skipped (want: 120, -45)")
                cv2.namedWindow(WINDOW)
                cv2.setMouseCallback(WINDOW, on_click)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("u") and image_points:
                image_points.pop()
                table_points.pop()
                print(f"undo -- {len(image_points)} points remain")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return np.array(image_points, dtype=np.float32), np.array(table_points, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect pixel<->table correspondences")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index (overhead cam)")
    ap.add_argument("--out", type=Path, default=Path("calibration/table_homography.npz"))
    ap.add_argument("--z-table", type=float, default=0.0, help="table height in base frame, mm")
    args = ap.parse_args()

    img_pts, tbl_pts = collect_interactive(args.camera)
    if len(img_pts) < 4:
        raise SystemExit(f"need >= 4 points, got {len(img_pts)}")

    calib = fit_homography(img_pts, tbl_pts, z_table=args.z_table)
    calib.save(args.out)

    print(f"\nsaved {args.out}  (RMS {calib.rms_error_mm:.2f} mm over {len(img_pts)} points)")
    if calib.rms_error_mm > 5.0:
        print("WARNING: >5 mm residual. Recollect -- most likely a mis-clicked tip or a")
        print("         mistyped table coordinate. Grasping will be unreliable at this error.")


if __name__ == "__main__":
    main()
