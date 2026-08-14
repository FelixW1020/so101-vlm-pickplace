"""Stage-1 node: serves GroundTarget.

Subscribes to the overhead camera, and on request grounds a phrase in the latest
frame and projects it onto the table plane via the calibrated homography.

Runs the model in-process for now. If Grounding DINO on CPU inside the container
turns out too slow (likely -- the container gets no MPS), move the model to a
native host process behind the same ZMQ pattern as the serial bridge and make
this node a thin client. The service interface does not change either way.

Publishes an annotated debug image on ~/debug_image. Look at it. A grounding
stage you cannot see is a grounding stage you cannot debug, and a systematic
coordinate-convention offset is invisible in numbers but obvious in one frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from so101_msgs.srv import GroundTarget


class GroundingNode(Node):
    def __init__(self):
        super().__init__("so101_grounding")

        self.declare_parameter("backend", "gdino")
        self.declare_parameter("fallback_backend", "")  # "claude" to enable
        self.declare_parameter("min_confidence", 0.35)
        self.declare_parameter("calibration_path", "calibration/table_homography.npz")
        self.declare_parameter("camera_topic", "camera/overhead/image_raw")

        self._cv_bridge = CvBridge()
        self._latest_frame: np.ndarray | None = None
        self._grounder = None
        self._homography = None

        self.create_subscription(
            Image, self.get_parameter("camera_topic").value, self._on_image, 1
        )
        self._debug_pub = self.create_publisher(Image, "~/debug_image", 1)
        self._srv = self.create_service(GroundTarget, "ground_target", self._on_request)

        self._load_calibration()
        self.get_logger().info("grounding service ready on ~/ground_target")

    def _load_calibration(self) -> None:
        path = Path(self.get_parameter("calibration_path").value)
        if not path.exists():
            self.get_logger().warn(
                f"no calibration at {path} -- grounding will return pixels but cannot "
                "project to the table. Run so101_pickplace.perception.calibration.collect."
            )
            return
        from so101_pickplace.perception.calibration.homography import TableHomography

        self._homography = TableHomography.load(path)
        self.get_logger().info(
            f"loaded homography (RMS {self._homography.rms_error_mm:.1f} mm)"
        )

    def _ensure_grounder(self):
        if self._grounder is not None:
            return self._grounder
        from so101_pickplace.perception.grounding import CascadeGrounder, make_backend

        primary = make_backend(self.get_parameter("backend").value)
        fallback_name = self.get_parameter("fallback_backend").value
        fallback = make_backend(fallback_name) if fallback_name else None
        self._grounder = CascadeGrounder(
            primary, fallback, min_conf=self.get_parameter("min_confidence").value
        )
        return self._grounder

    def _on_image(self, msg: Image) -> None:
        self._latest_frame = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _on_request(self, request, response):
        if self._latest_frame is None:
            response.found = False
            response.message = "no camera frame received yet"
            return response

        frame = self._latest_frame.copy()
        grounder = self._ensure_grounder()
        if not request.use_fallback:
            grounder = grounder.primary

        result = grounder.ground(frame, request.phrase)
        if result is None:
            response.found = False
            response.message = f"abstained: could not confidently locate {request.phrase!r}"
            self.get_logger().info(response.message)
            self._publish_debug(frame, None, request.phrase)
            return response

        u, v = result.contact_point()
        response.found = True
        response.confidence = float(result.conf)
        response.source = result.source
        if result.bbox:
            response.bbox = [int(max(0, c)) for c in result.bbox]

        if self._homography is not None:
            x_mm, y_mm = self._homography.pixel_to_table((u, v))
            # Service contract is metres; the homography works in mm.
            response.target.x = x_mm / 1000.0
            response.target.y = y_mm / 1000.0
            response.target.z = self._homography.z_table / 1000.0
            response.message = f"{result.source} -> ({x_mm:.0f}, {y_mm:.0f}) mm"
        else:
            response.message = "no calibration loaded; target is unprojected"

        self.get_logger().info(response.message)
        self._publish_debug(frame, result, request.phrase)
        return response

    def _publish_debug(self, frame, result, phrase: str) -> None:
        if result is None:
            cv2.putText(frame, f"ABSTAIN: {phrase}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if result.bbox:
                x1, y1, x2, y2 = (int(c) for c in result.bbox)
                # Thin outline, not a filled box -- never occlude the pixels the
                # policy needs to see.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            u, v = (int(c) for c in result.contact_point())
            cv2.circle(frame, (u, v), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"{phrase} [{result.source} {result.conf:.2f}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._debug_pub.publish(self._cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8"))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
