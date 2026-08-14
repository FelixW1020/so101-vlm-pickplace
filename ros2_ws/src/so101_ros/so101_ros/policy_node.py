"""Stage-2 node: runs the learned policy at 30 Hz when enabled.

IMPORTANT DEPLOYMENT NOTE. Do not run the policy inside the container in
production. A container on Apple Silicon sees CPU only -- no MPS, no CUDA -- and
LeRobot's dependency tree (torch, transformers, opencv-python, av) collides with
the apt-installed python3-numpy/python3-opencv that cv_bridge links against.

Two supported configurations:

  (a) DEV / SIM: run this node in the container on CPU. Fine for wiring up the
      graph and testing the state machine with an untrained or tiny checkpoint.

  (b) REAL: run the policy natively on macOS where it gets MPS, using LeRobot's
      own async inference (lerobot.async_inference PolicyServer + RobotClient,
      gRPC), and let this node just mirror the resulting joint targets into ROS
      for visualization and logging.

      Security: the async PolicyServer had an unauthenticated pickle RCE
      (CVE-2026-25874). Bind it to loopback and firewall the port.

This node implements (a) and leaves (b) as the flag `passthrough_only`.
"""

from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64MultiArray

from .bridge_node import GRIPPER_RANGE_RAD, ROS_JOINT_NAMES, rad_to_deg


class PolicyNode(Node):
    def __init__(self):
        super().__init__("so101_policy")

        self.declare_parameter("checkpoint", "")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("passthrough_only", False)
        self.declare_parameter("device", "cpu")

        self._cv_bridge = CvBridge()
        self._runner = None
        self._enabled = False
        self._joint_state: list[float] | None = None
        self._frames: dict[str, object] = {}

        # Camera names MUST match the ones used at record time. LeRobot keys
        # observations as observation.images.{name}; a mismatch here silently
        # feeds the policy the wrong view and it will fail in a way that looks
        # like undertraining.
        self.create_subscription(Image, "camera/top/image_raw",
                                 lambda m: self._on_image("top", m), 1)
        self.create_subscription(Image, "camera/wrist/image_raw",
                                 lambda m: self._on_image("wrist", m), 1)
        self.create_subscription(JointState, "joint_states", self._on_joint_state, 10)
        self.create_subscription(Bool, "so101/policy_enable", self._on_enable, 1)

        self._goal_pub = self.create_publisher(Float64MultiArray, "so101/goal_positions", 10)

        rate = self.get_parameter("rate_hz").value
        self.create_timer(1.0 / rate, self._step)
        self.get_logger().info(f"policy node up at {rate:.0f} Hz (disabled until enabled)")

    def _on_image(self, name: str, msg: Image) -> None:
        self._frames[name] = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _on_joint_state(self, msg: JointState) -> None:
        index = {n: i for i, n in enumerate(msg.name)}
        try:
            self._joint_state = [msg.position[index[n]] for n in ROS_JOINT_NAMES]
        except KeyError:
            pass  # partial joint state, e.g. from a different publisher

    def _on_enable(self, msg: Bool) -> None:
        self._enabled = bool(msg.data)
        if self._enabled:
            self._ensure_runner()
            if self._runner is not None:
                # Clear the action queue -- leftovers from the previous episode
                # are aimed at the previous object.
                self._runner.reset()
            self.get_logger().info("policy ENABLED")
        else:
            self.get_logger().info("policy disabled")

    def _ensure_runner(self):
        if self._runner is not None or self.get_parameter("passthrough_only").value:
            return
        checkpoint = self.get_parameter("checkpoint").value
        if not checkpoint:
            self.get_logger().error("no checkpoint set -- nothing to run")
            return
        from so101_pickplace.policy.infer import ACTPolicyRunner

        self._runner = ACTPolicyRunner(checkpoint, device=self.get_parameter("device").value)
        self._runner.load()

    def _step(self) -> None:
        if not self._enabled or self._runner is None:
            return
        if self._joint_state is None or "wrist" not in self._frames:
            return

        positions_deg = rad_to_deg(self._joint_state)
        action_deg = self._runner.select_action(positions_deg, dict(self._frames))

        import math

        out = [math.radians(a) for a in action_deg[:5]]
        lo, hi = GRIPPER_RANGE_RAD
        out.append(lo + (action_deg[5] / 100.0) * (hi - lo))

        msg = Float64MultiArray()
        msg.data = out
        self._goal_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
