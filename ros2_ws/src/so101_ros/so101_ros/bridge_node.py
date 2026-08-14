"""Bridges ROS 2 <-> the native macOS serial process over ZMQ.

The container cannot see the USB adapter (Docker Desktop on macOS has no USB
passthrough), so this node is the seam. It:

  - SUBs joint state from the host bridge -> publishes /joint_states
  - SUBs /so101/goal_positions from ROS  -> PUBs goals to the host bridge

Units: the host bridge speaks LeRobot's convention (body joints in degrees,
gripper 0-100). ROS wants radians, and the URDF's gripper joint is in radians
too. The conversion happens HERE and nowhere else -- doing it in two places is
how you end up with an arm that moves to plausible-looking wrong poses.

If you get access to a Linux machine, this node becomes unnecessary: use the
real feetech_ros2_driver hardware interface (`hardware_type:=real`) and delete
this seam entirely. The rest of the stack does not change.
"""

from __future__ import annotations

import math

import rclpy
import zmq
from rclpy.node import Node
from sensor_msgs.msg import JointState as JointStateMsg
from std_msgs.msg import Float64MultiArray

# Duplicated from so101_pickplace.control.protocol -- the ROS package must be
# installable inside the container without the host-side Python package on the
# path. Keep the two in sync; they are the same wire format.
JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
ROS_JOINT_NAMES = [f"{n}_joint" for n in JOINT_NAMES]
STATE_PORT = 5555
GOAL_PORT = 5556
TOPIC_STATE = b"state"
TOPIC_GOAL = b"goal"

GRIPPER_RANGE_RAD = (-0.174533, 1.74533)
"""From so101_new_calib.urdf. LeRobot reports the gripper as 0-100."""


def deg_to_rad(positions: list[float]) -> list[float]:
    out = [math.radians(p) for p in positions[:5]]
    lo, hi = GRIPPER_RANGE_RAD
    out.append(lo + (positions[5] / 100.0) * (hi - lo))
    return out


def rad_to_deg(positions: list[float]) -> list[float]:
    out = [math.degrees(p) for p in positions[:5]]
    lo, hi = GRIPPER_RANGE_RAD
    out.append((positions[5] - lo) / (hi - lo) * 100.0)
    return out


class BridgeNode(Node):
    def __init__(self):
        super().__init__("so101_bridge")

        self.declare_parameter("host", "host.docker.internal")
        self.declare_parameter("rate_hz", 30.0)
        host = self.get_parameter("host").value
        rate_hz = self.get_parameter("rate_hz").value

        try:
            import msgpack

            self._msgpack = msgpack
        except ImportError as e:
            raise RuntimeError("pip install msgpack inside the container") from e

        ctx = zmq.Context.instance()
        self._state_sub = ctx.socket(zmq.SUB)
        self._state_sub.connect(f"tcp://{host}:{STATE_PORT}")
        self._state_sub.setsockopt(zmq.SUBSCRIBE, TOPIC_STATE)
        self._state_sub.setsockopt(zmq.CONFLATE, 1)

        self._goal_pub_zmq = ctx.socket(zmq.PUB)
        self._goal_pub_zmq.connect(f"tcp://{host}:{GOAL_PORT}")

        self._joint_pub = self.create_publisher(JointStateMsg, "joint_states", 10)
        self._goal_sub = self.create_subscription(
            Float64MultiArray, "so101/goal_positions", self._on_goal, 10
        )

        self._timer = self.create_timer(1.0 / rate_hz, self._pump_state)
        self._warned_no_state = False
        self.get_logger().info(f"bridging to host {host} (state :{STATE_PORT}, goal :{GOAL_PORT})")

    def _pump_state(self) -> None:
        try:
            _, raw = self._state_sub.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            if not self._warned_no_state:
                self.get_logger().warn(
                    "no state from the host bridge yet -- is so101_bridge.py running natively?"
                )
                self._warned_no_state = True
            return

        self._warned_no_state = False
        state = self._msgpack.unpackb(raw, raw=False)

        msg = JointStateMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ROS_JOINT_NAMES
        msg.position = deg_to_rad(state["positions"])
        if state.get("velocities"):
            msg.velocity = [math.radians(v) for v in state["velocities"]]
        self._joint_pub.publish(msg)

    def _on_goal(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != len(JOINT_NAMES):
            self.get_logger().error(
                f"goal has {len(msg.data)} values, expected {len(JOINT_NAMES)}"
            )
            return

        import time

        payload = self._msgpack.packb(
            {
                "positions": rad_to_deg(list(msg.data)),
                "timestamp": time.time(),
                "source": "ros",
            },
            use_bin_type=True,
        )
        self._goal_pub_zmq.send_multipart([TOPIC_GOAL, payload])


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
