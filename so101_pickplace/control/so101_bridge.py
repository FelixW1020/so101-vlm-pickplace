"""Native macOS serial bridge -- runs OUTSIDE Docker, owns the servo bus.

This is the process that makes the whole ROS-2-on-a-Mac architecture work. It
holds /dev/tty.usbmodem*, publishes follower joint state, and applies incoming
goal positions. Everything else (ROS 2 nodes, MoveIt, RViz/Foxglove) lives in the
container and talks to this over TCP.

Run it in its own terminal, natively:

    python -m so101_pickplace.control.so101_bridge \\
        --port /dev/tty.usbmodem585A0076841 --id my_follower_arm

Uses LeRobot's SO101Follower rather than raw pyserial, because it already handles
calibration files, the 0-100 gripper convention and normalization -- reimplementing
that is how you end up with a policy trained in one frame and executed in another.

NOTE the import path: lerobot.robots.so_follower, NOT so101_follower. SO-100 and
SO-101 were merged into one generic SOFollower class and SO101Follower is now
just an alias. The CLI type string `so101_follower` is unchanged.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time

import zmq

from .protocol import (
    GOAL_PORT,
    STATE_PORT,
    TOPIC_GOAL,
    TOPIC_STATE,
    GoalPositions,
    JointState,
    from_lerobot_observation,
    to_lerobot_action,
)

logger = logging.getLogger(__name__)

DEFAULT_HZ = 30.0
STALE_GOAL_S = 0.5
"""Goals older than this are ignored. If the container dies mid-rollout the arm
holds its last commanded pose instead of replaying a stale target."""


class SO101Bridge:
    def __init__(self, port: str, robot_id: str, hz: float = DEFAULT_HZ, bind_host: str = "*"):
        self.port = port
        self.robot_id = robot_id
        self.hz = hz
        self.bind_host = bind_host
        self._running = False
        self._robot = None

    def _connect_robot(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        # Cameras are deliberately NOT configured here. Frames go to the
        # grounding node and the policy over their own path; muxing video
        # through this 30 Hz control bridge would just add latency.
        cfg = SO101FollowerConfig(port=self.port, id=self.robot_id)
        robot = SO101Follower(cfg)
        robot.connect()
        logger.info("connected to follower on %s (id=%s)", self.port, self.robot_id)
        return robot

    def run(self) -> None:
        ctx = zmq.Context.instance()
        state_pub = ctx.socket(zmq.PUB)
        state_pub.bind(f"tcp://{self.bind_host}:{STATE_PORT}")

        goal_sub = ctx.socket(zmq.SUB)
        goal_sub.bind(f"tcp://{self.bind_host}:{GOAL_PORT}")
        goal_sub.setsockopt(zmq.SUBSCRIBE, TOPIC_GOAL)
        # Only the newest goal matters; queueing them would fight the policy.
        goal_sub.setsockopt(zmq.CONFLATE, 1)

        self._robot = self._connect_robot()
        self._running = True

        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        period = 1.0 / self.hz
        logger.info("bridge up: state on :%d, goals on :%d, %.0f Hz",
                    STATE_PORT, GOAL_PORT, self.hz)

        try:
            while self._running:
                loop_start = time.perf_counter()

                obs = self._robot.get_observation()
                state = JointState(positions=from_lerobot_observation(obs))
                state_pub.send_multipart([TOPIC_STATE, state.pack()])

                try:
                    _, raw = goal_sub.recv_multipart(zmq.NOBLOCK)
                    goal = GoalPositions.unpack(raw)
                    age = time.time() - goal.timestamp
                    if age < STALE_GOAL_S:
                        self._robot.send_action(to_lerobot_action(goal.positions))
                    else:
                        logger.warning("dropped stale goal (%.2f s old)", age)
                except zmq.Again:
                    pass  # no new goal this tick -- servos hold position

                sleep_for = period - (time.perf_counter() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    logger.debug("loop overran by %.1f ms", -sleep_for * 1000)
        finally:
            self.shutdown(state_pub, goal_sub)

    def stop(self) -> None:
        self._running = False

    def shutdown(self, *sockets) -> None:
        logger.info("shutting down")
        for s in sockets:
            s.close(linger=0)
        if self._robot is not None:
            # SO101FollowerConfig defaults disable_torque_on_disconnect=True, so
            # the arm goes limp here. Support it before pulling the plug.
            self._robot.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser(description="Native macOS SO-101 serial bridge")
    ap.add_argument("--port", required=True, help="e.g. /dev/tty.usbmodem585A0076841")
    ap.add_argument("--id", dest="robot_id", required=True,
                    help="calibration id -- MUST match the id used with lerobot-calibrate")
    ap.add_argument("--hz", type=float, default=DEFAULT_HZ)
    ap.add_argument("--bind", default="*", help="bind host; use 127.0.0.1 to stay local")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    SO101Bridge(args.port, args.robot_id, args.hz, args.bind).run()


if __name__ == "__main__":
    main()
