"""Task orchestrator: the state machine that ties both stages together.

    IDLE -> GROUNDING -> APPROACH -> GRASP -> RETREAT -> PLACE -> IDLE
                  |          |         |
                  v          v         v
              (abstain)  (IK fail)  (timeout)  -> ABORT

Serves the PickAndPlace action. The split of responsibility is the whole design:

  GROUNDING   VLM, once, on a static scene
  APPROACH    scripted -- coarse, large motion, from the homography
  GRASP       LEARNED -- the only part that is hard to script
  RETREAT     scripted
  PLACE       scripted -- fixed container location

Everything scripted is verifiable without a trained policy, which means you can
bring up and debug the entire loop before a single demonstration is recorded.
"""

from __future__ import annotations

import time
from enum import Enum, auto

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from so101_msgs.action import PickAndPlace
from so101_msgs.srv import GroundTarget
from std_msgs.msg import Bool, Float64MultiArray


class Stage(Enum):
    IDLE = auto()
    GROUNDING = auto()
    APPROACH = auto()
    GRASP = auto()
    RETREAT = auto()
    PLACE = auto()
    ABORT = auto()


class OrchestratorNode(Node):
    def __init__(self):
        super().__init__("so101_orchestrator")

        self.declare_parameter("approach_height_mm", 80.0)
        self.declare_parameter("grasp_timeout_s", 15.0)
        self.declare_parameter("policy_enabled", False)  # flip on once trained

        cb = ReentrantCallbackGroup()
        self._ground_client = self.create_client(GroundTarget, "ground_target", callback_group=cb)
        self._goal_pub = self.create_publisher(Float64MultiArray, "so101/goal_positions", 10)
        # Must match policy_node's subscription type -- Bool, not Float64MultiArray.
        self._policy_enable_pub = self.create_publisher(Bool, "so101/policy_enable", 1)

        self._action_server = ActionServer(
            self,
            PickAndPlace,
            "pick_and_place",
            execute_callback=self._execute,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=cb,
        )
        self._stage = Stage.IDLE
        self.get_logger().info("orchestrator ready -- action server on ~/pick_and_place")

    def _send_joints(self, positions: list[float], _source: str = "scripted") -> None:
        msg = Float64MultiArray()
        msg.data = [float(p) for p in positions]
        self._goal_pub.publish(msg)

    def _feedback(self, handle, stage: Stage, progress: float, detail: str = "") -> None:
        fb = PickAndPlace.Feedback()
        fb.stage = stage.name.lower()
        fb.progress = float(progress)
        fb.detail = detail
        handle.publish_feedback(fb)
        self.get_logger().info(f"[{stage.name}] {detail}")

    async def _execute(self, handle):
        started = time.time()
        request = handle.request
        result = PickAndPlace.Result()

        # --- stage 1: ground -------------------------------------------------
        self._stage = Stage.GROUNDING
        self._feedback(handle, Stage.GROUNDING, 0.0, f"grounding {request.phrase!r}")

        if not self._ground_client.wait_for_service(timeout_sec=5.0):
            result.success = False
            result.outcome = "grounding_failed"
            result.duration_s = time.time() - started
            handle.abort()
            return result

        ground_req = GroundTarget.Request(phrase=request.phrase, use_fallback=True)
        ground_res = await self._ground_client.call_async(ground_req)

        if not ground_res.found:
            self._feedback(handle, Stage.ABORT, 1.0, ground_res.message)
            result.success = False
            result.outcome = "grounding_failed"
            result.duration_s = time.time() - started
            handle.abort()
            return result

        result.grounding_confidence = ground_res.confidence
        self._feedback(
            handle, Stage.GROUNDING, 0.2,
            f"found via {ground_res.source} at "
            f"({ground_res.target.x:.3f}, {ground_res.target.y:.3f}) m",
        )

        if request.dry_run:
            result.success = True
            result.outcome = "success"
            result.duration_s = time.time() - started
            handle.succeed()
            return result

        # --- stage 2a: scripted approach -------------------------------------
        self._stage = Stage.APPROACH
        from so101_pickplace.control import scripted
        from so101_pickplace.control.kinematics import IKError, pre_grasp_ik

        try:
            pre_grasp = pre_grasp_ik(
                x_mm=ground_res.target.x * 1000.0,
                y_mm=ground_res.target.y * 1000.0,
                z_mm=self.get_parameter("approach_height_mm").value,
            )
        except IKError as e:
            self._feedback(handle, Stage.ABORT, 1.0, f"unreachable: {e}")
            result.success = False
            result.outcome = "approach_failed"
            result.duration_s = time.time() - started
            handle.abort()
            return result

        self._feedback(handle, Stage.APPROACH, 0.4, "moving to pre-grasp")
        scripted.execute(scripted.build_approach(pre_grasp), send_goal=self._send_joints)

        if handle.is_cancel_requested:
            handle.canceled()
            result.outcome = "cancelled"
            return result

        # --- stage 2b: learned grasp -----------------------------------------
        self._stage = Stage.GRASP
        if not self.get_parameter("policy_enabled").value:
            # Before a policy exists this is where you verify the rest of the
            # loop: the arm should be hovering directly over the object.
            self._feedback(handle, Stage.GRASP, 0.6, "policy disabled -- stopping at pre-grasp")
            result.success = True
            result.outcome = "success"
            result.duration_s = time.time() - started
            handle.succeed()
            return result

        self._feedback(handle, Stage.GRASP, 0.6, "handing off to policy")
        # TODO(felix): hand control to the policy node and wait for it to report
        # gripper-closed or the timeout to expire. Needs the trained checkpoint,
        # so it is a stub until data collection happens.
        grasped = self._run_policy_grasp(handle)

        if not grasped:
            self._feedback(handle, Stage.ABORT, 1.0, "grasp timed out")
            result.success = False
            result.outcome = "grasp_failed"
            result.duration_s = time.time() - started
            handle.abort()
            return result

        # --- stage 2c: scripted retreat and drop -----------------------------
        self._stage = Stage.RETREAT
        self._feedback(handle, Stage.RETREAT, 0.8, "lifting and moving to container")
        scripted.execute(
            scripted.build_retreat_and_drop(pre_grasp), send_goal=self._send_joints
        )

        self._stage = Stage.IDLE
        result.success = True
        result.outcome = "success"
        result.duration_s = time.time() - started
        self._feedback(handle, Stage.PLACE, 1.0, f"done in {result.duration_s:.1f} s")
        handle.succeed()
        return result

    def _run_policy_grasp(self, handle) -> bool:
        """Hand the control loop to the learned policy until it grasps or times out.

        Enables the policy node, then waits. The "did it grasp?" check is the
        missing piece -- it needs gripper feedback, which needs a trained policy
        to test against, so it currently just times out.

        Stub. The policy node owns the 30 Hz loop; this should enable it, watch
        the gripper state, and return once it closes.
        """
        timeout = self.get_parameter("grasp_timeout_s").value
        self._policy_enable_pub.publish(Bool(data=True))
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if handle.is_cancel_requested:
                    return False
                time.sleep(0.1)
            return False
        finally:
            # Always hand control back, including on cancel -- a policy left
            # enabled keeps driving the arm after the action ends.
            self._policy_enable_pub.publish(Bool(data=False))


def main(args=None) -> None:
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init(args=args)
    node = OrchestratorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
