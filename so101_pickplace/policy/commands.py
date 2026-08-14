"""Generates the LeRobot CLI invocations for this project.

These commands have a lot of surface area and most of it is easy to get subtly
wrong (mismatched --id breaks calibration lookup, a short --steps without a
matching scheduler decay means the LR never decays). Generating them from one
config keeps record/train/rollout consistent with each other.

Print them with:
    python -m so101_pickplace.policy.commands --help
    python -m so101_pickplace.policy.commands record

Requires lerobot >= 0.6.0. On 0.5.1 and earlier `lerobot-rollout` does not exist
and real-robot inference went through `lerobot-record --policy.path=...`.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field


@dataclass
class RobotSetup:
    follower_port: str = "/dev/tty.usbmodemFOLLOWER"
    leader_port: str = "/dev/tty.usbmodemLEADER"
    follower_id: str = "so101_follower"
    leader_id: str = "so101_leader"
    top_camera: int = 1
    wrist_camera: int = 0
    fps: int = 30

    hf_user: str = "FelixW1020"
    dataset_name: str = "so101_lang_pickplace"
    task: str = "pick up the red block and drop it in the bowl"

    num_episodes: int = 50
    episode_time_s: int = 30
    reset_time_s: int = 10

    def cameras_arg(self) -> str:
        return (
            "{ "
            f"top: {{type: opencv, index_or_path: {self.top_camera}, "
            f"width: 640, height: 480, fps: {self.fps}}}, "
            f"wrist: {{type: opencv, index_or_path: {self.wrist_camera}, "
            f"width: 640, height: 480, fps: {self.fps}}}"
            " }"
        )

    @property
    def repo_id(self) -> str:
        return f"{self.hf_user}/{self.dataset_name}"

    @property
    def policy_repo_id(self) -> str:
        return f"{self.hf_user}/act_{self.dataset_name}"


@dataclass
class TrainConfig:
    policy_type: str = "act"
    batch_size: int = 8
    steps: int = 20_000
    device: str = "mps"
    output_dir: str = "outputs/train/act_so101_pickplace"
    job_name: str = "act_so101_pickplace"
    wandb: bool = False
    extra: list[str] = field(default_factory=list)


def find_port() -> list[str]:
    return ["lerobot-find-port"]


def find_cameras() -> list[str]:
    return ["lerobot-find-cameras"]


def setup_motors(s: RobotSetup, which: str = "follower") -> list[str]:
    """One-time EEPROM write of motor IDs.

    Motors must be connected ONE AT A TIME, not daisy-chained, and are assigned
    backwards: gripper=6 first, down to shoulder_pan=1.
    """
    if which == "follower":
        return ["lerobot-setup-motors",
                "--robot.type=so101_follower", f"--robot.port={s.follower_port}"]
    return ["lerobot-setup-motors",
            "--teleop.type=so101_leader", f"--teleop.port={s.leader_port}"]


def calibrate(s: RobotSetup, which: str = "follower") -> list[str]:
    """The --id here MUST match every later command, or calibration is not found."""
    if which == "follower":
        return ["lerobot-calibrate", "--robot.type=so101_follower",
                f"--robot.port={s.follower_port}", f"--robot.id={s.follower_id}"]
    return ["lerobot-calibrate", "--teleop.type=so101_leader",
            f"--teleop.port={s.leader_port}", f"--teleop.id={s.leader_id}"]


def teleoperate(s: RobotSetup) -> list[str]:
    return [
        "lerobot-teleoperate",
        "--robot.type=so101_follower", f"--robot.port={s.follower_port}",
        f"--robot.id={s.follower_id}", f"--robot.cameras={s.cameras_arg()}",
        "--teleop.type=so101_leader", f"--teleop.port={s.leader_port}",
        f"--teleop.id={s.leader_id}", "--display_data=true",
    ]


def record(s: RobotSetup) -> list[str]:
    """Collect demonstrations.

    Recording keys: right-arrow / n = save and continue, left-arrow / r =
    discard and redo, ESC / q = stop and encode. On macOS the arrow keys need
    Accessibility permission; the letter keys always work.

    Structure the episodes the way LeRobot's own SO-100 pick-place reference
    dataset does -- 50 episodes over ~5 distinct object positions, 10 each --
    rather than 50 from one spot. Position diversity is what buys generalization
    to held-out placements.
    """
    return [
        "lerobot-record",
        "--robot.type=so101_follower", f"--robot.port={s.follower_port}",
        f"--robot.id={s.follower_id}", f"--robot.cameras={s.cameras_arg()}",
        "--teleop.type=so101_leader", f"--teleop.port={s.leader_port}",
        f"--teleop.id={s.leader_id}",
        f"--dataset.repo_id={s.repo_id}",
        f"--dataset.num_episodes={s.num_episodes}",
        f"--dataset.single_task={s.task}",
        f"--dataset.episode_time_s={s.episode_time_s}",
        f"--dataset.reset_time_s={s.reset_time_s}",
        "--display_data=true",
    ]


def train(s: RobotSetup, t: TrainConfig) -> list[str]:
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={s.repo_id}",
        f"--policy.type={t.policy_type}",
        f"--output_dir={t.output_dir}",
        f"--job_name={t.job_name}",
        f"--policy.device={t.device}",
        f"--batch_size={t.batch_size}",
        f"--steps={t.steps}",
        # Without this the LR schedule is sized for the default step count and
        # never decays on a shortened run.
        f"--policy.scheduler_decay_steps={t.steps}",
        f"--policy.repo_id={s.policy_repo_id}",
        f"--wandb.enable={'true' if t.wandb else 'false'}",
    ]
    return cmd + t.extra


def rollout(s: RobotSetup, checkpoint: str | None = None, duration: int = 60) -> list[str]:
    """Real-robot deployment. NOTE: lerobot-eval is gym-only, not this."""
    return [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={checkpoint or s.policy_repo_id}",
        "--robot.type=so101_follower", f"--robot.port={s.follower_port}",
        f"--robot.id={s.follower_id}", f"--robot.cameras={s.cameras_arg()}",
        f"--task={s.task}", f"--duration={duration}", "--display_data=true",
    ]


def rollout_eval(s: RobotSetup, n_episodes: int = 20) -> list[str]:
    """Episodic rollout for the evaluation sweep.

    `episodic` gives indexed episodes with reset phases between them, which is
    what you want for held-out placements. It does NOT record success/failure --
    that has to be tallied separately, see so101_pickplace.evaluation.
    """
    return [
        "lerobot-rollout",
        "--strategy.type=episodic",
        f"--policy.path={s.policy_repo_id}",
        "--robot.type=so101_follower", f"--robot.port={s.follower_port}",
        f"--robot.id={s.follower_id}", f"--robot.cameras={s.cameras_arg()}",
        f"--dataset.repo_id={s.hf_user}/eval_{s.dataset_name}",
        f"--dataset.single_task={s.task}",
        f"--dataset.num_episodes={n_episodes}",
        f"--dataset.episode_time_s={s.episode_time_s}",
        "--dataset.reset_time_s=15",
    ]


STAGES = {
    "find-port": lambda s, t: find_port(),
    "find-cameras": lambda s, t: find_cameras(),
    "setup-motors-follower": lambda s, t: setup_motors(s, "follower"),
    "setup-motors-leader": lambda s, t: setup_motors(s, "leader"),
    "calibrate-follower": lambda s, t: calibrate(s, "follower"),
    "calibrate-leader": lambda s, t: calibrate(s, "leader"),
    "teleoperate": lambda s, t: teleoperate(s),
    "record": lambda s, t: record(s),
    "train": lambda s, t: train(s, t),
    "rollout": lambda s, t: rollout(s),
    "eval": lambda s, t: rollout_eval(s),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Print the LeRobot command for a stage")
    ap.add_argument("stage", choices=sorted(STAGES) + ["all"])
    ap.add_argument("--follower-port", default=RobotSetup.follower_port)
    ap.add_argument("--leader-port", default=RobotSetup.leader_port)
    ap.add_argument("--device", default=TrainConfig.device, choices=["mps", "cuda", "cpu"])
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    args = ap.parse_args()

    setup = RobotSetup(follower_port=args.follower_port, leader_port=args.leader_port)
    train_cfg = TrainConfig(device=args.device, steps=args.steps)

    stages = sorted(STAGES) if args.stage == "all" else [args.stage]
    for name in stages:
        print(f"\n# {name}")
        print(" \\\n    ".join(shlex.quote(p) for p in STAGES[name](setup, train_cfg)))


if __name__ == "__main__":
    main()
