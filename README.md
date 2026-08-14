# SO-101 Language-Grounded Pick & Place

Two-stage manipulation stack on a 6-DoF [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm.
A vision-language model grounds a natural-language object target in the camera
frame; an imitation-learned policy emits the joint trajectory that grasps it and
releases it into a container.

```
  "put the red block in the bowl"
              │
              ▼
   ┌──────────────────────┐
   │  STAGE 1: GROUNDING  │   Grounding DINO (local) → hosted VLM on low confidence
   │  phrase + frame      │   runs ONCE PER EPISODE, not per control step
   │       → pixel        │
   └──────────┬───────────┘
              │  homography (table plane, no depth camera needed)
              ▼
        table (x, y) mm
              │
   ┌──────────┴───────────┐
   │   SCRIPTED APPROACH  │   coarse, large motion, IK from the URDF
   │   → hover 80mm above │
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │  STAGE 2: POLICY     │   ACT, fine-tuned on teleoperated demos
   │  descend + grasp     │   30 Hz joint targets
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │  SCRIPTED RETREAT    │   lift → container → release → home
   └──────────────────────┘
```

**Status: scaffold.** Interfaces and wiring are real; the grasp policy is a stub
until demonstrations are collected. Nothing here needs the arm plugged in to
import, and the whole loop runs end-to-end with `policy_enabled:=false` — the arm
just stops at the pre-grasp pose, which is exactly what you want for verifying
stage 1 and the calibration first.

---

## Three constraints that shaped the design

**1. Docker on macOS cannot see the USB servo bus.** Docker Desktop runs its
daemon in a Linux VM with no USB passthrough — closed as won't-fix upstream
([docker/for-mac#5263](https://github.com/docker/for-mac/issues/5263)), and the
USB/IP escape hatch needs a macOS server still blocked on Apple driver approval.
So a **native macOS process owns `/dev/tty.usbmodem*`** and ROS 2 reaches it over
ZMQ/TCP. That seam is `so101_pickplace/control/so101_bridge.py` ↔
`so101_ros/bridge_node.py`.

> On a Linux box this problem disappears: use
> [`feetech_ros2_driver`](https://github.com/ros-physical-ai/feetech_ros2_driver)
> with `hardware_type:=real` and delete the bridge. Nothing else changes.

**2. ACT has no language input.** There is no tokenizer in its config, and
LeRobot's own docs note `--task` "can be skipped for ACT". In a two-stage design
that is the point rather than a limitation — the VLM carries the language, and
the policy only executes on an already-grounded target. It is also the only
option that trains on an M3 Pro (~6–14 h on MPS for ~50 episodes; SmolVLA has no
MPS row in LeRobot's compute table at all).

**3. Grounding is not in the control loop.** Fixed camera, static object: ground
once at t=0, then execute. A 1–3 s VLM call is free. Do not build a 30 Hz
grounding loop.

---

## Layout

```
so101_pickplace/            host-side Python (runs natively on macOS)
  perception/grounding/     stage 1 — swappable backends behind one interface
  perception/calibration/   pixel → table-plane homography, teach-by-touch
  control/                  serial bridge, wire protocol, scripted primitives, IK
  policy/                   stage 2 — ACT inference + LeRobot CLI generation
  evaluation/               success-rate tally (LeRobot has no real-robot metric)

ros2_ws/src/
  so101_msgs/               GroundTarget.srv, PickAndPlace.action
  so101_ros/                bridge / grounding / policy / orchestrator nodes
  so101_bringup/            launch + params

docker/                     ros:jazzy-ros-base, arm64-native
tests/                      geometry + protocol, no hardware required
```

---

## Getting started

```bash
# 0. hardware repo (meshes + URDF, ~428 MB, not committed)
./scripts/fetch_hardware.sh

# 1. host-side environment (Python 3.12; lerobot requires >=3.12)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,grounding,robot]"
brew install ffmpeg          # lerobot needs it for episode video encoding

# 2. tests — these pass with no arm, no camera, no GPU
pytest -q

# 3. arm setup. Print every command with real ports filled in:
python -m so101_pickplace.policy.commands all \
    --follower-port /dev/tty.usbmodemXXXX --leader-port /dev/tty.usbmodemYYYY
```

Run those in order: `find-port` → `setup-motors-*` → `calibrate-*` →
`teleoperate`. The `--id` you calibrate with must match every later command or
LeRobot will not find the calibration.

```bash
# 4. camera → table calibration (teach by touch, ~12–20 points)
python -m so101_pickplace.perception.calibration.collect --camera 0

# 5. ROS 2 stack
docker compose -f docker/docker-compose.yml up --build
# and, in a separate NATIVE terminal (not in the container):
python -m so101_pickplace.control.so101_bridge \
    --port /dev/tty.usbmodemXXXX --id my_follower_arm
```

Connect [Foxglove Studio](https://foxglove.dev/) to `ws://localhost:8765`.
RViz over XQuartz on Apple Silicon falls back to software rendering
([ros2/rviz#929](https://github.com/ros2/rviz/issues/929)) and crawls on the
SO-101's multi-megabyte collision meshes.

```bash
# 6. try grounding alone, no motion
ros2 action send_goal /pick_and_place so101_msgs/action/PickAndPlace \
    "{phrase: 'the red block', dry_run: true}"
```

---

## Still to do

- [ ] **Measure `WRIST_OFFSET_DEG`** — jog until the gripper points straight down and
      read `wrist_flex`. Link lengths are already read from the URDF, and this one
      constant is all that stands between the IK and a closed workspace: without it
      top-down targets overshoot the ±95° wrist limit by 5–16°, and `pre_grasp_ik`
      raises rather than silently clamping
- [ ] Record real `HOME_POSE` / `DROP_POSE` by jogging the arm (`control/scripted.py`)
- [ ] Collect ~50 demonstrations, ~10 each across 5 object positions
- [ ] Train ACT, wire the policy handoff in `orchestrator_node._run_policy_grasp`
- [ ] Build the held-out grounding test set and run the eval sweep
- [ ] Own `so101_description` package (the vendored URDF is not ROS-ready:
      relative mesh paths, ROS 1 `<transmission>` tags, no `<ros2_control>` block)

## Notes

Demonstrations must be recorded **starting from the same scripted pre-grasp pose**
the orchestrator produces. If the handoff lands the arm somewhere the policy
never saw in training, the grasp fails for reasons that look like a bad policy
but aren't.

The two calibration conventions in the upstream repo differ by ~90°:
`so101_new_calib` (mid-range zero, matches current LeRobot) and `so101_old_calib`
(fully-extended zero). This project assumes **new_calib** throughout.
