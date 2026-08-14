"""Stage-2 policy inference -- load a trained checkpoint, emit joint targets.

Runs NATIVELY on macOS, not in the container: only the native process gets MPS.
A container on Apple Silicon sees CPU only, which is tolerable for ACT and not
tolerable for anything larger.

Why ACT and not a language-conditioned VLA: ACT has no text input at all (no
tokenizer in its config -- LeRobot's own docs note `--task` "can be skipped for
ACT"). In this two-stage design that is fine and in fact the point -- the VLM
carries the language, and the policy only has to execute on an already-grounded
target. It is also the only option that trains on an M3 Pro: roughly 6-14 h on
MPS for ~50 episodes, where SmolVLA has no MPS row in LeRobot's compute table at
all.

If a language-conditioned baseline is wanted later, SmolVLA drops in behind the
same `select_action` interface -- but it needs cloud training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ACTPolicyRunner:
    """Thin wrapper over a trained LeRobot ACT checkpoint.

    ACT predicts a chunk of future actions per forward pass (chunk_size 100 by
    default). LeRobot's ACTPolicy handles the queue internally -- select_action()
    only runs the network when the queue is empty -- so calling it every tick is
    correct and cheap.
    """

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.checkpoint = str(checkpoint)
        self.device = device or _pick_device()
        self._policy = None
        self._preprocess = None
        self._postprocess = None

    def load(self) -> None:
        if self._policy is not None:
            return
        from lerobot.policies.act.modeling_act import ACTPolicy

        logger.info("loading ACT checkpoint %s on %s", self.checkpoint, self.device)
        self._policy = ACTPolicy.from_pretrained(self.checkpoint)
        self._policy.to(self.device)
        self._policy.eval()
        self._policy.reset()

        # Normalization moved OUT of policy weights into an external processor
        # pipeline (lerobot PR #1452). A checkpoint from before that change needs
        # migrate_policy_normalization.py before it will load correctly here.
        try:
            from lerobot.processor import make_default_processors

            self._preprocess, self._postprocess = make_default_processors()
        except ImportError:
            logger.warning("processor pipeline unavailable; assuming baked-in normalization")

    def reset(self) -> None:
        """Clear the action queue. Call between episodes -- otherwise the first
        actions of a new episode are leftovers aimed at the previous object."""
        if self._policy is not None:
            self._policy.reset()

    def select_action(
        self,
        joint_positions: list[float],
        images: dict[str, np.ndarray],
    ) -> list[float]:
        """One control step.

        Args:
            joint_positions: current state, JOINT_NAMES order.
            images: camera name -> HxWx3 BGR frame. Keys must match the camera
                names used at record time ("top", "wrist") or the policy will
                silently receive the wrong view.

        Returns the commanded joint positions, same order and units.
        """
        self.load()
        import torch

        obs: dict[str, torch.Tensor] = {
            "observation.state": torch.tensor(
                joint_positions, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        }
        for cam_name, frame_bgr in images.items():
            rgb = frame_bgr[:, :, ::-1].copy()
            tensor = torch.from_numpy(rgb).to(self.device)
            # HWC uint8 -> BCHW float in [0, 1]
            tensor = tensor.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            obs[f"observation.images.{cam_name}"] = tensor

        if self._preprocess is not None:
            obs = self._preprocess(obs)

        with torch.no_grad():
            action = self._policy.select_action(obs)

        if self._postprocess is not None:
            action = self._postprocess(action)

        return action.squeeze(0).cpu().numpy().tolist()
