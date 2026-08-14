"""Backend registry + the cascade policy.

Keeping construction behind a name string means the ROS node, the CLI tools and
the offline benchmark all select backends the same way, and swapping models is a
config change rather than an edit.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Grounding, GroundingBackend

logger = logging.getLogger(__name__)


def make_backend(name: str, **kwargs) -> GroundingBackend:
    if name == "gdino":
        from .gdino import GroundingDinoBackend

        return GroundingDinoBackend(**kwargs)
    if name == "claude":
        from .api_vlm import ClaudeGroundingBackend

        return ClaudeGroundingBackend(**kwargs)
    raise ValueError(f"unknown grounding backend {name!r} (have: gdino, claude)")


class CascadeGrounder:
    """Local detector first, hosted VLM only when it abstains or is unsure.

    This is the recommended runtime configuration: the detector handles the
    common colored-noun case fast and offline, and the API call -- which costs
    money and a second of latency -- only fires on the hard queries.
    """

    def __init__(
        self,
        primary: GroundingBackend,
        fallback: GroundingBackend | None = None,
        min_conf: float = 0.35,
    ):
        self.primary = primary
        self.fallback = fallback
        self.min_conf = min_conf

    def ground(self, image_bgr: np.ndarray, phrase: str) -> Grounding | None:
        result = self.primary.ground(image_bgr, phrase)
        if result is not None and result.conf >= self.min_conf:
            return result

        if self.fallback is None:
            return result

        why = "abstained" if result is None else f"conf {result.conf:.2f} < {self.min_conf}"
        logger.info("primary %s %s, escalating to %s", self.primary.name, why, self.fallback.name)
        return self.fallback.ground(image_bgr, phrase) or result
