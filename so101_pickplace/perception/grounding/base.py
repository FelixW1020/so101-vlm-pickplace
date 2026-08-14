"""Stage-1 grounding interface.

This is the contract every backend normalizes into. Freeze it early: the whole
downstream stack (homography -> table XY -> pre-grasp pose) consumes only a
`Grounding`, so backends stay swappable without touching control code.

Two rules that matter:

1.  `uv` and `bbox` are ALWAYS in *original* image pixel coordinates. Every model
    has its own internal resize/pad convention (OWLv2 pads to square before a
    960x960 resize; Qwen2.5-VL returns absolute pixels of a smart-resized image;
    Qwen3-VL returns 0-1000 normalized; Claude returns absolute pixels of its own
    resized image). Each backend is responsible for undoing its own convention.
    Getting this wrong produces a systematic offset that looks like bad
    calibration and will waste days.

2.  A backend MUST be able to abstain by returning None. "The referent is not in
    this scene" is a real answer, and a detector that always returns its
    best-scoring box will confidently grasp empty air.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Grounding:
    """A single grounded referent, in original-image pixel space."""

    uv: tuple[float, float]
    """Contact pixel -- where we intend to grasp."""

    bbox: tuple[float, float, float, float] | None
    """(x1, y1, x2, y2), original image pixels."""

    conf: float
    """Backend confidence in [0, 1]. Thresholding is the caller's job."""

    source: str
    """Backend name, e.g. "gdino" | "claude". Logged for every call."""

    mask: np.ndarray | None = field(default=None, repr=False)
    """Optional boolean mask, same HxW as the input image."""

    def contact_point(self) -> tuple[float, float]:
        """Pixel to actually reach for.

        Uses the BOTTOM edge midpoint of the box, not the centre. On a table the
        object's footprint is what we grasp, and for anything with height the
        centre of the box projects behind the footprint -- a 40 mm cube viewed
        30 degrees off-nadir lands ~23 mm off if you use the centre. The bottom
        edge is a much better proxy for where the object meets the table.
        """
        if self.bbox is None:
            return self.uv
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


class GroundingBackend(Protocol):
    """Structural type for a stage-1 backend."""

    name: str

    def ground(self, image_bgr: np.ndarray, phrase: str) -> Grounding | None:
        """Localize `phrase` in `image_bgr`, or return None to abstain."""
        ...


def select_by_spatial_qualifier(
    candidates: list[Grounding], phrase: str
) -> Grounding | None:
    """Resolve simple spatial words over multiple detections.

    Detector-style models (Grounding DINO, OWLv2) are phrase detectors, not
    referring-expression resolvers -- they largely ignore "leftmost"/"nearest".
    Rather than reaching for a 7B VLM, do the arithmetic here: get all the boxes
    for the noun, then sort. Three lines of Python beats a language model for
    "which of these is furthest left".

    Anything beyond these qualifiers should fall through to a real VLM backend.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    p = phrase.lower()
    cx = lambda g: g.contact_point()[0]  # noqa: E731
    cy = lambda g: g.contact_point()[1]  # noqa: E731

    if "leftmost" in p or "left " in p:
        return min(candidates, key=cx)
    if "rightmost" in p or "right " in p:
        return max(candidates, key=cx)
    # Image y grows downward; "nearest"/"front" == closer to the bottom of frame.
    if "nearest" in p or "closest" in p or "front" in p:
        return max(candidates, key=cy)
    if "farthest" in p or "furthest" in p or "back" in p:
        return min(candidates, key=cy)

    return max(candidates, key=lambda g: g.conf)
