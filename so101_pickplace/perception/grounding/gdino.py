"""Grounding DINO backend -- the default stage-1 model.

Chosen over the VLM options because the vocabulary here is small and
colored-noun shaped ("the red block", "the blue marker"), which is exactly the
regime where a detector beats an MLLM: tighter boxes, deterministic output, no
JSON parsing failures, no hallucinated objects. Apache-2.0 and ungated, so a
grader can reproduce it without an API key.

Runs ONCE PER EPISODE, not per control step -- the camera is fixed and the object
is static until we touch it. A 1-3 s call is fine. Do not build a 30 Hz loop.

MPS note: use the `transformers` port, not the original IDEA-Research repo. The
latter needs a custom CUDA MultiScaleDeformableAttention kernel; the HF port
falls back to pure PyTorch, which is slower but actually runs on Apple Silicon.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Grounding, select_by_spatial_qualifier

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_detector_prompt(phrase: str) -> str:
    """Grounding DINO's text format is load-bearing.

    Queries must be lowercase and period-terminated. A missing trailing period
    is the single most common cause of "it returns nothing". We also strip
    spatial qualifiers, which the detector ignores anyway -- they get resolved
    downstream in select_by_spatial_qualifier().
    """
    p = phrase.lower().strip()
    for qualifier in (
        "leftmost", "rightmost", "nearest", "closest", "farthest", "furthest",
        "the left", "the right", "in front", "at the back",
    ):
        p = p.replace(qualifier, "")
    p = " ".join(p.split())
    if not p.endswith("."):
        p += "."
    return p


class GroundingDinoBackend:
    name = "gdino"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ):
        self.model_id = model_id
        self.device = device or _pick_device()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None
        self._processor = None

    def _load(self):
        """Lazy load -- importing torch costs seconds and the ROS nodes that
        merely reference this class should not pay for it."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        logger.info("loading %s on %s", self.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        self._model.eval()
        self._torch = torch

    def ground(self, image_bgr: np.ndarray, phrase: str) -> Grounding | None:
        self._load()
        from PIL import Image

        # cv2 gives BGR; every HF vision model expects RGB.
        image_rgb = Image.fromarray(image_bgr[:, :, ::-1])
        prompt = _to_detector_prompt(phrase)

        inputs = self._processor(images=image_rgb, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            # Original (H, W) -- post-processing rescales boxes back for us.
            target_sizes=[image_rgb.size[::-1]],
        )[0]

        candidates: list[Grounding] = []
        for box, score in zip(results["boxes"], results["scores"], strict=False):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            candidates.append(
                Grounding(
                    uv=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    bbox=(x1, y1, x2, y2),
                    conf=float(score),
                    source=self.name,
                )
            )

        if not candidates:
            logger.info("abstain: no detection over threshold for %r", phrase)
            return None

        return select_by_spatial_qualifier(candidates, phrase)
