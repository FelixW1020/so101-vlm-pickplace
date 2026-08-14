"""Hosted-VLM grounding backend -- fallback for referring expressions.

Use when the local detector abstains, or when the phrase needs actual reasoning
("the block behind the marker") rather than noun matching. At one call per
episode this costs single-digit dollars for a whole project, so it is a cheap
robustness win.

COORDINATE CONVENTION -- the thing that will bite you:

Claude works in ABSOLUTE PIXELS, and explicitly does not do well with normalized
0-1000 coordinates (unlike Gemini, whose native format IS 0-1000). The pixels it
returns are in the image *after Claude's own resize*: largest aspect-preserving
size with max edge <= 1568 px and a visual-token budget of
ceil(w/28)*ceil(h/28) <= 1568.

So we pre-resize the image ourselves to a size that survives that untouched, and
then scale the returned coordinates back to the original. Skipping this step
gives a systematic offset that looks exactly like a bad homography.
"""

from __future__ import annotations

import base64
import json
import logging
import os

import cv2
import numpy as np

from .base import Grounding

logger = logging.getLogger(__name__)

MAX_EDGE = 1568
DEFAULT_MODEL = "claude-sonnet-5"

_PROMPT = """Locate the object described as: "{phrase}"

Return ONLY a JSON object, no prose:
{{"found": true, "box": [x1, y1, x2, y2], "confidence": 0.0-1.0}}

Use ABSOLUTE PIXEL coordinates in the image as given to you.
The image is {width} x {height} pixels.

If the described object is NOT present in the image, return:
{{"found": false}}

Do not guess. A wrong location makes the robot grasp empty air, which is worse
than reporting that the object is absent."""


def _resize_for_api(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale so the API does not resize behind our back.

    Returns the resized image and the scale factor to map returned coordinates
    back to original pixels (original = returned / scale).
    """
    h, w = image_bgr.shape[:2]
    scale = min(1.0, MAX_EDGE / max(h, w))
    if scale >= 1.0:
        return image_bgr, 1.0
    resized = cv2.resize(
        image_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
    )
    return resized, scale


class ClaudeGroundingBackend:
    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        import anthropic

        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set -- export it or pass api_key= explicitly"
            )
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def ground(self, image_bgr: np.ndarray, phrase: str) -> Grounding | None:
        self._load()

        sent, scale = _resize_for_api(image_bgr)
        h, w = sent.shape[:2]
        ok, buf = cv2.imencode(".png", sent)
        if not ok:
            raise RuntimeError("failed to encode frame")

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(buf.tobytes()).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": _PROMPT.format(phrase=phrase, width=w, height=h),
                        },
                    ],
                }
            ],
        )

        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        # Models like to wrap JSON in fences even when told not to.
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("unparseable grounding response: %r", text[:200])
            return None

        if not parsed.get("found"):
            logger.info("abstain: %s reports %r absent", self.name, phrase)
            return None

        x1, y1, x2, y2 = (float(v) / scale for v in parsed["box"])
        return Grounding(
            uv=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            bbox=(x1, y1, x2, y2),
            conf=float(parsed.get("confidence", 0.5)),
            source=self.name,
        )
