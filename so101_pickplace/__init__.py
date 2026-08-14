"""Two-stage language-grounded pick-and-place on a 6-DoF SO-101 arm.

    stage 1  natural language + camera frame -> grounded target  (VLM)
    stage 2  grounded target                 -> grasp            (imitation learning)

Import layout is deliberately lazy: torch, transformers and lerobot are only
imported inside the functions that need them, so the geometry and orchestration
code stays importable on a machine with no arm, no camera and no GPU.
"""

__version__ = "0.1.0"
