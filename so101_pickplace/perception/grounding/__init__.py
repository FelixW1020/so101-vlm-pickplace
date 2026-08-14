"""Stage-1 grounding: natural language + image -> target pixel."""

from .base import Grounding, GroundingBackend, select_by_spatial_qualifier
from .registry import CascadeGrounder, make_backend

__all__ = [
    "CascadeGrounder",
    "Grounding",
    "GroundingBackend",
    "make_backend",
    "select_by_spatial_qualifier",
]
