"""
Segmenter registry — maps string names to Segmenter classes.

To add a new segmenter, decorate the class with @register_segmenter("name").
The factory will resolve it via config.segmenter.
"""

from .base import BaseSegmenter

# ── Registry ───────────────────────────────────────────────────────────

_SEGMENTER_REGISTRY: dict[str, type] = {}


def register_segmenter(name: str):
    """Decorator: register a Segmenter class under a string key."""

    def _register(cls):
        key = name.lower()
        if key not in _SEGMENTER_REGISTRY:
            _SEGMENTER_REGISTRY[key] = cls
        return cls

    return _register


def get_segmenter(name: str) -> type:
    """Look up a Segmenter class by name. Must be a non-None string."""
    if name is None:
        raise TypeError("segmenter name must not be None. "
                        "The factory should resolve the default before calling get_segmenter.")
    key = name.lower()
    if key not in _SEGMENTER_REGISTRY:
        raise KeyError(
            f"Unknown segmenter '{name}'. "
            f"Registered: {list(_SEGMENTER_REGISTRY.keys())}"
        )
    return _SEGMENTER_REGISTRY[key]


# ── Built-in segmenters ────────────────────────────────────────────────

from .patch import PatchSegmenter
from .slic import SLICSegmenter
