"""
Masker registry — maps string names to Masker classes.

Registered keys: ``"vision_mean"``, ``"vision_blur"``, ``"text_attn"``,
``"crossmodal_mean"``, ``"crossmodal_blur"``, ``"crossmodal_gaussian"``,
``"attention"``.

To add a new masker, decorate the class with ``@register_masker("name")``.
The factory will resolve it via ``MaskerConfig.strategy``.
"""

from .base import BaseMasker

# ── Registry ───────────────────────────────────────────────────────────

_MASKER_REGISTRY: dict[str, type] = {}


def register_masker(name: str):
    """Decorator: register a Masker class under a string key."""

    def _register(cls):
        key = name.lower()
        if key not in _MASKER_REGISTRY:
            _MASKER_REGISTRY[key] = cls
        return cls

    return _register


def get_masker(name: str) -> type:
    """Look up a Masker class by name. Must be a non-None string."""
    if name is None:
        raise TypeError("masker name must not be None. "
                        "The factory should resolve the default before calling get_masker.")
    key = name.lower()
    if key not in _MASKER_REGISTRY:
        raise KeyError(
            f"Unknown masker '{name}'. "
            f"Registered: {list(_MASKER_REGISTRY.keys())}"
        )
    return _MASKER_REGISTRY[key]


# ── Built-in maskers ───────────────────────────────────────────────────

from .vision_mean import VisionMeanMasker
from .vision_blur import VisionBlurMasker
from .text_attention import TextAttentionMasker
from .crossmodal_mean import CrossModalMeanMasker
from .crossmodal_blur import CrossModalBlurMasker
from .crossmodal_gaussian import CrossModalGaussianMasker
from .attention import AttentionMasker

