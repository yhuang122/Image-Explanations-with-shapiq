from .base import BaseMasker
from . import register_masker


@register_masker("attention")
class AttentionMasker(BaseMasker):
    """Intercepts self-attention mechanism handling negative-infinity mask matrices."""
    pass
