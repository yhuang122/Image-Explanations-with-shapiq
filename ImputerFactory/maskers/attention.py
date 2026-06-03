from typing import Optional

from .base import BaseMasker
from . import register_masker
from ImputerFactory.data import MaskerConfig


@register_masker("attention")
class AttentionMasker(BaseMasker):
    """Intercepts self-attention mechanism handling negative-infinity mask matrices."""

    def __init__(self, config: Optional[MaskerConfig] = None):
        super().__init__(config)
