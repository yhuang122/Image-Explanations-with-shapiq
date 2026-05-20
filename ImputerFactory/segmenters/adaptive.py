from .base import BaseSegmenter
from ImputerFactory.data import ImputerConfig
from . import register_segmenter

@register_segmenter("adaptive")
class AdaptiveSegmenter(BaseSegmenter):
    """Dynamic, coarse-to-fine scoring-driven spatial division."""

    def __init__(self, config: ImputerConfig):
        super().__init__(config)
