from .base import BaseSegmenter
from ImputerFactory.data import ImputerConfig


class AdaptiveSegmenter(BaseSegmenter):
    """Dynamic, coarse-to-fine scoring-driven spatial division."""

    def __init__(self, config: ImputerConfig):
        super().__init__(config)
