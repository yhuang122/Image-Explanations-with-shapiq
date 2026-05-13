from .base import BaseSegmenter
from ImputerFactory.data import ImputerConfig


class GradientGuidedSegmenter(BaseSegmenter):
    """Static non-uniform layout using gradients."""

    def __init__(self, config: ImputerConfig):
        super().__init__(config)
