from .base import BaseSegmenter
from ImputerFactory.data import ImputerConfig
from . import register_segmenter

@register_segmenter("gradient_guided")
class GradientGuidedSegmenter(BaseSegmenter):
    """Static non-uniform layout using gradients."""

    def __init__(self, config: ImputerConfig):
        super().__init__(config)
