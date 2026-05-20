from .base import BaseSegmenter
from ImputerFactory.data import ImputerConfig
from . import register_segmenter

@register_segmenter("slic")
class SLICSegmenter(BaseSegmenter):
    """Perceptual superpixels for CNNs (Using skimage SLIC)."""

    def __init__(self, config: ImputerConfig):
        super().__init__(config)
