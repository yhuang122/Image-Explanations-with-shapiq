from .base import BaseSegmenter
from ImputerFactory.data import ImputerConfig


class SLICSegmenter(BaseSegmenter):
    """Perceptual superpixels for CNNs (Using skimage SLIC)."""

    def __init__(self, config: ImputerConfig):
        super().__init__(config)
