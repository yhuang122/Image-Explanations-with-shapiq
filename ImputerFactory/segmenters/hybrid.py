from .base import BaseSegmenter
from ImputerFactory.data import ImputerConfig


class HybridSegmenter(BaseSegmenter):
    """Gradient priors paired with adaptive segmenting."""

    def __init__(self, config: ImputerConfig):
        super().__init__(config)
