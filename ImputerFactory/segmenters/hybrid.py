from .base import BaseSegmenter
from ..data import SegmenterConfig

from . import register_segmenter

@register_segmenter("hybrid")
class HybridSegmenter(BaseSegmenter):
    """Gradient priors paired with adaptive segmenting."""

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)
