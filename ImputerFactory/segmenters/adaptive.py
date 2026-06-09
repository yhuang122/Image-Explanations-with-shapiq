from .base import BaseSegmenter
from ..data import SegmenterConfig
from . import register_segmenter

@register_segmenter("adaptive")
class AdaptiveSegmenter(BaseSegmenter):
    """Dynamic, coarse-to-fine scoring-driven spatial division."""

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)
