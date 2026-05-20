"""
CrossModalCompositeMasker — Composite masker for Vision-Language Models.

Follows the Composite Pattern: internally instantiates two atomic maskers
(VisionMeanMasker + TextAttentionMasker) and delegates image/text occlusion
to each respectively. Owns no low-level tensor math itself.
"""

from .base import BaseMasker
from .vision_mean import VisionMeanMasker
from .text_attention import TextAttentionMasker
from ImputerFactory.data import PhysicalMask, ProcessorOutput


class CrossModalCompositeMasker(BaseMasker):
    """
    Cross-modal occlusion orchestrator for VLMs.

    Delegates:
        - Image occlusion → VisionMeanMasker
        - Text occlusion  → TextAttentionMasker

    The composite itself performs no element-wise operations.
    """

    def __init__(self):
        self._vision_masker = VisionMeanMasker()
        self._text_masker = TextAttentionMasker()

    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        # 1. Delegate image occlusion to VisionMeanMasker
        masked = self._vision_masker.apply(processor_output, physical_mask)

        # 2. Delegate text occlusion to TextAttentionMasker (pixel_values pass-through)
        masked = self._text_masker.apply(masked, physical_mask)

        return masked
